"""작업 폴더 백엔드. 파일 도구는 작업 폴더에 갇히지만(virtual_mode) 셸은 격리되지 않는다. 셸의 안전장치는 승인이다.

0.7.32 의 LocalShellBackend.execute 는 subprocess.run(shell=True) 라 Windows 에서 cmd.exe 로 돌고,
타임아웃 때 자식 프로세스가 남으며, 출력을 로캘 코드페이지(cp949)로 디코드한다. 그래서 execute 를 바꾼다.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from code_agent.environment import is_store_stub, trusted_tool, uv_python_find

ARTIFACTS_ROUTE = "/.code-agent-artifacts/"  # 큰 도구 결과·대화 기록·blob 오프로드가 모두 이 아래로 간다
SECRET_ENV = re.compile(r"_KEY$|TOKEN|SECRET|PASSWORD", re.IGNORECASE)  # *_API_KEY, LANGFUSE_PUBLIC_KEY 등
# code-agent 자신의 venv. 물려주면 셸의 python 이 deepagents 가 깔린 도구 venv 가 되고 pip install 이 그것을 오염시킨다
TOOL_VENV_ENV = ("VIRTUAL_ENV", "UV", "UV_RUN_RECURSION_DEPTH", "PYTHONHOME")
# 진행률 표시를 끈다. PS 5.1 은 진행률을 CLIXML 로 stderr 에 섞어 내보낼 수 있다
_PS_UTF8 = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; $ProgressPreference = 'SilentlyContinue'; "
if sys.platform == "win32":
    import ctypes

    _OEM_ENCODING = f"cp{ctypes.windll.kernel32.GetOEMCP()}"
else:
    _OEM_ENCODING = "utf-8"


def _tool_venv_scripts() -> str | None:
    """code-agent 가 venv 에서 돌면 그 venv 의 실행 파일 폴더."""
    if sys.prefix == sys.base_prefix:
        return None
    return str(Path(sys.prefix, "Scripts" if sys.platform == "win32" else "bin"))


def shell_env(base: Mapping[str, str] | None = None, *, tool_scripts: str | None = None,
              find_python: Callable[[Mapping[str, str]], str | None] = uv_python_find) -> dict[str, str]:
    """셸에 넘길 환경. 부모 환경에서 비밀값과 code-agent 의 도구 venv 를 뺀다.

    빈 env 를 주면 PATH·SYSTEMROOT 가 없어 셸이 제대로 뜨지 않으므로 나머지는 물려준다. 도구 venv 를 빼서 python 이
    없거나 스토어 스텁만 남으면, uv 가 관리하는 인터프리터(PEP 668 표식이 있어 pip install 이 막힌다)를 PATH 앞에 둔다.
    tool_scripts 는 PATH 에서 뺄 도구 venv 의 실행 파일 폴더다(기본: 지금 venv 에서 돌면 그 폴더).
    """
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if not SECRET_ENV.search(k) and k not in TOOL_VENV_ENV}
    scripts = tool_scripts if tool_scripts is not None else _tool_venv_scripts()
    if scripts:
        drop = os.path.normcase(os.path.normpath(scripts))
        entries = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(e for e in entries if not e or os.path.normcase(os.path.normpath(e)) != drop)
    python = trusted_tool(shutil.which("python", path=env.get("PATH", "")))
    if python is None or is_store_stub(python):
        found = find_python(env)
        if found:
            env["PATH"] = os.pathsep.join([str(Path(found).parent), env.get("PATH", "")])
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", CHERE_INVOKING="1")
    return env


@functools.cache
def _process_shell_env() -> dict[str, str]:
    return shell_env()


def process_shell_env() -> dict[str, str]:
    """이 프로세스의 셸 환경. uv python find 를 부르므로 한 번만 계산한다. 셸 백엔드와 환경 탐지가 같은 것을 쓴다."""
    return dict(_process_shell_env())


def find_git_bash() -> Path | None:
    """git.exe 의 상위 폴더 중 bin/bash.exe 를 가진 Git 루트를 찾는다. System32 의 bash.exe 는 WSL 이라 쓰지 않는다."""
    git = shutil.which("git")
    roots = [*Path(git).resolve().parents] if git else []
    roots += [Path(p, "Git") for p in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")) if p]
    return next((root / "bin" / "bash.exe" for root in roots if (root / "bin" / "bash.exe").is_file()), None)


def resolve_shell(preference: str = "") -> tuple[str, list[str]]:
    """(셸 이름, 명령 앞에 붙일 argv). 로그인 셸(-l)은 프로필을 읽어 느리고 cwd 를 바꿀 수 있어 쓰지 않는다."""
    if sys.platform != "win32":
        return "bash", ["/bin/bash", "-c"]
    if preference in ("", "bash"):
        bash = find_git_bash()
        if bash:
            return "bash", [str(bash), "-c"]
        if preference == "bash":
            raise RuntimeError("Git Bash(bash.exe)를 찾지 못했습니다. CODE_AGENT_SHELL=powershell 로 바꾸세요.")
    return "powershell", ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]


class WindowsShellBackend(LocalShellBackend):
    """Git Bash 또는 PowerShell 로 실행하고, 타임아웃 때 프로세스 트리를 끝내고, 출력을 UTF-8 로 읽는다."""

    def __init__(self, root_dir: str | Path, *, shell: str = "", timeout: int = 120,
                 env: Mapping[str, str] | None = None) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True, timeout=timeout,
                         env=process_shell_env() if env is None else dict(env), inherit_env=False)
        self.shell, self._argv = resolve_shell(shell)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1, truncated=False)
        limit = timeout if timeout is not None else self._default_timeout
        if limit <= 0:
            raise ValueError(f"timeout must be positive, got {limit}")
        script = _PS_UTF8 + command if self.shell == "powershell" else command
        try:
            proc = subprocess.Popen(
                [*self._argv, script], cwd=str(self.cwd), env=self._env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=sys.platform != "win32",
            )
        except OSError as e:
            return ExecuteResponse(output=f"Error executing command ({type(e).__name__}): {e}", exit_code=1,
                                   truncated=False)
        try:
            stdout, stderr = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            return ExecuteResponse(
                output=f"Error: Command timed out after {limit} seconds. For long-running commands, "
                       "re-run using the timeout parameter.",
                exit_code=124, truncated=False,
            )
        return self._response(_decode(stdout), _decode(stderr), proc.returncode)

    def _response(self, stdout: str, stderr: str, code: int) -> ExecuteResponse:
        """기본 구현과 같은 형식: stderr 줄에 [stderr] 를 붙이고, 0 이 아닌 종료 코드를 덧붙이고, 크기를 자른다."""
        parts = [stdout] if stdout else []
        if stderr:
            parts.extend(f"[stderr] {line}" for line in stderr.strip().split("\n"))
        output = "\n".join(parts) if parts else "<no output>"
        truncated = False
        data = output.encode("utf-8")
        if len(data) > self._max_output_bytes:
            output = data[: self._max_output_bytes].decode("utf-8", errors="ignore")
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True
        if code != 0:
            output = f"{output.rstrip()}\n\nExit code: {code}"
        return ExecuteResponse(output=output, exit_code=code, truncated=truncated)


def _decode(data: bytes) -> str:
    """UTF-8 이 아니면 OEM 코드페이지로 읽는다. tasklist 같은 Windows 기본 도구는 파이프에 cp949 로 쓴다."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode(_OEM_ENCODING, errors="replace")
    return text.replace("\r\n", "\n")


def _kill_tree(proc: subprocess.Popen) -> None:
    """Windows 에는 프로세스 그룹 종료가 없어 taskkill /T 로 자식까지 끝낸다."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def make_backend(workspace: Path, *, artifacts_dir: Path, shell: str = "") -> CompositeBackend:
    """오프로드를 ~/.code-agent/artifacts 로 보내 작업 폴더를 더럽히지 않는다. execute 는 항상 default 로 간다."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return CompositeBackend(
        default=WindowsShellBackend(workspace, shell=shell),
        routes={ARTIFACTS_ROUTE: FilesystemBackend(root_dir=artifacts_dir, virtual_mode=True)},
        artifacts_root=ARTIFACTS_ROUTE.rstrip("/"),
    )
