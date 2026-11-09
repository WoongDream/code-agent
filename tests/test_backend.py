"""셸 백엔드: 셸 선택, 종료 코드, 한글, 타임아웃, 비밀값 제거, 경로 탈출, 오프로드 위치."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from deepagents.middleware.filesystem import FilesystemMiddleware

from code_agent.backend import (
    ARTIFACTS_ROUTE,
    WindowsShellBackend,
    find_git_bash,
    make_backend,
    shell_env,
)

SHELLS = ["bash", "powershell"] if sys.platform == "win32" else ["bash"]
needs_bash = pytest.mark.skipif(sys.platform == "win32" and find_git_bash() is None,
                                reason="Git Bash 없음")


def run_python(shell: str, script: str) -> str:
    """셸마다 다른 따옴표 규칙으로 현재 인터프리터를 실행하는 명령."""
    exe = Path(sys.executable).as_posix()
    return f"'{exe}' {script}" if shell == "bash" else f"& '{exe}' {script}"


def alive(pid: int) -> bool:
    if sys.platform == "win32":  # tasklist 출력은 OEM 코드페이지라 바이트로 비교한다
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                             check=False).stdout
        return str(pid).encode() in out
    try:
        import os
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture(params=SHELLS)
def shell_backend(request, tmp_path: Path) -> WindowsShellBackend:
    if request.param == "bash" and sys.platform == "win32" and find_git_bash() is None:
        pytest.skip("Git Bash 없음")
    return WindowsShellBackend(tmp_path, shell=request.param, timeout=30)


def test_echo_and_exit_code(shell_backend: WindowsShellBackend) -> None:
    ok = shell_backend.execute("echo hello")
    assert (ok.output.strip(), ok.exit_code) == ("hello", 0)
    failed = shell_backend.execute("echo oops; exit 3")
    assert failed.exit_code == 3
    assert failed.output.endswith("Exit code: 3")


def test_stderr_is_prefixed(shell_backend: WindowsShellBackend) -> None:
    result = shell_backend.execute(
        run_python(shell_backend.shell, '-c "import sys; sys.stderr.write(\'bad\\n\')"'))
    # 가끔 실패했다. PS 진행률 CLIXML 이 섞였는지 본다
    assert "[stderr] bad" in result.output, result.output


def test_korean_output_is_not_garbled(shell_backend: WindowsShellBackend) -> None:
    assert "한글 출력" in shell_backend.execute("echo '한글 출력'").output
    script = Path(shell_backend.cwd, "ko.py")
    script.write_text("print('파이썬 한글')\n", encoding="utf-8")
    assert "파이썬 한글" in shell_backend.execute(run_python(shell_backend.shell, "ko.py")).output


def test_runs_in_workspace(shell_backend: WindowsShellBackend) -> None:
    Path(shell_backend.cwd, "marker.txt").write_text("x", encoding="utf-8")
    assert "marker.txt" in shell_backend.execute("ls").output


def test_timeout_kills_process_tree(shell_backend: WindowsShellBackend) -> None:
    """손자 프로세스까지 끝나야 한다.

    기본 구현은 손자가 쥔 파이프가 닫힐 때까지(여기선 60초) 돌아오지 않는다.
    """
    Path(shell_backend.cwd, "spawn.py").write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "open('child.pid', 'w').write(str(child.pid))\n"
        "time.sleep(60)\n", encoding="utf-8")
    started = time.perf_counter()
    result = shell_backend.execute(run_python(shell_backend.shell, "spawn.py"), timeout=5)
    assert time.perf_counter() - started < 20
    assert result.exit_code == 124
    assert "timed out" in result.output
    pid = int(Path(shell_backend.cwd, "child.pid").read_text())
    assert not alive(pid)


@pytest.mark.skipif(sys.platform != "win32", reason="OEM 코드페이지는 Windows 전용")
def test_oem_codepage_output_is_decoded(shell_backend: WindowsShellBackend) -> None:
    import ctypes
    encoding = f"cp{ctypes.windll.kernel32.GetOEMCP()}"
    text = "한글" if encoding == "cp949" else "café"
    Path(shell_backend.cwd, "oem.py").write_text(
        f"import sys; sys.stdout.buffer.write({text!r}.encode({encoding!r}))\n", encoding="utf-8")
    assert text in shell_backend.execute(run_python(shell_backend.shell, "oem.py")).output


def test_shell_env_drops_secrets() -> None:
    secrets = {"ANTHROPIC_API_KEY", "GITHUB_TOKEN", "DB_PASSWORD", "AWS_SECRET_ACCESS_KEY",
               "OPENROUTER_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "gpg_key"}
    env = shell_env({"PATH": "p", "SYSTEMROOT": "s", "LANGFUSE_BASE_URL": "u",
                     **dict.fromkeys(secrets, "x")},
                    find_python=lambda env, workspace: None)
    assert {"PATH", "SYSTEMROOT", "LANGFUSE_BASE_URL"} <= env.keys()
    assert not secrets & env.keys()


def test_shell_env_drops_tool_venv(tmp_path: Path) -> None:
    """code-agent 자신의 venv 를 셸에 물려주지 않는다.

    물려주면 셸의 python·pip 이 deepagents 가 깔린 도구 venv 가 된다.
    """
    scripts, other = tmp_path / ".venv" / "Scripts", tmp_path / "bin"
    base = {"PATH": os.pathsep.join([str(scripts), str(other)]),
            "VIRTUAL_ENV": str(tmp_path / ".venv"),
            "UV": "uv.exe", "UV_RUN_RECURSION_DEPTH": "1", "PYTHONHOME": "x", "SYSTEMROOT": "s"}
    env = shell_env(base, tool_scripts=str(scripts), find_python=lambda env, workspace: None)
    assert not {"VIRTUAL_ENV", "UV", "UV_RUN_RECURSION_DEPTH", "PYTHONHOME"} & env.keys()
    assert env["PATH"].split(os.pathsep) == [str(other)] and env["SYSTEMROOT"] == "s"


def test_shell_env_prepends_uv_python_when_stub(tmp_path: Path) -> None:
    stubs = tmp_path / "AppData" / "Local" / "Microsoft" / "WindowsApps"
    stubs.mkdir(parents=True)
    (stubs / "python.exe").write_bytes(b"")  # 스토어 실행 별칭은 크기가 0 이다
    uv_python = tmp_path / "uv" / "python" / "cpython-3.13" / "python.exe"
    uv_python.parent.mkdir(parents=True)
    uv_python.write_bytes(b"x")
    asked: list[dict] = []
    env = shell_env({"PATH": str(stubs)}, tool_scripts=str(tmp_path / "none"),
                    find_python=lambda env, workspace: asked.append(dict(env)) or str(uv_python))
    assert env["PATH"].split(os.pathsep) == [str(uv_python.parent), str(stubs)]
    assert "VIRTUAL_ENV" not in asked[0]  # 정리한 env 로 묻는다


def test_shell_env_without_uv(tmp_path: Path) -> None:
    """uv 가 없으면(uv python find 실패) PATH 를 더하지 않고 예외도 없다.

    프롬프트에는 python 이 없음으로 나온다.
    """
    env = shell_env({"PATH": str(tmp_path)}, tool_scripts=str(tmp_path / "none"))
    assert env["PATH"] == str(tmp_path)


def test_shell_env_keeps_activated_project_venv(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """대상 프로젝트의 venv 를 켜고 그 폴더에서 띄우면 셸의 python 은 그 venv 다(PR #8 리뷰 1).

    고치기 전에는 '현재 폴더 안'이라 버려서
    uv 인터프리터를 PATH 맨 앞에 붙여 프로젝트 venv 를 가렸다.
    """
    project = tmp_path / "proj"
    venv_scripts = project / ".venv" / "Scripts"
    venv_scripts.mkdir(parents=True)
    (venv_scripts / "python.exe").write_bytes(b"x")
    monkeypatch.chdir(project)
    asked: list[Path | None] = []
    env = shell_env({"PATH": os.pathsep.join([str(venv_scripts), str(tmp_path / "bin")])},
                    workspace=project, tool_scripts=str(tmp_path / "none"),
                    find_python=lambda env, workspace: (asked.append(workspace)
                                                        or str(tmp_path / "uv" / "python.exe")))
    assert env["PATH"].split(os.pathsep) == [str(venv_scripts), str(tmp_path / "bin")]
    assert asked == []


def test_shell_env_drops_relative_path_entries(tmp_path: Path) -> None:
    """상대 경로 항목은 셸의 현재 폴더(작업 폴더) 기준으로 풀린다.

    저장소에 심은 실행 파일이 자동 승인 명령으로 불리지 않게 뺀다.
    """
    # Windows PATH 는 따옴표로 감싼 항목도 받는다
    absolute, quoted = str(tmp_path / "a"), f'"{tmp_path / "b c"}"'
    base = {"PATH": os.pathsep.join([".", "bin", "", absolute, "node_modules/.bin", quoted, ".."])}
    env = shell_env(base, tool_scripts=str(tmp_path / "none"),
                    find_python=lambda env, workspace: None)
    assert env["PATH"].split(os.pathsep) == [absolute, quoted]


def test_shell_env_passes_workspace_to_find_python(tmp_path: Path) -> None:
    asked: list[Path | None] = []
    shell_env({"PATH": str(tmp_path)}, workspace=tmp_path / "ws",
              tool_scripts=str(tmp_path / "none"),
              find_python=lambda env, workspace: asked.append(workspace))
    assert asked == [tmp_path / "ws"]


def test_shell_backend_and_detection_share_workspace_env(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """셸 백엔드와 환경 탐지(cli.detect_environment)는 같은 작업 폴더 기준 환경을 쓰고,
    작업 폴더마다 한 번만 만든다.
    """
    from code_agent import backend

    made: list[Path | None] = []

    def fake_shell_env(**kwargs) -> dict[str, str]:
        made.append(kwargs["workspace"])
        return {"PATH": f"env{len(made)}"}

    monkeypatch.setattr(backend, "shell_env", fake_shell_env)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    shell = WindowsShellBackend(workspace, shell=SHELLS[-1])
    assert (backend.process_shell_env(tmp_path / "ws" / ".." / "ws") == shell._env
            == {"PATH": "env1"})
    assert made == [workspace.resolve()]


def test_powershell_hides_progress() -> None:
    from code_agent import backend

    assert "$ProgressPreference = 'SilentlyContinue'" in backend._PS_UTF8


@needs_bash
def test_shell_cannot_see_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    backend = WindowsShellBackend(tmp_path, shell="bash")
    result = backend.execute("printenv ANTHROPIC_API_KEY || echo missing")
    assert "sk-should-not-leak" not in result.output
    assert "missing" in result.output


@pytest.mark.parametrize("path", ["/../outside.txt", "../outside.txt", "C:/Windows/win.ini",
                                  "/C:/Windows/win.ini"])
def test_file_tools_stay_in_workspace(tmp_path: Path, path: str) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("SECRET", encoding="utf-8")
    backend = make_backend(workspace, artifacts_dir=tmp_path / "artifacts")
    with pytest.raises(ValueError):
        backend.read(path)


def test_offload_goes_to_artifacts_not_workspace(tmp_path: Path) -> None:
    workspace, artifacts = tmp_path / "ws", tmp_path / "artifacts"
    workspace.mkdir()
    backend = make_backend(workspace, artifacts_dir=artifacts)
    middleware = FilesystemMiddleware(backend=backend)
    for prefix in (middleware._large_tool_results_prefix, middleware._conversation_history_prefix):
        assert prefix.startswith(ARTIFACTS_ROUTE)
    assert backend.write(f"{middleware._large_tool_results_prefix}/call-1", "big").error is None
    assert (artifacts / "large_tool_results" / "call-1").read_text(encoding="utf-8") == "big"
    assert list(workspace.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell 5.1 전용")
def test_powershell_progress_suppressed(tmp_path: Path) -> None:
    """진행률 출력이 CLIXML 로 stderr 에 섞이지 않는다($ProgressPreference).

    이 PC 에서는 설정이 없어도 CLIXML 이 재현되지 않았다(Write-Progress, 모듈 가져오기 모두).
    그래서 수정의 증거가 아니라 출력 모양을 지키는 안전망이다.
    다른 환경(CI 러너 등)에서 섞이면 이 시험이 잡는다.
    """
    backend = WindowsShellBackend(tmp_path, shell="powershell", timeout=60)
    result = backend.execute(
        "Write-Progress -Activity x -Status y; [Console]::Error.WriteLine('bad')")
    assert "CLIXML" not in result.output and "[stderr] bad" in result.output, result.output
