"""실행 환경 탐지. 셸이 쓸 PATH 에서 도구·버전·브라우저를 확인해 시스템 프롬프트의 한 절로 만든다.

에이전트가 없는 도구(node 등)를 매번 명령으로 확인하거나,
code-agent 자신의 venv python 을 쓰지 않게 하려는 것이다.

- 셸에 넘기는 env 의 PATH 로 찾는다(shutil.which(path=...)). 셸이 실제로 보는 것과 같아야 한다.
- WindowsApps 아래 크기 0 인 실행 파일은 스토어 스텁이라 없는 것으로 본다(python3, python 등).
- 버전은 병렬로 --version 을 돌려 숫자만 남긴다.
  실패하면 '버전 확인 실패' 로 적되 있는 것으로 친다.
- 결과 텍스트는 결정적이다(항목 순서 고정, 시각·소요 시간 없음).
  system 이 바뀌면 프롬프트 캐시와 Opus 5.5 의 보존된 thinking 이 깨지므로,
  텍스트는 스레드와 함께 저장하고 --resume 은 그것을 다시 쓴다(cli.Sessions).
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

# 표시 순서이기도 하다. 결과 텍스트가 결정적이어야 한다
TOOLS = ("python", "uv", "git", "node", "npm", "deno", "py", "pwsh", "powershell")
VERSION_TIMEOUT_S = 3
# 셸은 쓰지만 시작할 때 실행하지 않은 도구의 덧붙임
IN_WORKSPACE = "작업 폴더 안이라 버전을 확인하지 않았다"
_VERSION = re.compile(r"\d+(?:\.\d+)+")
_PS_REGISTRY = (r"SOFTWARE\Microsoft\PowerShell\3\PowerShellEngine", "PowerShellVersion")
_APP_PATHS = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
# (표시 이름, 실행 파일, ProgramFiles·ProgramFiles(x86)·LOCALAPPDATA 아래 상대 경로)
BROWSERS = (
    ("Chrome", "chrome.exe", r"Google\Chrome\Application\chrome.exe"),
    ("Edge", "msedge.exe", r"Microsoft\Edge\Application\msedge.exe"),
)

Which = Callable[..., str | None]
Run = Callable[..., subprocess.CompletedProcess]
ReadRegistry = Callable[[str, str, str], str | None]  # (hive, key, value) → 값


@dataclass(frozen=True)
class Tool:
    """도구 하나의 탐지 결과."""

    name: str
    path: str | None = None  # 없으면 None
    version: str | None = None  # 있는데 확인하지 못했으면 None
    note: str = ""  # 없을 때 이유(스토어 스텁) 또는 있을 때 덧붙임


@dataclass(frozen=True)
class EnvInfo:
    """탐지 결과 전체. render 가 시스템 프롬프트의 한 절로 만든다."""

    tools: tuple[Tool, ...]
    browsers: tuple[tuple[str, str], ...]  # (표시 이름, 경로)


def is_store_stub(path: str | os.PathLike) -> bool:
    """%LOCALAPPDATA%\\Microsoft\\WindowsApps 아래 크기 0 인 실행 별칭.

    실행하면 스토어가 열린다.
    """
    parts = [p.lower() for p in Path(path).parts]
    if "windowsapps" not in parts:
        return False
    try:
        return os.lstat(path).st_size == 0
    except OSError:
        return True


def trusted_tool(path: str | None, workspace: Path | None = None) -> str | None:
    """which 결과 중 절대 경로이고 현재 폴더·작업 폴더 밖인 것만.
    저장소에 심은 node.cmd 같은 것을 고르지 않는다.

    Windows 의 which 는 NoDefaultCurrentDirectoryInExePath 가 없으면
    PATH 를 주어도 현재 폴더를 먼저 찾는다.
    저장소 안 폴더가 PATH 에 들어간 경우(direnv 등)도 막으려고 '바로 아래'가 아니라 '안'을 본다.
    다만 드라이브 루트나 홈을 품은 폴더는 저장소가 아니고 넓어서
    (~\\.local\\bin 의 uv 까지 버리게 된다) 기준에서 뺀다.
    """
    if not path or not Path(path).is_absolute():
        return None
    resolved, home = Path(path).resolve(), Path.home().resolve()
    for root in (Path.cwd(), workspace):
        if root is None:
            continue
        root = Path(root).resolve()
        if root == Path(root.anchor) or home.is_relative_to(root):
            continue
        if resolved.is_relative_to(root):
            return None
    return path


def uv_python_find(env: Mapping[str, str], workspace: Path | None = None) -> str | None:
    """uv 가 관리하는 인터프리터 경로.
    venv 를 고르지 않게 VIRTUAL_ENV 없는 env 와 임시 폴더에서 묻는다. 실패하면 None.

    uv 를 실행하므로 현재 폴더·작업 폴더 안의 uv 는 쓰지 않는다(trusted_tool).
    -w 로 다른 폴더에서 띄워도 같다.
    """
    uv = trusted_tool(shutil.which("uv", path=env.get("PATH", "")), workspace)
    if uv is None:
        return None
    return _uv_python_find(uv, frozenset(env.items()))


@functools.cache
def _uv_python_find(uv: str, env: frozenset[tuple[str, str]]) -> str | None:
    """작업 폴더마다 셸 환경을 만들어도(backend.process_shell_env)
    uv 는 같은 uv·env 에 한 번만 묻는다.
    """
    try:
        result = subprocess.run([uv, "python", "find"], cwd=tempfile.gettempdir(), env=dict(env),
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = result.stdout.strip().splitlines()
    return found[0] if result.returncode == 0 and found and Path(found[0]).is_file() else None


def _read_registry(hive: str, key: str, value: str) -> str | None:
    if sys.platform != "win32":
        return None
    import winreg

    root = {"HKLM": winreg.HKEY_LOCAL_MACHINE, "HKCU": winreg.HKEY_CURRENT_USER}[hive]
    try:
        with winreg.OpenKey(root, key) as handle:
            data, _ = winreg.QueryValueEx(handle, value)
    except OSError:
        return None
    return str(data) if data else None


def _version(path: str, env: Mapping[str, str], run: Run) -> str | None:
    """임시 폴더에서 돌린다. 현재 폴더(작업 폴더일 수 있다)에서 돌리면 npm.cmd 가 부르는 node 처럼
    그 폴더의 실행 파일이나 .npmrc 같은 설정을 집을 수 있다."""
    try:
        result = run([path, "--version"], cwd=tempfile.gettempdir(), env=dict(env),
                     capture_output=True, text=True, encoding="utf-8", errors="replace",
                     timeout=VERSION_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    first = (result.stdout.strip() or result.stderr.strip()).splitlines()
    match = _VERSION.search(first[0]) if first else None
    return match[0] if match else None


def _browsers(env: Mapping[str, str], which: Which,
              read_registry: ReadRegistry) -> tuple[tuple[str, str], ...]:
    found = []
    for label, exe, relative in BROWSERS:
        candidates = [read_registry(hive, rf"{_APP_PATHS}\{exe}", "") for hive in ("HKLM", "HKCU")]
        candidates += [str(Path(base, relative))
                       for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
                       if (base := env.get(var) or os.environ.get(var))]
        if sys.platform != "win32":
            candidates.append(which(exe.removesuffix(".exe"), path=env.get("PATH", "")))
        path = next((c.strip('"') for c in candidates if c and Path(c.strip('"')).is_file()), None)
        if path:
            found.append((label, path))
    return tuple(found)


def probe(env: Mapping[str, str], *, workspace: Path | None = None, which: Which = shutil.which,
          run: Run = subprocess.run, read_registry: ReadRegistry = _read_registry) -> EnvInfo:
    """셸에 넘기는 env 로 도구를 찾고 버전을 병렬로 확인한다. 하나당 VERSION_TIMEOUT_S 초.

    상대 경로 결과(Windows which 가 현재 폴더를 먼저 본 것)는 셸이 찾지 않으므로 없는 것으로 본다.
    현재 폴더·작업 폴더 안의 절대 경로(켜 둔 대상 프로젝트 venv 등)는 셸이 쓰므로 있음으로 적되
    --version 으로 실행하지 않는다(trusted_tool).
    """
    paths: dict[str, str | None] = {}
    notes: dict[str, str] = {}
    for name in TOOLS:
        found = which(name, path=env.get("PATH", ""))
        path = found if found and Path(found).is_absolute() else None
        if path and is_store_stub(path):
            path, notes[name] = None, "스토어 스텁"
        paths[name] = path
    if (paths["python"]
            and f"{os.sep}uv{os.sep}python{os.sep}" in os.path.normcase(paths["python"])):
        notes["python"] = "패키지 없는 uv 관리 인터프리터"
    for name, path in paths.items():
        if path and trusted_tool(path, workspace) is None:
            notes[name] = IN_WORKSPACE
    versions: dict[str, str | None] = {}
    if paths["powershell"] and notes.get("powershell") != IN_WORKSPACE:
        raw = read_registry("HKLM", *_PS_REGISTRY)
        versions["powershell"] = ".".join(raw.split(".")[:2]) if raw else None
    targets = [n for n in TOOLS if paths[n] and n not in versions and notes.get(n) != IN_WORKSPACE]
    with ThreadPoolExecutor(4) as pool:
        for name, version in zip(targets,
                                 pool.map(lambda n: _version(str(paths[n]), env, run), targets),
                                 strict=True):
            versions[name] = version
    tools = tuple(Tool(n, paths[n], versions.get(n), notes.get(n, "")) for n in TOOLS)
    return EnvInfo(tools, _browsers(env, which, read_registry))


def shell_path(path: str, shell: str) -> str:
    """셸에서 쓰는 형식. bash 는 /c/…, powershell 은 C:\\… 이고 공백이 있으면 따옴표로 감싼다."""
    if shell == "bash" and PureWindowsPath(path).drive:
        windows = PureWindowsPath(path)
        path = "/" + windows.drive.rstrip(":").lower() + "/" + "/".join(windows.parts[1:])
    return f'"{path}"' if " " in path else path


def render(info: EnvInfo, shell: str) -> str:
    """시스템 프롬프트에 붙일 절. 같은 입력이면 같은 문자열이다."""
    present, missing = [], []
    for tool in info.tools:
        if tool.path is None:
            missing.append(f"{tool.name}({tool.note})" if tool.note else tool.name)
            continue
        label = "Windows PowerShell" if tool.name == "powershell" else tool.name
        if tool.version:
            label = f"{label} {tool.version}"
        elif tool.note != IN_WORKSPACE:
            label = f"{label} (버전 확인 실패)"
        present.append(label + (f"({tool.note})" if tool.note else ""))
    if info.browsers:
        listed = ", ".join(f"{label} {shell_path(path, shell)}" for label, path in info.browsers)
        how = ". PowerShell 에서는 & \"경로\" 로 실행한다" if shell == "powershell" else ""
        browsers = f"{listed} (headless 로 HTML 을 확인할 수 있다{how})"
    else:
        browsers = "없음"
    return (
        "\n# 이 PC 의 도구 (세션 시작 때 확인했다)\n"
        f"- 있음: {', '.join(present) or '없음'}\n"
        f"- 브라우저: {browsers}\n"
        f"- 없음: {', '.join(missing) or '없음'}\n"
        "- 없는 도구를 확인하려고 명령을 다시 실행하지 않는다. "
        "꼭 필요하면 설치하지 말고 사용자에게 묻는다.\n"
    )
