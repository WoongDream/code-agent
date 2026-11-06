"""실행 환경 탐지: 있음·없음, 스토어 스텁, 버전 실패, 브라우저 경로 형식, 결정적 출력. which·run·레지스트리는 가짜다."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from code_agent.environment import is_store_stub, probe, render, shell_path

VERSIONS = {"python": "Python 3.13.18", "uv": "uv 0.13.7 (abc1234 2026-10-01)", "git": "git version 2.56.0.windows.1",
            "node": "v22.11.0"}


def fake_env(tmp_path: Path) -> dict[str, str]:
    """브라우저 고정 경로가 실제 PC 의 Program Files 를 보지 않게 임시 폴더로 둔다."""
    roots = {var: str(tmp_path / name) for var, name in
             (("ProgramFiles", "pf"), ("ProgramFiles(x86)", "pf86"), ("LOCALAPPDATA", "local"))}
    return {"PATH": str(tmp_path / "bin"), **roots}


def tool(tmp_path: Path, *parts: str, size: int = 10) -> str:
    path = tmp_path.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return str(path)


def fakes(paths: dict[str, str], *, fail: set[str] = frozenset()):
    def which(name: str, path: str = "") -> str | None:
        return paths.get(name)

    def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        name = next(n for n, p in paths.items() if p == args[0])
        if name in fail:
            raise subprocess.TimeoutExpired(args, 3)
        return subprocess.CompletedProcess(args, 0, stdout=VERSIONS.get(name, "") + "\n", stderr="")

    def registry(hive: str, key: str, value: str) -> str | None:
        return "5.1.26100.1" if value == "PowerShellVersion" else None

    return {"which": which, "run": run, "read_registry": registry}


def present_on_this_pc(tmp_path: Path) -> dict[str, str]:
    return {"python": tool(tmp_path, "AppData", "Roaming", "uv", "python", "cpython-3.13", "python.exe"),
            "uv": tool(tmp_path, ".local", "bin", "uv.exe"), "git": tool(tmp_path, "Git", "bin", "git.exe"),
            "powershell": tool(tmp_path, "WindowsPowerShell", "powershell.exe")}


def test_probe_found_and_missing(tmp_path: Path) -> None:
    info = probe(fake_env(tmp_path), **fakes(present_on_this_pc(tmp_path)))
    text = render(info, "bash")
    assert "- 있음: python 3.13.18(패키지 없는 uv 관리 인터프리터), uv 0.13.7, git 2.56.0, Windows PowerShell 5.1\n" in text
    assert "- 없음: node, npm, deno, py, pwsh\n" in text
    assert "- 브라우저: 없음\n" in text
    assert text.startswith("\n# 이 PC 의 도구 (세션 시작 때 확인했다)\n")


def test_windowsapps_stub_counts_as_missing(tmp_path: Path) -> None:
    stub = tool(tmp_path, "AppData", "Local", "Microsoft", "WindowsApps", "python.exe", size=0)
    assert is_store_stub(stub) and not is_store_stub(tool(tmp_path, "real", "python.exe"))
    info = probe(fake_env(tmp_path), **fakes({"python": stub}))
    assert "python(스토어 스텁)" in render(info, "powershell")


def test_version_timeout_tolerated(tmp_path: Path) -> None:
    info = probe(fake_env(tmp_path), **fakes(present_on_this_pc(tmp_path), fail={"git"}))
    text = render(info, "bash")
    assert "git (버전 확인 실패)" in text and "uv 0.13.7" in text  # 있음으로 친다


@pytest.mark.parametrize(("shell", "expected"), [
    ("bash", '"/c/Program Files/Google/Chrome/Application/chrome.exe"'),
    ("powershell", '"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"'),
])
def test_browser_path_forms(shell: str, expected: str) -> None:
    assert shell_path(r"C:\Program Files\Google\Chrome\Application\chrome.exe", shell) == expected
    assert shell_path(r"D:\tools\chrome.exe", shell) == ("/d/tools/chrome.exe" if shell == "bash"
                                                         else r"D:\tools\chrome.exe")  # 공백 없으면 따옴표 없음


def test_browsers_found_by_fixed_path(tmp_path: Path) -> None:
    env = fake_env(tmp_path)
    chrome = tool(Path(env["ProgramFiles"]), "Google", "Chrome", "Application", "chrome.exe")
    edge = tool(Path(env["ProgramFiles(x86)"]), "Microsoft", "Edge", "Application", "msedge.exe")
    info = probe(env, **fakes({}))
    assert info.browsers == (("Chrome", chrome), ("Edge", edge))
    text = render(info, "powershell")
    assert f"- 브라우저: Chrome {shell_path(chrome, 'powershell')}, Edge {shell_path(edge, 'powershell')}" in text
    assert 'PowerShell 에서는 & "경로" 로 실행한다' in text


def test_render_deterministic(tmp_path: Path) -> None:
    """같은 PC 면 같은 문자열이다. system 이 바뀌면 캐시와 보존된 thinking 이 깨진다."""
    env, paths = fake_env(tmp_path), present_on_this_pc(tmp_path)
    tool(Path(env["ProgramFiles"]), "Google", "Chrome", "Application", "chrome.exe")
    texts = {render(probe(env, **fakes(paths)), shell) for shell in ("bash", "bash", "bash")}
    assert len(texts) == 1
