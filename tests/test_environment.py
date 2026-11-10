"""실행 환경 탐지: 있음·없음, 스토어 스텁, 버전 실패, 브라우저 경로 형식, 결정적 출력.

which·run·레지스트리는 가짜다.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from code_agent.environment import is_store_stub, probe, render, shell_path

VERSIONS = {"python": "Python 3.13.18", "uv": "uv 0.13.7 (abc1234 2026-10-01)",
            "git": "git version 2.56.0.windows.1", "node": "v22.11.0"}


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
        return subprocess.CompletedProcess(args, 0, stdout=VERSIONS.get(name, "") + "\n",
                                           stderr="")

    def registry(hive: str, key: str, value: str) -> str | None:
        return "5.1.26100.1" if value == "PowerShellVersion" else None

    return {"which": which, "run": run, "read_registry": registry}


def present_on_this_pc(tmp_path: Path) -> dict[str, str]:
    return {"python": tool(tmp_path, "AppData", "Roaming", "uv", "python", "cpython-3.13",
                           "python.exe"),
            "uv": tool(tmp_path, ".local", "bin", "uv.exe"),
            "git": tool(tmp_path, "Git", "bin", "git.exe"),
            "powershell": tool(tmp_path, "WindowsPowerShell", "powershell.exe")}


def test_probe_found_and_missing(tmp_path: Path) -> None:
    info = probe(fake_env(tmp_path), **fakes(present_on_this_pc(tmp_path)))
    text = render(info, "bash")
    assert ("- 있음: python 3.13.18(패키지 없는 uv 관리 인터프리터), uv 0.13.7, git 2.56.0, "
            "Windows PowerShell 5.1\n") in text
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
    # 공백 없으면 따옴표 없음
    assert shell_path(r"D:\tools\chrome.exe", shell) == ("/d/tools/chrome.exe" if shell == "bash"
                                                         else r"D:\tools\chrome.exe")


def test_browsers_found_by_fixed_path(tmp_path: Path) -> None:
    env = fake_env(tmp_path)
    chrome = tool(Path(env["ProgramFiles"]), "Google", "Chrome", "Application", "chrome.exe")
    edge = tool(Path(env["ProgramFiles(x86)"]), "Microsoft", "Edge", "Application", "msedge.exe")
    info = probe(env, **fakes({}))
    assert info.browsers == (("Chrome", chrome), ("Edge", edge))
    text = render(info, "powershell")
    assert (f"- 브라우저: Chrome {shell_path(chrome, 'powershell')}, "
            f"Edge {shell_path(edge, 'powershell')}") in text
    assert 'PowerShell 에서는 & "경로" 로 실행한다' in text


@pytest.mark.skipif(sys.platform != "win32", reason="현재 폴더 실행 파일 탐색은 Windows 동작")
def test_probe_skips_planted_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """저장소 폴더에 심은 node.cmd 를 찾아 --version 으로 실행하지 않는다.

    NoDefaultCurrentDirectoryInExePath 를 지우지 않으면
    which 가 현재 폴더를 보지 않아 이 시험은 무의미하다.
    심은 파일은 빈 파일이고 실행되지 않는다(가짜 run 이 불린 경로만 본다).
    """
    import shutil

    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    repo.mkdir()
    bin_dir.mkdir()
    (repo / "node.cmd").write_bytes(b"")
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.chdir(repo)
    found = shutil.which("node", path=str(bin_dir))
    if found is None or Path(found).resolve().parent != repo.resolve():
        pytest.skip("이 Python 의 which 는 현재 폴더를 먼저 찾지 않는다")
    ran: list[str] = []

    def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        ran.append(args[0])
        return subprocess.CompletedProcess(args, 0, stdout="v1.0.0", stderr="")

    env = {**fake_env(tmp_path), "PATH": str(bin_dir)}
    info = probe(env, workspace=repo, run=run, read_registry=lambda hive, key, value: None)
    assert next(t for t in info.tools if t.name == "node").path is None
    assert not any(Path(p).resolve().is_relative_to(repo.resolve()) for p in ran)


def test_trusted_tool_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """저장소(현재 폴더·작업 폴더) 안은 버리고,
    홈을 품은 폴더나 드라이브 루트에서 띄운 경우는 기준에서 뺀다.
    """
    from code_agent.environment import trusted_tool

    repo = tmp_path / "repo"
    (repo / "bin").mkdir(parents=True)
    tool_path = tool(repo, "bin", "git.exe")
    monkeypatch.chdir(repo)
    # 안이거나 상대 경로
    assert trusted_tool(tool_path) is None and trusted_tool("git.exe") is None
    elsewhere = tool(tmp_path, "tools", "git.exe")
    assert trusted_tool(elsewhere) == elsewhere
    assert trusted_tool(elsewhere, workspace=tmp_path / "tools") is None
    monkeypatch.chdir(Path.home())  # 홈에서 띄워도 ~ 아래 도구(uv 등)는 버리지 않는다
    assert trusted_tool(elsewhere) == elsewhere


def test_probe_workspace_tool_present_but_not_run(tmp_path: Path) -> None:
    """작업 폴더 안의 절대 경로(켜 둔 프로젝트 venv)는 셸이 쓰므로 있음으로 적되,
    --version 으로 실행하지 않는다.

    상대 경로 결과(Windows which 가 현재 폴더를 먼저 본 것)는 셸이 찾지 않으므로 없음이다.
    """
    project = tmp_path / "proj"
    venv_python = tool(project, ".venv", "Scripts", "python.exe")
    paths = {**present_on_this_pc(tmp_path), "python": venv_python, "node": "node.cmd"}
    ran: list[str] = []
    fake = fakes(paths)
    run = fake["run"]
    fake["run"] = lambda args, **kwargs: ran.append(args[0]) or run(args, **kwargs)
    text = render(probe(fake_env(tmp_path), workspace=project, **fake), "bash")
    assert ("- 있음: python(작업 폴더 안이라 버전을 확인하지 않았다), uv 0.13.7, git 2.56.0, "
            "Windows PowerShell 5.1\n") in text
    assert "- 없음: node, npm, deno, py, pwsh\n" in text
    assert venv_python not in ran and len(ran) == 2  # uv, git


def test_version_runs_in_temp_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--version 은 현재 폴더(작업 폴더일 수 있다)가 아니라 임시 폴더에서 돈다.

    그 폴더의 실행 파일·설정을 집지 않는다.
    """
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    cwds: list[str | None] = []
    fake = fakes(present_on_this_pc(tmp_path))
    run = fake["run"]
    fake["run"] = lambda args, **kwargs: cwds.append(kwargs.get("cwd")) or run(args, **kwargs)
    probe(fake_env(tmp_path), workspace=project, **fake)
    assert len(cwds) == 3 and set(cwds) == {tempfile.gettempdir()}


@pytest.mark.skipif(sys.platform != "win32",
                    reason="cmd.exe 가 현재 폴더를 먼저 찾는 Windows 동작")
def test_version_does_not_run_planted_tool_via_cmd_shim(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """npm.cmd 같은 cmd 래퍼는 안에서 node 를 이름만으로 부르고,
    cmd.exe 는 그것을 현재 폴더에서 먼저 찾는다.

    현재 폴더에서 --version 을 돌리면 저장소에 심은 node.cmd 가 실행됐다.
    심은 파일은 표식 파일만 남기는 배치 파일이다.
    """
    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    repo.mkdir()
    bin_dir.mkdir()
    (bin_dir / "npm.cmd").write_text("@echo off\r\nnode --version\r\n", encoding="ascii")
    (repo / "node.cmd").write_text('@echo off\r\necho planted> "%~dp0planted.txt"\r\n'
                                   'echo v6.6.6\r\n', encoding="ascii")
    monkeypatch.chdir(repo)
    # NoDefaultCurrentDirectoryInExePath 없음: cmd 가 현재 폴더를 본다
    env = {**fake_env(tmp_path), "PATH": str(bin_dir),
           **{k: os.environ[k] for k in ("SYSTEMROOT", "COMSPEC", "PATHEXT") if k in os.environ}}
    info = probe(env, workspace=repo,
                 which=lambda name, path="": str(bin_dir / "npm.cmd") if name == "npm" else None,
                 read_registry=lambda hive, key, value: None)
    npm = next(t for t in info.tools if t.name == "npm")
    assert npm.path == str(bin_dir / "npm.cmd") and npm.version != "6.6.6"
    assert not (repo / "planted.txt").exists()


def test_uv_python_find_skips_workspace_uv(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """-w 로 다른 폴더에서 띄워도
    작업 폴더 안의 uv 로 uv python find 를 돌리지 않는다(PR #8 리뷰 3).

    심은 uv.exe 는 빈 파일이고 실행하지 않는다(subprocess.run 은 기록만 한다).
    """
    from code_agent import environment
    from code_agent.environment import uv_python_find

    repo, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    (repo / "bin").mkdir(parents=True)
    elsewhere.mkdir()
    planted_uv = tool(repo, "bin", "uv.exe")
    monkeypatch.chdir(elsewhere)
    ran: list[str] = []

    def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        ran.append(args[0])
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    monkeypatch.setattr(environment.subprocess, "run", run)
    env = {"PATH": str(repo / "bin")}
    assert uv_python_find(env, repo) is None and ran == []
    uv_python_find(env)  # 작업 폴더를 모르면 돌렸다(고치기 전 동작)
    # which 는 uv.EXE 로 돌려준다
    assert [os.path.normcase(p) for p in ran] == [os.path.normcase(planted_uv)]


def test_render_deterministic(tmp_path: Path) -> None:
    """같은 PC 면 같은 문자열이다. system 이 바뀌면 캐시와 보존된 thinking 이 깨진다."""
    env, paths = fake_env(tmp_path), present_on_this_pc(tmp_path)
    tool(Path(env["ProgramFiles"]), "Google", "Chrome", "Application", "chrome.exe")
    texts = {render(probe(env, **fakes(paths)), shell) for shell in ("bash", "bash", "bash")}
    assert len(texts) == 1
