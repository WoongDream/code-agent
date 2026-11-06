"""보안 1차(security-review.md S1~S5, S12, S13): .env 가드와 셸 자동 승인의 우회 사례. 비밀값은 가짜(DUMMY)다.

모듈 수준에서는 main 에도 있던 이름만 가져온다. 수정 전 코드로 돌리면 사례마다 실패해야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fakes import tool_call
from langchain_core.messages import AIMessage, ToolMessage
from test_agent import Harness
from test_cli import make_app

from code_agent.approval import APPROVE, is_read_only_command, reject
from code_agent.guards import is_env_path

SECRET = "DUMMY_SECRET_VALUE_1234"


# --- S1: 파일 도구 가드의 이름 변형 ---

@pytest.mark.parametrize("name", [".ENV", ".Env.local", ".env ", ".env::$DATA", ".env:x", ".env.", "/sub/.ENV",
                                  "\\sub\\.eNv.production"])
def test_env_guard_variants(name: str) -> None:
    assert is_env_path(name)


@pytest.mark.parametrize("name", [".env.Example", ".ENV.SAMPLE", "env.txt", "my.env.md"])
def test_env_templates_and_others_still_allowed(name: str) -> None:
    assert not is_env_path(name)


@pytest.mark.parametrize("path", ["/.ENV", "/.env ", "/.env::$DATA", "/.env."])
def test_file_tools_cannot_read_env_variants(tmp_path: Path, path: str) -> None:
    """Windows 는 이 이름들을 모두 .env 로 연다. 수정 전에는 가짜 비밀값이 모델 입력으로 들어갔다."""
    h = Harness(tmp_path, [tool_call("read_file", {"file_path": path}), AIMessage(content="끝")])
    (h.workspace / ".env").write_text(f"API_KEY={SECRET}\n", encoding="utf-8")
    h.run()
    [message] = h.tool_messages()
    assert "권한 거부" in message.text
    assert all(SECRET not in str(m.content) for m in h.model.calls[-1])


@pytest.mark.parametrize("path", ["/.ENV", "/.env "])
def test_file_tools_cannot_overwrite_env_variants(tmp_path: Path, path: str) -> None:
    h = Harness(tmp_path, [tool_call("edit_file", {"file_path": path, "old_string": SECRET, "new_string": "x"}),
                           AIMessage(content="끝")], [APPROVE])
    (h.workspace / ".env").write_text(f"API_KEY={SECRET}\n", encoding="utf-8")
    h.run()
    assert SECRET in (h.workspace / ".env").read_text(encoding="utf-8")


def short_name(path: Path) -> str:
    import ctypes

    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 260)
    return Path(buf.value).name


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 짧은 이름은 Windows 전용")
def test_env_guard_short_name(tmp_path: Path) -> None:
    """8.3 이름(ENV~1)은 이름만으로는 알 수 없어 실제 파일의 긴 이름으로 다시 판정한다."""
    from code_agent.guards import DotenvGuardMiddleware

    h = Harness(tmp_path, [])
    (h.workspace / ".env").write_text(f"API_KEY={SECRET}\n", encoding="utf-8")
    short = short_name(h.workspace / ".env")
    if short == ".env":
        pytest.skip("이 볼륨은 8.3 이름 생성이 꺼져 있다")
    guard = DotenvGuardMiddleware(workspace=h.workspace)
    assert guard.blocks("/" + short)
    assert guard.blocks("/NEWFIL~1.TXT")  # 없는 대상은 8.3 모양이면 보수적으로 막는다
    assert not guard.blocks("/app.py")
    h.model.steps += [tool_call("read_file", {"file_path": "/" + short}), AIMessage(content="끝")]
    h.run()
    assert all(SECRET not in str(m.content) for m in h.model.calls[-1])


# --- S2~S5, S12, S13: 셸 자동 승인 판정 ---

@pytest.mark.parametrize("command", [
    # S2 대소문자와 8.3
    "cat .ENV", "cat sub/.Env.local", "head .eNv", "cat ENV~1", "cat SUBDIR~1/x.txt",
    # S3 옵션 값으로 경로 검사 우회
    "wc --files0-from=/c/Users/x/victim.txt", "wc --files0-from=C:/x", "wc --files0-from victim.txt",
    "cat --foo=/etc/passwd", "head --lines=5 --x=../secret README.md",
    # S4·S13 PowerShell 공급자 경로와 드라이브
    "cat FileSystem::C:\\x\\victim.txt", "cat Microsoft.PowerShell.Core\\FileSystem::C:\\x", "ls HKCU:\\Software",
    "cat env:PATH", "ls Cert:", "cat Variable:x", "ls Env:",
    # S5 둥근 따옴표·긴 대시·숨은 문자
    "cat \u2018.env\u2019", "cat \u201c.env\u201d", "wc \u2014files0-from=x", "cat a.txt\u00a0.env",
    "cat \u202e.env", "cat a\u200b.txt",
    # S12 tail 이 끝나지 않음
    "tail -f log.txt", "tail -F log.txt", "tail --follow=name log.txt", "tail -fn5 log.txt",
    # 개인 키·자격 증명(사용자 결정: 묻기)
    "cat id_rsa", "cat sub/id_ed25519", "cat server.pem", "cat ID_RSA", "cat credentials.json",
    "cat .aws/credentials", "head .netrc", "git show HEAD:id_rsa",
])
def test_shell_bypasses_require_approval(command: str) -> None:
    assert not is_read_only_command(command)


@pytest.mark.parametrize("command", [
    "git diff --stat HEAD~1", "git log HEAD~3..HEAD", "git show HEAD:README.md", "cat id_rsa.pub",
    "tail -n 5 log.txt", "wc -l a.py", "head --lines=5 README.md", "cat 'docs/한글 문서.md'", "ls -la src",
    "git log --format=%h -n 3",
])
def test_ordinary_reads_stay_auto_approved(command: str) -> None:
    """고친 뒤에도 평범한 읽기는 묻지 않는다(git 리비전의 ~·: 는 경로 규칙을 적용하지 않는다)."""
    assert is_read_only_command(command)


# --- 개인 키 파일 읽기는 묻기(사용자 결정) ---

def read_key_app(tmp_path: Path, *answers: str, mode: str = "manual"):
    steps = [tool_call("read_file", {"file_path": "/id_rsa"}), AIMessage(content="끝")]
    app, model, out, script = make_app(tmp_path, steps, "키 읽어", *answers, "/exit", mode=mode)
    (app.workspace / "id_rsa").write_text(f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SECRET}\n", encoding="utf-8")
    app.run()
    return model, out, script


def test_private_key_read_asks_and_reject_keeps_content_out(tmp_path: Path) -> None:
    model, out, script = read_key_app(tmp_path, "n", "")
    assert script.prompts.count("[y] 승인 / [n] 거부 › ") == 1
    assert "승인 요청: read_file" in out.getvalue()
    [result] = [m for m in model.calls[-1] if isinstance(m, ToolMessage)]
    assert result.status == "error" and SECRET not in result.text


def test_private_key_read_approved(tmp_path: Path) -> None:
    model, _, _ = read_key_app(tmp_path, "y")
    [result] = [m for m in model.calls[-1] if isinstance(m, ToolMessage)]
    assert SECRET in result.text


@pytest.mark.parametrize(("mode", "asked"), [("auto-edit", 1), ("yolo", 0)])
def test_private_key_read_by_mode(tmp_path: Path, mode: str, asked: int) -> None:
    _, _, script = read_key_app(tmp_path, "n", "", mode=mode)
    assert script.prompts.count("[y] 승인 / [n] 거부 › ") == asked


def test_ordinary_read_not_asked(tmp_path: Path) -> None:
    h = Harness(tmp_path, [tool_call("read_file", {"file_path": "/id_rsa.pub"}), AIMessage(content="끝")])
    (h.workspace / "id_rsa.pub").write_text("ssh-ed25519 AAAA public", encoding="utf-8")
    h.run()
    assert h.requests == [] and "AAAA public" in h.tool_messages()[0].text


def test_reject_reason_reaches_model(tmp_path: Path) -> None:
    h = Harness(tmp_path, [tool_call("read_file", {"file_path": "/server.pem"}), AIMessage(content="끝")],
                [reject("키는 안 돼")])
    (h.workspace / "server.pem").write_text(SECRET, encoding="utf-8")
    h.run()
    [[request]] = h.requests
    assert request["name"] == "read_file" and request["description"]
    assert "키는 안 돼" in h.tool_messages()[0].text and SECRET not in h.tool_messages()[0].text


# --- 현재 폴더의 실행 파일 가로채기(PR #8 리뷰에서 발견, main 에도 있는 경로) ---

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="현재 폴더 실행 파일 탐색은 Windows 동작")


@pytest.fixture
def planted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """악성 저장소처럼 현재 폴더에 git.cmd 와 bin/bash.exe 를 둔다. 빈 파일이고 실행하지 않는다(존재만 본다).

    NoDefaultCurrentDirectoryInExePath 를 지우지 않으면 which 가 현재 폴더를 보지 않아 이 시험들은 무의미하다.
    """
    import shutil

    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "git.cmd").write_bytes(b"")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "bash.exe").write_bytes(b"")
    found = shutil.which("git")
    if found is None or Path(found).resolve().parent != tmp_path.resolve():
        pytest.skip("이 Python 의 which 는 현재 폴더를 먼저 찾지 않는다")
    return tmp_path.resolve()


def outside(path: str | Path | None, root: Path) -> bool:
    return path is None or not Path(path).resolve().is_relative_to(root)


@windows_only
def test_git_bash_not_taken_from_current_folder(planted: Path) -> None:
    from code_agent.backend import find_git_bash, resolve_shell

    assert outside(find_git_bash(), planted)
    assert outside(resolve_shell("")[1][0], planted)  # 에이전트 셸 자체가 바뀌지 않는다


@windows_only
def test_safe_which_skips_current_and_workspace(planted: Path) -> None:
    from code_agent.backend import safe_which

    assert outside(safe_which("git"), planted) and outside(safe_which("git", planted), planted)


@windows_only
def test_powershell_and_taskkill_are_absolute() -> None:
    from code_agent.backend import resolve_shell, system32

    powershell = resolve_shell("powershell")[1][0]
    assert Path(powershell).is_absolute() and Path(powershell).is_file()
    assert Path(system32("taskkill.exe")).is_absolute() and Path(system32("taskkill.exe")).is_file()


@windows_only
def test_diff_command_does_not_run_planted_git(planted: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from code_agent import cli

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return cli.subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    app, _, _, _ = make_app(planted, [], "/diff", "/exit", workspace=planted)
    app.run()
    assert all(outside(argv[0], planted) for argv in calls)


@windows_only
def test_cli_main_disables_current_folder_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from code_agent import cli
    from code_agent.settings import Settings

    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)  # 지우지 않으면 시험이 무의미하다
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(data_dir=tmp_path / "data"))
    assert cli.main(["--resume", "nope"]) == 2  # 시작 단계에서 끝나도 변수는 맨 앞에서 정한다
    assert cli.os.environ.get("NoDefaultCurrentDirectoryInExePath") == "1"


# --- O4: -w 를 빼먹어 code-agent 저장소(실제 키가 든 .env)에서 띄운 경우 ---

@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """REPO_ROOT 를 가짜 .env 가 든 임시 폴더로 바꾸고, 키 없는 설정으로 main 을 돌릴 수 있게 한다."""
    from code_agent import cli
    from code_agent.settings import Settings

    repo = tmp_path / "code-agent"
    repo.mkdir()
    (repo / ".env").write_text(f"OPENROUTER_API_KEY={SECRET}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "REPO_ROOT", repo)
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(data_dir=tmp_path / "data"))
    return repo


def test_repo_folder_warns_and_asks(fake_repo: Path, monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    from code_agent import cli

    asked: list[str] = []
    monkeypatch.setattr(cli, "_confirm", lambda prompt: asked.append(prompt) or False)
    monkeypatch.chdir(fake_repo)
    assert cli.main([]) == 2  # 거절하면 시작하지 않는다
    out = capsys.readouterr().out
    assert "작업 폴더가 code-agent 저장소다. 실제 키가 든 .env 가 있다" in out and len(asked) == 1
    monkeypatch.chdir(fake_repo.parent)  # 저장소를 품은 상위 폴더도 같다
    cli.main([])
    assert "code-agent 저장소를 품은 폴더다" in capsys.readouterr().out and len(asked) == 2


def test_repo_folder_explicit_workspace_warns_without_asking(fake_repo: Path, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    from code_agent import cli

    monkeypatch.setattr(cli, "_confirm", lambda prompt: pytest.fail("-w 를 명시했는데 물었다"))
    cli.main(["-w", str(fake_repo)])  # 키가 없어 모델 생성에서 끝나지만 경고는 그 전에 나온다
    assert "작업 폴더가 code-agent 저장소다" in capsys.readouterr().out


def test_other_folder_no_repo_warning(fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    from code_agent import cli

    other = tmp_path / "project"
    other.mkdir()
    monkeypatch.setattr(cli, "_confirm", lambda prompt: pytest.fail("물을 일이 아니다"))
    cli.main(["-w", str(other)])
    assert "code-agent 저장소" not in capsys.readouterr().out
