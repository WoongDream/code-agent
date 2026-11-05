"""승인 결정: 모드별 결정, allow-list 우회 차단, 다중 요청 순서."""
from __future__ import annotations

import pytest

from code_agent.approval import (
    APPROVE,
    auto_decision,
    decide_all,
    is_read_only_command,
    reject,
    reject_reason,
)


def action(name: str, **args) -> dict:
    return {"name": name, "args": args}


EDIT = action("edit_file", file_path="/src/app.py", old_string="a", new_string="b")
WRITE = action("write_file", file_path="/new.py", content="x")
DELETE = action("delete", file_path="/old.py")
PYTEST = action("execute", command="uv run pytest -q")
GIT_STATUS = action("execute", command="git status")


@pytest.mark.parametrize(("mode", "act", "expected"), [
    ("manual", EDIT, None), ("manual", WRITE, None), ("manual", DELETE, None), ("manual", PYTEST, None),
    ("auto-edit", EDIT, APPROVE), ("auto-edit", WRITE, APPROVE), ("auto-edit", DELETE, APPROVE),
    ("auto-edit", PYTEST, None),
    ("yolo", EDIT, APPROVE), ("yolo", PYTEST, APPROVE),
    ("manual", GIT_STATUS, APPROVE), ("auto-edit", GIT_STATUS, APPROVE),
])
def test_mode_decisions(mode: str, act: dict, expected: dict | None) -> None:
    assert auto_decision(act, mode) == expected


@pytest.mark.parametrize("path", ["/.git/config", "/sub/.git/hooks/pre-commit", "\\.git\\config"])
def test_auto_edit_still_asks_for_git_dir(path: str) -> None:
    assert auto_decision(action("write_file", file_path=path, content="x"), "auto-edit") is None


@pytest.mark.parametrize("command", [
    "git status", "git diff", "git diff --stat HEAD~1", "git log -n 5 --oneline", "git show HEAD",
    "git diff main..feature", "ls", "ls -la src", "ls *.py", "pwd", "cat src/app.py", "head -n 20 README.md",
    "tail README.md", "wc -l src/app.py", "which git", "cat 'docs/한글 문서.md'",
])
def test_read_only_commands_are_auto_approved(command: str) -> None:
    assert is_read_only_command(command)


@pytest.mark.parametrize("command", [
    # 명령 연결·파이프·치환·리디렉션
    "git status && rm -rf x", "git log | sh", "ls; rm x", "ls\nrm x", "cat `whoami`", "cat $(echo a)",
    "echo hi > a.txt", "cat < a", "ls & rm x", "ls || rm x", "cat $HOME/.ssh/id_rsa",
    # 인터프리터·빌드 도구·쓰기 명령
    "python -c 1", "uv run pytest", "node x.js", "make", "bash -c ls", "rm -rf x", "find . -delete",
    "echo hi", "git commit -m x", "git checkout .", "git push",
    # git 전역 옵션과 파일 쓰기 옵션
    "git -c core.fsmonitor=evil status", "git -C /other status", "git --no-pager log",
    "git diff --output=patch.txt", "git log --output=x",
    # 작업 폴더 밖, .env, 와일드카드
    "cat ../secret", "cat ..\\..\\secret", "cat /etc/passwd", "cat C:\\Windows\\win.ini", "cat ~/.ssh/id_rsa",
    "ls ..", "cat .env", "cat sub/.env.local", "git show HEAD:.env", "git diff -- .env", "cat *", "head .e*",
    # 파싱 실패, 빈 명령
    "cat 'unclosed", "", "   ",
])
def test_bypass_attempts_require_approval(command: str) -> None:
    assert not is_read_only_command(command)


def test_decide_all_keeps_request_order() -> None:
    """병렬 도구 호출은 한 interrupt 에 여러 요청으로 온다. 결정은 같은 순서여야 한다."""
    asked: list[str] = []

    def ask(act: dict) -> dict:
        asked.append(act["args"]["command"] if act["name"] == "execute" else act["args"]["file_path"])
        return reject("싫음") if act["name"] == "execute" else APPROVE

    decisions = decide_all([EDIT, GIT_STATUS, PYTEST, WRITE], "manual", ask)
    assert [d["type"] for d in decisions] == ["approve", "approve", "reject", "approve"]
    assert asked == ["/src/app.py", "uv run pytest -q", "/new.py"]  # git status 는 묻지 않았다
    assert "싫음" in decisions[2]["message"]


def test_on_decision_reports_each_decision_in_order() -> None:
    seen: list[tuple[str, str, bool, str]] = []
    decide_all([GIT_STATUS, EDIT, PYTEST], "auto-edit", lambda act: reject("느림"),
               on_decision=lambda act, d, auto: seen.append((act["name"], d["type"], auto, reject_reason(d))))
    assert seen == [("execute", "approve", True, ""), ("edit_file", "approve", True, ""),
                    ("execute", "reject", False, "느림")]


def test_reject_reason() -> None:
    assert reject_reason(reject("다른 방법")) == "다른 방법"
    assert reject_reason(reject()) == ""
    assert reject_reason(APPROVE) == ""


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        decide_all([EDIT], "auto", lambda a: APPROVE)
