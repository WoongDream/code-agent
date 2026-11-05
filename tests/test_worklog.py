"""작업 기록: 도구 결과 요약, 호출과 결과 짝짓기, 스냅샷과 diff, 길이 제한."""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from code_agent import worklog
from code_agent.approval import APPROVE, reject
from code_agent.worklog import WorkLog, diff_snapshots, snapshot, summarize_result


def tool_message(name: str, text: str, *, status: str = "success", call_id: str = "c1") -> ToolMessage:
    return ToolMessage(content=text, name=name, tool_call_id=call_id, status=status)


@pytest.mark.parametrize(("message", "expected"), [
    (tool_message("execute", "4 passed in 0.1s\n\n[Command succeeded with exit code 0]"), "exit 0 · 4 passed in 0.1s"),
    (tool_message("execute", "E   assert 3.5 == 2.5\n1 failed\n\nExit code: 1"), "exit 1 · 1 failed"),
    (tool_message("execute", "<no output>"), "exit ? · <no output>"),
    (tool_message("read_file", "@@ lines 1-3 of 3 @@\nx\ny\nz"), "1-3줄 읽음 (전체 3줄)"),
    (tool_message("read_file", "@@ lines 101-200 of 950 @@\n..."), "101-200줄 읽음 (전체 950줄)"),
    (tool_message("read_file", "x\ny"), "2줄 읽음"),
    (tool_message("edit_file", "Successfully replaced 1 instance(s) in /a.py"), "Successfully replaced 1 instance(s) in /a.py"),
    (tool_message("read_file", "Error: 권한 거부\n자세히", status="error"), "오류: Error: 권한 거부"),
    (tool_message("ls", ""), "(출력 없음)"),
])
def test_summarize_result(message: ToolMessage, expected: str) -> None:
    assert summarize_result(message) == expected


def test_long_lines_are_clipped() -> None:
    summary = summarize_result(tool_message("grep", "x" * 1000))
    assert len(summary) == worklog.MAX_LINE and summary.endswith("…")


def updates(node: str, *messages) -> tuple:
    return ((), "updates", {node: {"messages": list(messages)}})


def test_worklog_pairs_calls_with_results_and_approvals() -> None:
    log = WorkLog()
    calls = [{"name": "execute", "args": {"command": "uv run pytest"}, "id": "a", "type": "tool_call"},
             {"name": "read_file", "args": {"file_path": "/x.py"}, "id": "b", "type": "tool_call"}]
    log.on_part(updates("model", AIMessage(content="", tool_calls=calls)))
    log.on_part(updates("tools", tool_message("read_file", "@@ lines 1-1 of 1 @@\nx", call_id="b")))  # 순서가 바뀌어도 짝을 찾는다
    log.on_part(updates("tools", tool_message("execute", "ok\n[Command succeeded with exit code 0]", call_id="a")))
    log.on_part(((), "messages", ("무시", {})))
    log.decision(calls[0], reject("느림"), auto=False)
    log.decision({"name": "execute", "args": {"command": "git status"}}, APPROVE, auto=True)
    text = log.render(answer="끝", changes=["M x.py (+1 -1)"], diff="-a\n+b")
    assert "1. $ uv run pytest → exit 0 · ok" in text
    assert "2. read_file /x.py → 1-1줄 읽음 (전체 1줄)" in text
    assert "- $ uv run pytest → 거부(사용자): 느림" in text
    assert "- $ git status → 승인(자동)" in text
    assert "- M x.py (+1 -1)" in text and "```diff\n-a\n+b\n```" in text


def test_render_marks_skipped_snapshot_and_empty_sections() -> None:
    text = WorkLog().render(answer="", changes=None, diff="")
    assert "## 최종 답변\n(없음)" in text
    assert "## 실행한 도구 (0개)\n(없음)" in text
    assert "(작업 폴더가 커서 확인하지 않았다)" in text
    assert "## diff\n(없음)" in text


def test_steps_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worklog, "MAX_STEPS", 2)
    log = WorkLog()
    calls = [{"name": "ls", "args": {"path": f"/{i}"}, "id": str(i), "type": "tool_call"} for i in range(5)]
    log.on_part(updates("model", AIMessage(content="", tool_calls=calls)))
    text = log.render(answer="끝", changes=[], diff="")
    assert "## 실행한 도구 (5개)" in text and "3. ls" not in text and "도구 호출 5개 중 2개까지" in text


def test_snapshot_skips_env_and_heavy_dirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("a", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / ".env.example").write_text("SECRET=", encoding="utf-8")
    for skipped in ("node_modules", ".git", ".venv"):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / "x").write_text("x", encoding="utf-8")
    assert snapshot(tmp_path) == {"src/app.js": b"a", ".env.example": b"SECRET="}


def test_snapshot_gives_up_on_large_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worklog, "MAX_FILES", 2)
    for i in range(3):
        (tmp_path / f"{i}.txt").write_text("x", encoding="utf-8")
    assert snapshot(tmp_path) is None


def test_diff_snapshots() -> None:
    before = {"keep.txt": b"same", "edit.txt": b"a\nb\n", "gone.txt": b"x\n", "img.png": b"\x89PNG\xff"}
    after = {"keep.txt": b"same", "edit.txt": b"a\nc\n", "new.txt": b"hi\n", "img.png": b"\x89PNG\xfe"}
    changes, diff = diff_snapshots(before, after)
    assert changes == ["M edit.txt (+1 -1)", "D gone.txt (+0 -1)", "M img.png (바이너리)", "A new.txt (+1 -0)"]
    assert "-b" in diff and "+c" in diff and "+hi" in diff and "keep.txt" not in diff


def test_diff_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worklog, "MAX_DIFF_CHARS", 50)
    _, diff = diff_snapshots({}, {"big.txt": ("x" * 100 + "\n").encode() * 3})
    assert diff.endswith("자 중 50자까지)") and len(diff) < 120
