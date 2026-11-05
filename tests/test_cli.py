"""CLI: 승인 프롬프트, 슬래시 커맨드, 세션 저장·재개, buggy_repo 수정 흐름. 모델은 대본이다."""
from __future__ import annotations

import io
import shutil
import sqlite3
import sys
from pathlib import Path

from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console

from code_agent.agent import build_agent
from code_agent.cli import App, Sessions, parse_args
from code_agent.settings import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "buggy_repo"


class Script:
    """read_line 대역. 미리 정한 답을 차례로 돌려주고, 받은 프롬프트를 남긴다."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def make_app(tmp_path: Path, steps: list, *answers: str, workspace: Path | None = None,
             mode: str = "manual") -> tuple[App, ScriptedChatModel, io.StringIO, Script]:
    if workspace is None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
    settings = Settings(data_dir=tmp_path / "data")
    sessions = Sessions(sqlite3.connect(tmp_path / "sessions.db", check_same_thread=False))
    model = ScriptedChatModel(steps=steps)
    agent = build_agent(settings, workspace, SqliteSaver(sessions.conn), model=model)
    out, script = io.StringIO(), Script(*answers)
    app = App(settings, workspace, mode, sessions, sessions.create(workspace),
              Console(file=out, width=120, highlight=False), script, agent=agent)
    return app, model, out, script


def edit_step(path: str = "/app.py") -> AIMessage:
    return tool_call("edit_file", {"file_path": path, "old_string": "1 + 1", "new_string": "1 + 2"})


def test_manual_mode_asks_and_reject_keeps_file(tmp_path: Path) -> None:
    app, model, out, script = make_app(tmp_path, [edit_step(), AIMessage(content="알겠다")],
                                       "고쳐줘", "", "n", "다른 방법으로")
    (app.workspace / "app.py").write_text("x = 1 + 1\n", encoding="utf-8")
    assert app.run() == 0
    assert (app.workspace / "app.py").read_text(encoding="utf-8") == "x = 1 + 1\n"
    text = out.getvalue()
    assert "승인 요청: edit_file" in text and "+x = 1 + 2" not in text  # diff 는 old/new 조각 기준
    assert "-1 + 1" in text and "+1 + 2" in text
    assert script.prompts.count("[y] 승인 / [n] 거부 › ") == 2  # 빈 입력은 다시 묻는다
    [rejected] = [m for m in model.calls[-1] if isinstance(m, ToolMessage)]
    assert "다른 방법으로" in rejected.text
    assert "알겠다" in text


def test_auto_edit_mode_skips_file_approval(tmp_path: Path) -> None:
    app, _, out, script = make_app(tmp_path, [edit_step(), AIMessage(content="고쳤다")],
                                   "/mode auto-edit", "고쳐줘", "/exit")
    (app.workspace / "app.py").write_text("x = 1 + 1\n", encoding="utf-8")
    app.run()
    assert (app.workspace / "app.py").read_text(encoding="utf-8") == "x = 1 + 2\n"
    assert "[y] 승인 / [n] 거부 › " not in script.prompts
    assert "승인 모드 → auto-edit" in out.getvalue()


def test_slash_commands(tmp_path: Path) -> None:
    todos = tool_call("write_todos", {"todos": [{"content": "테스트 읽기", "status": "in_progress"}]})
    app, _, out, _ = make_app(tmp_path, [todos, AIMessage(content="계획 세움")],
                              "/help", "/mode nope", "/mode yolo", "계획 세워", "/todos", "/diff", "/what", "/new",
                              "/exit")
    first = app.thread_id
    app.run()
    text = out.getvalue()
    assert "슬래시 커맨드" in text
    assert "모드는 manual, auto-edit, yolo 중 하나다" in text
    assert "yolo 모드" in text
    assert text.count("테스트 읽기") >= 2  # 턴 중 갱신 + /todos
    assert "알 수 없는 커맨드: /what" in text
    assert app.thread_id != first
    assert f"--resume {app.thread_id}" in text


def test_session_persists_across_apps(tmp_path: Path) -> None:
    """같은 sessions.db 와 thread 로 새 App 을 띄우면 이전 대화가 이어진다(--resume)."""
    app, _, _, _ = make_app(tmp_path, [AIMessage(content="첫 답")], "기억해: 파랑", "/exit")
    app.run()
    assert app.sessions.workspace_of(app.thread_id) == app.workspace

    model = ScriptedChatModel(steps=[AIMessage(content="파랑")])
    agent = build_agent(app.settings, app.workspace, SqliteSaver(app.sessions.conn), model=model)
    resumed = App(app.settings, app.workspace, "manual", app.sessions, app.thread_id,
                  Console(file=io.StringIO()), Script("뭐라고 했지?", "/exit"), agent=agent)
    resumed.run()
    humans = [m.text for m in model.calls[0] if isinstance(m, HumanMessage)]
    assert humans == ["기억해: 파랑", "뭐라고 했지?"]


def test_unknown_session_is_none(tmp_path: Path) -> None:
    sessions = Sessions(sqlite3.connect(tmp_path / "s.db"))
    assert sessions.workspace_of("nope") is None


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert (args.workspace, args.mode, args.resume) == (None, "manual", None)


def test_fixes_buggy_repo_with_approvals(tmp_path: Path) -> None:
    """MVP 완료 조건의 배선 확인: 실패 확인 → 읽기 → 수정 승인 → pytest 승인 → 통과. 실제 모델 E2E 는 수동으로 한다."""
    workspace = tmp_path / "repo"
    shutil.copytree(FIXTURE, workspace)
    pytest_cmd = f"'{Path(sys.executable).as_posix()}' -m pytest -q -p no:cacheprovider"
    steps = [
        tool_call("execute", {"command": pytest_cmd}),
        tool_call("read_file", {"file_path": "/stats.py"}),
        tool_call("edit_file", {"file_path": "/stats.py", "old_string": "ordered[mid + 1]",
                                "new_string": "ordered[mid - 1]"}),
        tool_call("execute", {"command": pytest_cmd}),
        AIMessage(content="짝수 길이 중앙값 인덱스를 고쳤다."),
    ]
    app, model, _out, script = make_app(tmp_path, steps, "실패하는 테스트를 고쳐줘", "y", "y", "y",
                                       workspace=workspace)
    app.run()
    results = [m.text for m in model.calls[-1] if isinstance(m, ToolMessage) and m.name == "execute"]
    assert "1 failed, 3 passed" in results[0]
    assert "4 passed" in results[-1]
    assert "ordered[mid - 1]" in (workspace / "stats.py").read_text(encoding="utf-8")
    assert script.prompts.count("[y] 승인 / [n] 거부 › ") == 3
    assert not (workspace / ".code-agent-artifacts").exists()
