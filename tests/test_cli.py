"""CLI: 승인 프롬프트, 슬래시 커맨드, 세션 저장·재개, buggy_repo 수정 흐름. 모델은 대본이다."""
from __future__ import annotations

import io
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest
from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from test_usage import ai

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
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def make_app(tmp_path: Path, steps: list, *answers: str, workspace: Path | None = None,
             mode: str = "manual", tracer=None,
             model: ScriptedChatModel | None = None) -> tuple[App, ScriptedChatModel, io.StringIO, Script]:
    """model 을 주면 steps 대신 그 모델(예: StreamingChatModel)을 쓴다."""
    if workspace is None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
    settings = Settings(data_dir=tmp_path / "data")
    sessions = Sessions(sqlite3.connect(tmp_path / "sessions.db", check_same_thread=False))
    model = model if model is not None else ScriptedChatModel(steps=steps)
    agent = build_agent(settings, workspace, SqliteSaver(sessions.conn), model=model)
    out, script = io.StringIO(), Script(*answers)
    app = App(settings, workspace, mode, sessions, sessions.create(workspace),
              Console(file=out, width=120, highlight=False), script, agent=agent, tracer=tracer)
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


def test_usage_line_and_cost_command(tmp_path: Path) -> None:
    """턴마다 사용량 한 줄을 보이고, /cost 는 sessions.db 에 쌓인 턴 합계를 보인다."""
    steps = [ai("하나", inp=1000, out=10, cost=0.01), ai("둘", inp=2000, out=20, cache_read=900, cost=0.002)]
    app, _, out, _ = make_app(tmp_path, steps, "/cost", "첫 질문", "두 번째 질문", "/cost", "/exit")
    app.run()
    text = out.getvalue()
    assert "이 세션에 기록된 사용량이 아직 없다" in text
    assert "모델 1회 · 입력 1,000 · 출력 10 · $0.0100" in text
    assert "모델 1회 · 입력 2,000 (캐시 읽기 900 · 쓰기 0) · 출력 20 · $0.0020" in text
    assert f"세션 {app.thread_id} · 턴 2개 · 모델 2회 · 입력 3,000 (캐시 읽기 900 · 쓰기 0) · 출력 30 · $0.0120" in text
    total, turns = app.sessions.usage_of(app.thread_id)
    assert (turns, total.calls, total.output_tokens) == (2, 2, 30)


def test_interrupted_turn_still_records_usage(tmp_path: Path) -> None:
    """승인 프롬프트에서 Ctrl+C 로 턴을 끊어도 이미 쓴 모델 호출은 남는다."""
    step = ai(inp=500, out=5, cost=0.004, tool_calls=[
        {"name": "execute", "args": {"command": "uv run pytest"}, "id": "c1", "type": "tool_call"}])
    app, _, out, _ = make_app(tmp_path, [step], "테스트 돌려", KeyboardInterrupt(), "/exit")
    app.run()
    assert "중단했다" in out.getvalue()
    total, turns = app.sessions.usage_of(app.thread_id)
    assert (turns, total.calls, round(total.cost, 6)) == (1, 1, 0.004)


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


ENV_A = "\n# 이 PC 의 도구 (세션 시작 때 확인했다)\n- 있음: python 3.13.18, uv 0.13.7\n- 없음: node\n"


def run_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], *answers: str,
             detected: str | None, model: ScriptedChatModel) -> int:
    """main() 을 키·네트워크 없이 돌린다. detected 가 None 이면 환경 탐지가 불리면 안 된다."""
    from code_agent import cli

    def detect(shell: str, workspace: Path | None = None) -> str:
        assert detected is not None, "--resume 인데 환경을 다시 탐지했다"
        return detected

    monkeypatch.setattr(cli, "load_settings", lambda: Settings(data_dir=tmp_path / "data"))
    monkeypatch.setattr(cli, "detect_environment", detect)
    monkeypatch.setattr(cli, "make_tracer", lambda settings: None)
    monkeypatch.setattr(cli, "_line_reader", lambda history: Script(*answers))
    monkeypatch.setattr(cli, "build_agent", lambda *args, **kwargs: build_agent(*args, model=model, **kwargs))
    return cli.main(argv)


def only_thread(tmp_path: Path) -> str:
    with sqlite3.connect(tmp_path / "data" / "sessions.db") as conn:
        [(thread_id,)] = conn.execute("SELECT thread_id FROM code_agent_threads").fetchall()
    return thread_id


def test_resume_reuses_stored_env_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """그사이 node 를 설치해 탐지 결과가 바뀌어도 --resume 의 system 은 첫 세션과 바이트 단위로 같다."""
    (tmp_path / "ws").mkdir()
    first = ScriptedChatModel(steps=[AIMessage(content="a")])
    assert run_main(tmp_path, monkeypatch, ["--workspace", str(tmp_path / "ws")], "하이", "/exit",
                    detected=ENV_A, model=first) == 0
    thread_id = only_thread(tmp_path)
    resumed = ScriptedChatModel(steps=[AIMessage(content="b")])  # 다시 탐지하면 detect 가 단언으로 실패한다
    assert run_main(tmp_path, monkeypatch, ["--resume", thread_id], "또", "/exit", detected=None, model=resumed) == 0
    assert ENV_A in first.calls[0][0].text  # deepagents 가 우리 프롬프트 뒤에 자기 절을 덧붙인다
    assert resumed.calls[0][0].text == first.calls[0][0].text


def test_old_thread_without_env_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """env 절이 생기기 전의 thread 를 이어 가면 절 없이(옛 system 과 같게) 쓰고 다시 탐지하지 않는다."""
    (tmp_path / "ws").mkdir()
    (tmp_path / "data").mkdir()
    with sqlite3.connect(tmp_path / "data" / "sessions.db") as conn:  # 옛 버전이 만든 표와 행
        conn.execute("CREATE TABLE code_agent_threads (thread_id TEXT PRIMARY KEY, workspace TEXT NOT NULL, "
                     "created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO code_agent_threads VALUES ('old1', ?, 'then')", (str(tmp_path / "ws"),))
    model = ScriptedChatModel(steps=[AIMessage(content="b")])
    assert run_main(tmp_path, monkeypatch, ["--resume", "old1"], "또", "/exit", detected=None, model=model) == 0
    assert "이 PC 의 도구" not in model.calls[0][0].text


def test_new_command_reuses_env_text(tmp_path: Path) -> None:
    app, _, _, _ = make_app(tmp_path, [], "/new", "/exit")
    app.env_text = ENV_A
    first = app.thread_id
    app.run()
    assert app.thread_id != first and app.sessions.env_text_of(app.thread_id) == ENV_A


def test_new_after_old_thread_redetects(tmp_path: Path) -> None:
    """절이 없던 옛 thread 를 이어 온 뒤 /new 하면, 다시 탐지한 절로 새 agent 를 만들어 새 thread 를 시작한다."""
    app, _, _, _ = make_app(tmp_path, [], "/new", "새 질문", "/exit")
    model = ScriptedChatModel(steps=[AIMessage(content="b")])
    fresh = build_agent(app.settings, app.workspace, SqliteSaver(app.sessions.conn), model=model, env_text=ENV_A)
    app.fresh_agent = lambda: (ENV_A, fresh)
    app.run()
    assert app.sessions.env_text_of(app.thread_id) == ENV_A
    assert ENV_A in model.calls[0][0].text


def test_old_process_can_still_insert_threads(tmp_path: Path) -> None:
    """env 는 표를 따로 둬서 code_agent_threads 열이 그대로다. 아직 돌고 있는 옛 프로세스의 INSERT 가 깨지지 않는다."""
    conn = sqlite3.connect(tmp_path / "s.db")
    Sessions(conn).create(tmp_path, ENV_A)
    conn.execute("INSERT INTO code_agent_threads VALUES (?, ?, ?)", ("old2", str(tmp_path), "now"))  # 옛 코드의 문장
    assert Sessions(conn).env_text_of("old2") == ""
