"""로컬 트레이스: 사건 순서와 필드, 지표 한 줄, 실패·거부 도구, .env 미기록, 도메인 사건,
CompositeTracer, /trace 출력, CODE_AGENT_LOCAL_TRACE. 모델은 대본이고 네트워크는 쓰지 않는다.
"""
from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.config import get_stream_writer
from rich.console import Console
from test_cli import make_app, run_main
from test_stops import truncated
from test_tracing import FakeTracer
from test_usage import ai

from code_agent import settings as settings_module
from code_agent.localtrace import (
    METRICS_FILE,
    TRACE_FILE,
    LocalTracer,
    LocalTurn,
    read_jsonl,
    run_dir,
    trace_view,
)
from code_agent.settings import load_settings
from code_agent.tracing import CompositeTracer, LangfuseTurn, combine_tracers
from code_agent.usage import Usage

METRIC_KEYS = {"turn", "thread_id", "started_at", "duration_s", "status", "mode", "model",
               "model_calls", "tool_calls", "tool_errors", "retries", "rejections",
               "input_tokens", "output_tokens", "cache_read", "cache_write", "reasoning_tokens",
               "cost"}


def local_app(tmp_path: Path, steps: list, *answers, mode: str = "manual",
              model: ScriptedChatModel | None = None):
    """make_app 과 같은 data_dir(tmp_path/data)에 LocalTracer 를 붙인 App."""
    tracer = LocalTracer(tmp_path / "data")
    app, model, out, _ = make_app(tmp_path, steps, *answers, mode=mode, tracer=tracer,
                                  model=model)
    return app, model, out


def folder(app) -> Path:
    return run_dir(app.settings.data_dir, app.thread_id)


def events(app) -> list[dict]:
    return read_jsonl(folder(app) / TRACE_FILE)


def metrics(app) -> list[dict]:
    return read_jsonl(folder(app) / METRICS_FILE)


def raw(app) -> str:
    return "".join((folder(app) / name).read_text(encoding="utf-8")
                   for name in (TRACE_FILE, METRICS_FILE))


def of_type(rows: list[dict], kind: str) -> list[dict]:
    return [e for e in rows if e["type"] == kind]


def test_one_turn_events_order_fields_and_metrics(tmp_path: Path) -> None:
    steps = [ai(inp=100, out=5, cache_read=40, cost=0.001,
                tool_calls=[{"name": "read_file", "args": {"file_path": "/a.txt"}, "id": "r1",
                             "type": "tool_call"}]),
             ai("다 읽었다", inp=200, out=7, cost=0.002)]
    app, _, _ = local_app(tmp_path, steps, "  a.txt   읽어 줘 ", "/exit")
    (app.workspace / "a.txt").write_text("BODY-ALPHA\nBODY-BETA\n", encoding="utf-8")
    app.run()

    rows = events(app)
    assert [e["type"] for e in rows] == ["turn_start", "llm", "tool", "llm", "turn_end"]
    for event in rows:
        assert event["turn"] == 1
        assert datetime.fromisoformat(event["ts"]).tzinfo == UTC
        assert isinstance(event["t"], float) and event["t"] >= 0
    start, first, tool, second, end = rows
    assert start["request"] == "a.txt 읽어 줘"  # 공백을 합친 앞 200자
    assert (start["mode"], start["model"], start["thread_id"]) == (
        "manual", app.settings.model, app.thread_id)
    assert start["workspace"] == str(app.workspace)
    assert (first["status"], first["node"], first["tool_calls"]) == ("ok", "model", 1)
    assert (first["input_tokens"], first["output_tokens"], first["cache_read"]) == (100, 5, 40)
    assert first["cost"] == 0.001 and first["duration_s"] >= 0
    assert (tool["name"], tool["call"], tool["status"]) == ("read_file", "read_file /a.txt", "ok")
    assert tool["result"] == "1-2줄 읽음 (전체 2줄)" and tool["duration_s"] >= 0
    assert "sub" not in tool and "error" not in tool
    assert second["tool_calls"] == 0
    assert end["status"] == "ok" and end["duration_s"] >= 0
    assert "BODY-ALPHA" not in raw(app)  # 도구 출력 원문은 남기지 않는다

    [line] = metrics(app)
    assert set(line) == METRIC_KEYS
    assert (line["turn"], line["thread_id"], line["status"]) == (1, app.thread_id, "ok")
    assert (line["model_calls"], line["tool_calls"], line["tool_errors"]) == (2, 1, 0)
    assert (line["retries"], line["rejections"]) == (0, 0)
    assert (line["input_tokens"], line["output_tokens"], line["cache_read"]) == (300, 12, 40)
    assert line["cost"] == 0.003 and line["duration_s"] == end["duration_s"]
    assert line["started_at"] == start["ts"]


def test_failing_tool_records_status_and_exit_code(tmp_path: Path) -> None:
    steps = [tool_call("execute", {"command": "exit 3"}), AIMessage(content="실패했다")]
    app, _, _ = local_app(tmp_path, steps, "돌려", "/exit", mode="yolo")
    app.run()
    [tool] = of_type(events(app), "tool")
    assert (tool["status"], tool["exit_code"]) == ("error", 3)
    assert tool["error"].startswith("exit 3") and "result" not in tool
    [approval] = of_type(events(app), "approval")
    assert (approval["decision"], approval["by"], approval["call"]) == ("approve", "auto",
                                                                         "$ exit 3")
    [line] = metrics(app)
    assert (line["tool_calls"], line["tool_errors"], line["status"]) == (1, 1, "ok")


def test_rejected_call_is_marked_rejected_not_error(tmp_path: Path) -> None:
    steps = [tool_call("write_file", {"file_path": "/x.txt", "content": "x"}),
             AIMessage(content="알겠다")]
    app, _, _ = local_app(tmp_path, steps, "써 줘", "n", "쓰지 마", "/exit")
    app.run()
    rows = events(app)
    [approval] = of_type(rows, "approval")
    assert (approval["tool"], approval["decision"], approval["by"], approval["reason"]) == (
        "write_file", "reject", "user", "쓰지 마")
    [tool] = of_type(rows, "tool")
    assert tool["status"] == "rejected" and "duration_s" not in tool  # 실행되지 않았다
    assert tool["error"] == "거부됨"  # 사유는 approval 사건에 있다
    [line] = metrics(app)
    assert (line["rejections"], line["tool_errors"]) == (1, 0)


def test_env_contents_never_written(tmp_path: Path) -> None:
    steps = [tool_call("read_file", {"file_path": "/.env"}),
             tool_call("execute", {"command": "cat .env"}),
             tool_call("grep", {"pattern": "SECRET", "path": "/", "output_mode": "content"}),
             AIMessage(content="끝")]
    app, model, _ = local_app(tmp_path, steps, "비밀 봐", "y", "/exit")
    (app.workspace / ".env").write_text("SECRET=topsecret-123\n", encoding="utf-8")
    app.run()
    # 사람이 cat .env 를 승인해 모델은 값을 받았다. 그래도 트레이스에는 남지 않는다
    assert any("topsecret-123" in m.text for call in model.calls for m in call
               if isinstance(m, ToolMessage))
    assert "topsecret-123" not in raw(app)
    blocked, cat, grep = of_type(events(app), "tool")
    assert (blocked["call"], blocked["status"]) == ("read_file /.env", "error")
    assert "권한 거부" in blocked["error"]
    assert (cat["status"], cat["exit_code"], cat["result"]) == ("ok", 0, "exit 0")
    assert grep["status"] == "ok"


def test_custom_stream_event_lands_in_trace(tmp_path: Path) -> None:
    """get_stream_writer 로 보낸 {"type": ...} 사건을 그대로 남긴다.

    직렬화되지 않는 필드는 빼고, 기본 필드(turn 등)는 덮이지 않는다.
    """
    def plan(messages: list[BaseMessage]) -> AIMessage:
        get_stream_writer()({"type": "plan_written", "version": 2, "files": ["a.py", "b/"],
                             "turn": 99, "obj": object()})
        get_stream_writer()({"no_type": True})  # type 이 없으면 사건이 아니다
        return AIMessage(content="계획 씀")

    app, _, _ = local_app(tmp_path, [plan], "계획 세워", "/exit")
    app.run()
    rows = events(app)
    [event] = of_type(rows, "plan_written")
    assert (event["turn"], event["version"], event["files"]) == (1, 2, ["a.py", "b/"])
    assert "obj" not in event and "sub" not in event
    # 도메인 사건은 스트림으로 받은 때 남긴다. 콜백으로 쓰는 llm 과의 앞뒤는 정하지 않는다
    types = [e["type"] for e in rows]
    assert (types[0], types[-1]) == ("turn_start", "turn_end")
    assert sorted(types[1:-1]) == ["llm", "plan_written"]


def test_subagent_events_are_marked(tmp_path: Path) -> None:
    """서브에이전트(네임스페이스가 있는 스트림, 중첩 checkpoint_ns)의 사건에
    sub·agent 가 붙는다.
    """
    turn = LocalTurn(tmp_path, 1, "요청", thread_id="t")
    [callback] = turn.callbacks
    run_id = uuid4()
    callback.on_chat_model_start({}, [[]], run_id=run_id, metadata={
        "langgraph_node": "model", "lc_agent_name": "plan-reviewer",
        "langgraph_checkpoint_ns": "tools:a|model:b"})
    callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage("x"))]]),
                        run_id=run_id)
    ns = ("tools:a",)
    call = {"name": "ls", "args": {"path": "/"}, "id": "s1", "type": "tool_call"}
    turn.on_part((ns, "updates", {"model": {"messages": [AIMessage("", tool_calls=[call])]}}))
    turn.on_part((ns, "updates", {"tools": {"messages": [
        ToolMessage("a.py", tool_call_id="s1", name="ls")]}}))
    turn.on_part((ns, "custom", {"type": "memory_loaded", "chars": 10}))
    turn.end(output="", usage=Usage())
    rows = read_jsonl(tmp_path / TRACE_FILE)
    [llm] = of_type(rows, "llm")
    assert (llm["sub"], llm["agent"]) == ("tools:a", "plan-reviewer")
    [tool] = of_type(rows, "tool")
    assert (tool["sub"], tool["result"]) == ("tools:a", "a.py")
    [memory] = of_type(rows, "memory_loaded")
    assert (memory["sub"], memory["chars"]) == ("tools:a", 10)


def test_tools_without_streamed_result_are_closed_at_end(tmp_path: Path) -> None:
    """실행은 끝났지만 결과가 스트림에 오지 않은 도구(예외·Ctrl+C 로 끝난 턴)도 턴 끝에 남긴다."""
    turn = LocalTurn(tmp_path, 1, "요청", thread_id="t")
    [callback] = turn.callbacks
    calls = [{"name": "execute", "args": {"command": "sleep 100"}, "id": "k1",
              "type": "tool_call"},
             {"name": "ls", "args": {"path": "/"}, "id": "k2", "type": "tool_call"},
             {"name": "glob", "args": {"pattern": "*"}, "id": "k3", "type": "tool_call"}]
    turn.on_part(((), "updates", {"model": {"messages": [AIMessage("", tool_calls=calls)]}}))
    for call, error in zip(calls, (KeyboardInterrupt(), RuntimeError("터짐"), None),
                           strict=True):
        run_id = uuid4()
        callback.on_tool_start({"name": call["name"]}, "", run_id=run_id,
                               tool_call_id=call["id"])
        if error is None:
            callback.on_tool_end("ok", run_id=run_id)
        else:
            callback.on_tool_error(error, run_id=run_id)
    turn.end(output="", usage=Usage(), error="사용자가 중단", interrupted=True)
    sleep, ls, glob = of_type(read_jsonl(tmp_path / TRACE_FILE), "tool")
    assert (sleep["call"], sleep["status"], sleep["error"]) == (
        "$ sleep 100", "interrupted", "KeyboardInterrupt")
    assert (ls["status"], ls["error"]) == ("error", "RuntimeError: 터짐")
    assert (glob["status"], glob["result"]) == ("ok", "(결과를 받기 전에 턴이 끝났다)")
    [line] = read_jsonl(tmp_path / METRICS_FILE)
    assert (line["tool_calls"], line["tool_errors"], line["status"]) == (3, 1, "interrupted")


def test_recovery_retries_and_duration_in_metrics(tmp_path: Path) -> None:
    app, _, _ = local_app(tmp_path, [truncated(), AIMessage(content="다 썼다")], "만들어",
                          "/exit")
    app.run()
    rows = events(app)
    assert [e["type"] for e in rows] == ["turn_start", "llm", "recovery", "llm", "warning",
                                         "turn_end"]
    first = rows[1]
    assert (first["finish_reason"], first["output_tokens"]) == ("length", 64000)
    assert rows[2]["reason"] == "length" and rows[2]["reasoning_chars"] > 0
    assert "자동으로 한 번 다시 요청해 이어 갔다" in rows[4]["message"]
    [line] = metrics(app)
    assert (line["retries"], line["status"], line["model_calls"]) == (1, "warning", 2)
    assert isinstance(line["duration_s"], float) and line["duration_s"] >= 0


def test_interrupted_and_failed_turns(tmp_path: Path) -> None:
    app, _, _ = local_app(tmp_path, [tool_call("execute", {"command": "uv run pytest"})],
                          "돌려", KeyboardInterrupt(), "/exit")
    app.run()
    rows = events(app)
    assert of_type(rows, "turn_end")[0]["status"] == "interrupted"
    assert not of_type(rows, "error")  # 사용자 중단은 오류로 세지 않는다
    assert metrics(app)[0]["status"] == "interrupted"

    (tmp_path / "b").mkdir()
    failed, _, _ = local_app(tmp_path / "b", [], "아무거나", "/exit")  # 대본이 비어 실패
    failed.run()
    rows = events(failed)
    [error] = of_type(rows, "error")
    assert error["message"].startswith("ScriptExhaustedError")
    [llm] = of_type(rows, "llm")
    assert llm["status"] == "error" and llm["error"].startswith("ScriptExhaustedError")
    assert metrics(failed)[0]["status"] == "error"


def test_turn_numbers_continue_across_processes(tmp_path: Path) -> None:
    for expected in (1, 2):
        tracer = LocalTracer(tmp_path)  # 프로세스마다 새로 만든다
        with tracer.turn("하이", session_id="th1", tags=[]) as turn:
            turn.end(output="", usage=Usage())
        assert read_jsonl(run_dir(tmp_path, "th1") / METRICS_FILE)[-1]["turn"] == expected
    with (run_dir(tmp_path, "th1") / TRACE_FILE).open("a", encoding="utf-8") as file:
        file.write('{"ts": "2026-11-09T00:00:00", "turn": 3, "ty')  # 강제 종료로 반만 쓴 줄
    tracer = LocalTracer(tmp_path)
    with tracer.turn("또", session_id="th1", tags=[]) as turn:
        assert turn.number == 3
    # thread_id 는 폴더 이름으로 안전한 글자만 남긴다
    assert run_dir(tmp_path, "../x") == tmp_path / "runs" / "_x"
    assert run_dir(tmp_path, "a/../b").parent == tmp_path / "runs"


def test_write_failure_does_not_break_turn(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    tracer = LocalTracer(blocker)  # runs 폴더를 만들 수 없다
    app, _, out, _ = make_app(tmp_path, [AIMessage(content="안녕")], "인사", "/exit",
                              tracer=tracer)
    assert app.run() == 0
    assert "안녕" in out.getvalue()


def test_composite_tracer_fans_out(tmp_path: Path) -> None:
    first, second = FakeTracer(), FakeTracer()
    composite = combine_tracers([None, first, second])
    assert isinstance(composite, CompositeTracer)
    # 하나여도 묶는다(그 트레이서의 예외가 턴을 깨지 않게). 없으면 None
    assert isinstance(combine_tracers([None, first]), CompositeTracer)
    assert combine_tracers([None]) is None
    steps = [tool_call("execute", {"command": "uv run pytest"}), AIMessage(content="끝")]
    app, _, out, _ = make_app(tmp_path, steps, "돌려", "n", "", "/exit", tracer=composite)
    app.run()
    for fake in (first, second):
        [turn] = fake.turns
        assert turn.decisions == [("uv run pytest", "reject", False, "")]
        assert turn.ended["output"] == "끝" and turn.ended["retries"] == 0
        assert turn.handler.root_starts == 2  # 처음과 재개가 모두 두 핸들러를 거쳤다
        assert turn.parts > 0
        assert fake.shut
    assert "Langfuse 트레이스 켜짐" in out.getvalue()


def test_banner_shows_local_folder_with_and_without_langfuse(tmp_path: Path) -> None:
    local = LocalTracer(tmp_path / "data")
    app, _, out, _ = make_app(tmp_path, [], "/exit", tracer=local)
    app.run()
    text = out.getvalue()
    assert "Langfuse" not in text
    assert "로컬 트레이스 →" in text and app.thread_id in text

    (tmp_path / "b").mkdir()
    app, _, out, _ = make_app(tmp_path / "b", [], "/exit",
                              tracer=CompositeTracer([FakeTracer(), local]))
    app.run()
    assert "Langfuse 트레이스 켜짐" in out.getvalue() and "로컬 트레이스 →" in out.getvalue()


def test_langfuse_turn_records_domain_events_duration_and_retries() -> None:
    class Root:
        def __init__(self) -> None:
            self.fields: dict = {}
            self.events: list[dict] = []

        def update(self, **fields) -> None:
            self.fields = fields

        def create_event(self, **fields) -> None:
            self.events.append(fields)

    root = Root()
    turn = LangfuseTurn(root, handler=None)
    turn.on_part(((), "custom", {"type": "plan_reviewed", "verdict": "revise", "x": object()}))
    turn.on_part(((), "custom", {"no_type": 1}))
    turn.end(output="끝", usage=Usage(), retries=1, interrupted=True)
    assert root.events == [{"name": "plan_reviewed", "input": {"verdict": "revise"},
                            "level": "DEFAULT"}]
    metadata = root.fields["metadata"]
    assert metadata["retries"] == 1 and metadata["interrupted"] is True
    assert isinstance(metadata["duration_s"], float) and metadata["duration_s"] >= 0


def test_trace_commands_render(tmp_path: Path) -> None:
    steps = [tool_call("execute", {"command": "exit 3"}), AIMessage(content="실패했다"),
             ai("둘째 답", inp=10, out=2, cost=0.5)]
    app, _, out = local_app(tmp_path, steps, "/trace", "돌려", "또", "/trace", "/trace 1",
                            "/trace errors", "/trace 9", "/trace x", "/exit", mode="yolo")
    app.run()
    text = out.getvalue()
    assert "이 세션의 로컬 트레이스가 아직 없다" in text  # 첫 턴 전
    last = text.index("턴 2 · 또")
    assert "turn_start" in text[last:] and "turn_end" in text[last:]
    assert "재시도 0" in text[last:] and "$0.5000" in text[last:] and "상태 ok" in text[last:]
    first = text.index("턴 1 · 돌려")
    assert "$ exit 3" in text[first:] and "exit 3 ·" in text[first:]
    assert "도구 1개(오류 1)" in text[first:]
    assert "이 세션의 실패·경고 1건" in text
    assert "턴 9 의 트레이스가 없다(있는 턴 1~2)" in text
    assert "사용법: /trace [n|errors]" in text


def test_trace_failure_rows_are_red(tmp_path: Path) -> None:
    app, _, _ = local_app(tmp_path, [tool_call("execute", {"command": "exit 3"}),
                                     AIMessage(content="끝")], "돌려", "/exit", mode="yolo")
    app.run()
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, force_terminal=True, color_system="standard")
    console.print(trace_view(folder(app)))
    lines = buffer.getvalue().splitlines()
    [failed] = [line for line in lines if "$ exit 3 → exit 3" in line]
    assert "\x1b[31m" in failed  # 빨강
    [approved] = [line for line in lines if "$ exit 3 → 승인(자동)" in line]
    assert "\x1b[31m" not in approved


def test_trace_when_disabled_or_missing(tmp_path: Path) -> None:
    app, _, out, _ = make_app(tmp_path, [], "/trace", "/exit")
    app.run()
    assert "이 세션의 로컬 트레이스가 아직 없다" in out.getvalue()
    app.settings = settings_module.Settings(data_dir=tmp_path / "data", local_trace=False)
    app.console = Console(file=(buffer := io.StringIO()), width=120)
    app.command("/trace")
    assert "로컬 트레이스가 꺼져 있다(CODE_AGENT_LOCAL_TRACE=0)" in buffer.getvalue()


def test_main_builds_local_tracer_without_langfuse(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "ws").mkdir()
    model = ScriptedChatModel(steps=[AIMessage(content="a")])
    assert run_main(tmp_path, monkeypatch, ["--workspace", str(tmp_path / "ws")], "하이",
                    "/exit", detected="", model=model) == 0
    [thread] = (tmp_path / "data" / "runs").iterdir()
    assert [m["status"] for m in read_jsonl(thread / METRICS_FILE)] == ["ok"]
    printed = capsys.readouterr().out
    assert "로컬 트레이스 →" in printed and "Langfuse" not in printed


def test_main_local_trace_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "ws").mkdir()
    model = ScriptedChatModel(steps=[AIMessage(content="a")])
    off = settings_module.Settings(data_dir=tmp_path / "data", local_trace=False)
    assert run_main(tmp_path, monkeypatch, ["--workspace", str(tmp_path / "ws")], "하이",
                    "/trace", "/exit", detected="", model=model, settings=off) == 0
    assert not (tmp_path / "data" / "runs").exists()
    printed = capsys.readouterr().out
    assert "로컬 트레이스 →" not in printed
    assert "로컬 트레이스가 꺼져 있다" in printed


@pytest.mark.parametrize(("raw_value", "expected"), [
    (None, True), ("", True), ("1", True), ("0", False), ("false", False), ("OFF", False),
    ("no", False)])
def test_local_trace_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                             raw_value: str | None, expected: bool) -> None:
    monkeypatch.setattr(settings_module, "REPO_ROOT", tmp_path)  # 저장소 .env 를 읽지 않는다
    if raw_value is None:
        monkeypatch.delenv("CODE_AGENT_LOCAL_TRACE", raising=False)
    else:
        monkeypatch.setenv("CODE_AGENT_LOCAL_TRACE", raw_value)
    assert load_settings(tmp_path).local_trace is expected


def test_jsonl_lines_are_plain_json(tmp_path: Path) -> None:
    """한 줄에 객체 하나, 한글은 그대로(ensure_ascii=False), 줄 끝은 LF 다."""
    app, _, _ = local_app(tmp_path, [AIMessage(content="안녕")], "인사해", "/exit")
    app.run()
    data = (folder(app) / TRACE_FILE).read_bytes()
    assert b"\r\n" not in data
    lines = data.decode("utf-8").splitlines()
    assert all(isinstance(json.loads(line), dict) for line in lines)
    assert "인사해" in lines[0]
