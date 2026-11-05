"""Langfuse 트레이스: 켜는 조건, 턴·승인·사용량 기록, 재개 때 같은 핸들러 재사용, 실제 SDK 배선. 네트워크는 쓰지 않는다."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fakes import tool_call
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from test_cli import make_app
from test_usage import ai

from code_agent import tracing
from code_agent.approval import reject_reason
from code_agent.settings import Settings
from code_agent.tracing import LangfuseTracer, make_tracer


class RootCounter(BaseCallbackHandler):
    """그래프 실행(stream 호출) 하나마다 루트 체인 시작이 한 번 온다."""

    def __init__(self) -> None:
        self.root_starts = 0

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs) -> None:
        if parent_run_id is None:
            self.root_starts += 1


class FakeTurn:
    def __init__(self, text: str, session_id: str, tags: list[str]) -> None:
        self.text, self.session_id, self.tags = text, session_id, tags
        self.handler = RootCounter()
        self.callbacks = [self.handler]
        self.decisions: list[tuple[str, str, bool, str]] = []
        self.ended: dict | None = None

    def decision(self, action: dict, decision: dict, auto: bool) -> None:
        self.decisions.append((action["args"].get("command", ""), decision["type"], auto, reject_reason(decision)))

    def end(self, *, output, usage, error=None) -> None:
        self.ended = {"output": output, "usage": usage, "error": error}


class FakeTracer:
    def __init__(self) -> None:
        self.turns: list[FakeTurn] = []
        self.shut = False

    @contextmanager
    def turn(self, text: str, *, session_id: str, tags: list[str]):
        self.turns.append(FakeTurn(text, session_id, tags))
        yield self.turns[-1]

    def shutdown(self) -> None:
        self.shut = True


def execute(command: str, *, inp: int = 100) -> AIMessage:
    return ai(inp=inp, out=5, cost=0.001, tool_calls=[
        {"name": "execute", "args": {"command": command}, "id": f"c-{abs(hash(command))}", "type": "tool_call"}])


def test_turn_records_input_output_decisions_and_usage(tmp_path: Path) -> None:
    tracer = FakeTracer()
    steps = [execute("git status"), execute("uv run pytest"), ai("끝", inp=300, out=7, cost=0.002)]
    app, _, out, _ = make_app(tmp_path, steps, "테스트 돌려", "n", "위험해", "/exit", tracer=tracer)
    app.run()
    [turn] = tracer.turns
    assert (turn.text, turn.session_id) == ("테스트 돌려", app.thread_id)
    assert turn.tags == [app.workspace.name, "mode:manual", f"model:{app.settings.model}"]
    # git status 는 자동 승인, pytest 는 사람이 거부했다
    assert turn.decisions == [("git status", "approve", True, ""), ("uv run pytest", "reject", False, "위험해")]
    assert turn.ended["output"] == "끝"
    assert turn.ended["error"] is None
    assert (turn.ended["usage"].calls, round(turn.ended["usage"].cost, 6)) == (3, 0.004)
    assert turn.handler.root_starts == 3  # 처음 + 재개 두 번이 같은 핸들러를 썼다
    assert tracer.shut
    assert "Langfuse 트레이스 켜짐" in out.getvalue()


def test_interrupted_turn_is_marked(tmp_path: Path) -> None:
    tracer = FakeTracer()
    app, _, _, _ = make_app(tmp_path, [execute("uv run pytest")], "돌려", KeyboardInterrupt(), "/exit",
                            tracer=tracer)
    app.run()
    assert tracer.turns[0].ended["error"] == "사용자가 중단"
    assert tracer.turns[0].ended["usage"].calls == 1


def test_failed_turn_is_marked(tmp_path: Path) -> None:
    tracer = FakeTracer()
    app, _, out, _ = make_app(tmp_path, [], "아무거나", "/exit", tracer=tracer)  # 대본이 비어 첫 호출에서 실패
    app.run()
    assert tracer.turns[0].ended["error"].startswith("ScriptExhausted")
    assert "오류: ScriptExhausted" in out.getvalue()


def test_banner_without_tracer(tmp_path: Path) -> None:
    app, _, out, _ = make_app(tmp_path, [], "/exit")
    app.run()
    assert "Langfuse" not in out.getvalue()


def test_make_tracer_needs_both_keys_and_package(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_tracer(Settings()) is None
    assert make_tracer(Settings(langfuse_public_key="pk-lf-x")) is None
    monkeypatch.setattr(tracing, "langfuse_installed", lambda: False)
    assert make_tracer(Settings(langfuse_public_key="pk-lf-x", langfuse_secret_key="sk-lf-x")) is None


def test_real_sdk_builds_one_trace_per_turn(tmp_path: Path) -> None:
    """실제 Langfuse 클라이언트와 CallbackHandler 를 메모리 exporter 로 돌린다(네트워크 없음)."""
    pytest.importorskip("langfuse")
    from langfuse import Langfuse
    from langfuse import LangfuseOtelSpanAttributes as A
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = Langfuse(public_key="pk-lf-test-memory", secret_key="sk-lf-test", base_url="http://127.0.0.1:9",
                      span_exporter=exporter)
    tracer = LangfuseTracer(client, public_key="pk-lf-test-memory")
    steps = [execute("ls"), tool_call("read_file", {"file_path": "/a.txt"}), ai("파일 하나", inp=200, out=3)]
    app, _, _, _ = make_app(tmp_path, steps, "파일 세어 줘", "/exit", tracer=tracer)
    (app.workspace / "a.txt").write_text("a", encoding="utf-8")
    app.run()  # 끝에서 shutdown 이 flush 한다

    spans = exporter.get_finished_spans()
    [root] = [s for s in spans if s.name == "turn"]
    attrs = dict(root.attributes)
    assert "파일 세어 줘" in attrs[A.OBSERVATION_INPUT]
    assert "파일 하나" in attrs[A.OBSERVATION_OUTPUT]
    assert attrs[A.TRACE_SESSION_ID] == app.thread_id
    assert any("usage" in key for key in attrs if key.startswith(A.OBSERVATION_METADATA))
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}  # 재개까지 한 트레이스
    [approval] = [s for s in spans if s.name == "approval:execute"]
    assert '"auto"' in approval.attributes[A.OBSERVATION_OUTPUT]
    generations = [s for s in spans if s.attributes.get(A.OBSERVATION_TYPE) == "generation"]
    assert len(generations) == 3  # 모델 호출 세 번
    assert any(s.name == "read_file" for s in spans)  # 도구 실행도 남는다


def test_unreachable_server_does_not_break_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """키가 틀리거나 서버에 닿지 않아도 턴은 끝나고, 종료 시 flush 가 오래 붙잡지 않는다."""
    pytest.importorskip("langfuse")
    monkeypatch.setattr(tracing, "TIMEOUT_S", 1)
    settings = Settings(langfuse_public_key="pk-lf-test-offline", langfuse_secret_key="sk-lf-test",
                        langfuse_base_url="http://127.0.0.1:9")
    tracer = make_tracer(settings)
    assert isinstance(tracer, LangfuseTracer)
    app, model, out, _ = make_app(tmp_path, [ai("안녕", inp=10, out=1)], "인사", "/exit", tracer=tracer)
    started = time.perf_counter()
    assert app.run() == 0
    assert time.perf_counter() - started < 20
    assert "안녕" in out.getvalue()
    assert len(model.calls) == 1
