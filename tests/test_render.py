"""생각 표시와 상태 줄: OpenRouter reasoning(메모·원문), 단계별 상태 줄, 터미널 판정, 승인 프롬프트 전 멈춤. 모델은 대본이다."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import cast

import pytest
from fakes import StreamingChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from test_cli import make_app

from code_agent import cli, render
from code_agent.agent import build_agent, run_turn
from code_agent.approval import APPROVE
from code_agent.render import MEMO_CHARS, Renderer, StatusLine
from code_agent.settings import Settings

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LONG = "I'm checking edge cases around game-over timing and whether the next piece spawns correctly. " * 3
NOTE = "Let me check the working folder first."


def openrouter(**delta: object) -> AIMessageChunk:
    """ChatOpenRouter 가 SSE delta 하나를 바꾼 청크. langchain-openrouter 의 실제 변환 함수를 쓴다."""
    pytest.importorskip("langchain_openrouter")
    from langchain_openrouter.chat_models import _convert_chunk_to_message_chunk

    raw = {"choices": [{"delta": {"role": "assistant", "content": "", **delta}}]}
    return cast(AIMessageChunk, _convert_chunk_to_message_chunk(raw, AIMessageChunk))


def thought(text: str) -> AIMessageChunk:
    """delta.reasoning 과 reasoning_details 에 같은 글이 함께 오는 청크(az-quiz 기록에서 두 길이가 같았다)."""
    return openrouter(reasoning=text, reasoning_details=[{"type": "reasoning.text", "text": text, "index": 0}])


def writing(index: int, arguments: str, name: str | None = None) -> AIMessageChunk:
    function = {"arguments": arguments, **({"name": name} if name else {})}
    call = {"index": index, "function": function, **({"id": f"c{index}", "type": "function"} if name else {})}
    return openrouter(tool_calls=[call])


def chunk(message: AIMessageChunk) -> tuple:
    return ((), "messages", (message, {"langgraph_node": "model"}))


def updates(node: str, *messages) -> tuple:
    return ((), "updates", {node: {"messages": list(messages)}})


def renderer(*, full: bool = False) -> tuple[Renderer, io.StringIO]:
    out = io.StringIO()
    console = Console(file=out, width=300, force_terminal=True, color_system="truecolor")
    return Renderer(console, status=False, full_reasoning=full), out


def plain(out: io.StringIO) -> list[str]:
    return [line for line in ANSI.sub("", out.getvalue()).splitlines() if line]


def test_openrouter_reasoning_is_shown_dim_as_short_memo() -> None:
    r, out = renderer()
    r.start()
    for piece in (f"{NOTE}\n\n", LONG[:120], LONG[120:], "\n\n"):
        r.on_part(chunk(thought(piece)))
    assert (r.status.label, r.status.detail) == ("생각 중", f"{len(NOTE) + 2 + len(LONG) + 2:,}자")
    r.on_part(chunk(openrouter(content="안녕")))
    r.end_turn()
    assert plain(out) == [NOTE, LONG[:MEMO_CHARS] + "…", "안녕"]  # 문단마다 앞부분만, 답변은 따로 줄
    assert f"\x1b[2;3m{NOTE}" in out.getvalue()  # Anthropic 경로처럼 흐린 기울임


def test_full_reasoning_shows_whole_paragraph() -> None:
    r, out = renderer(full=True)
    r.on_part(chunk(thought(f"{LONG}\n\n{NOTE}")))
    r.end_turn()
    assert plain(out) == [LONG.strip(), NOTE]


def test_reasoning_details_only_chunks_are_shown_without_double_count() -> None:
    """reasoning_details 에만 담긴 청크는 content_blocks 가 비어 전에는 안 보였다. 서명만 있는 항목은 글이 없다."""
    r, out = renderer()
    only_details = openrouter(reasoning_details=[{"type": "reasoning.text", "text": " 두 번째 메모.", "index": 0}])
    assert only_details.content_blocks == []
    for message in (thought("첫 메모."), only_details,
                    openrouter(reasoning_details=[{"type": "reasoning.text", "signature": "sig", "index": 0}])):
        r.on_part(chunk(message))
    assert r.status.detail == f"{len('첫 메모. 두 번째 메모.')}자"
    r.end_turn()
    assert plain(out) == ["첫 메모. 두 번째 메모."]


def test_status_follows_thinking_writing_running_phases() -> None:
    r, out = renderer()
    r.start()
    assert (r.status.visible, r.status.label) == (True, "생각 중")
    content = "x" * 12_600
    for message in (writing(0, "", name="write_file"), writing(0, '{"file_path": "/quizzes.js", "content": "'),
                    writing(0, content)):
        r.on_part(chunk(message))
    assert (r.status.label, r.status.detail) == ("write_file /quizzes.js 작성 중", "12.3KB")
    calls = [{"name": "write_file", "args": {"file_path": "/quizzes.js", "content": content}, "id": "w",
              "type": "tool_call"},
             {"name": "execute", "args": {"command": "uv run pytest -q"}, "id": "x", "type": "tool_call"}]
    r.on_part(updates("model", AIMessage(content="", tool_calls=calls)))
    assert r.status.label == "도구 2개 실행 중"
    r.on_part(updates("tools", ToolMessage(content="Updated file /quizzes.js", tool_call_id="w")))
    assert r.status.label == "$ uv run pytest -q 실행 중"
    r.on_part(updates("tools", ToolMessage(content="3 passed", tool_call_id="x")))
    assert r.status.label == "생각 중"
    r.on_part(chunk(openrouter(content="끝")))
    assert not r.status.visible  # 첫 텍스트가 나오면 사라진다
    r.on_part(updates("model", AIMessage(content="끝")))
    r.end_turn()
    assert not r.status.visible
    assert plain(out)[-1] == "끝"


def test_status_text_shows_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(render.time, "monotonic", lambda: now[0])
    status = StatusLine(Console(file=io.StringIO()), enabled=False)
    status.phase("생각 중", "1,234자")
    now[0] = 142.9
    assert status.text() == "생각 중 1,234자… 42초"
    now[0] = 285.0
    status.update("생각 중")
    assert status.text() == "생각 중… 3분 5초"  # 진행량만 바뀌면 시간은 이어서 센다


def test_status_is_drawn_only_on_terminal() -> None:
    out = io.StringIO()
    r = Renderer(Console(file=out, width=80, force_terminal=True))  # 기본값: 터미널이면 그린다
    r.start()
    r.pause()
    assert "생각 중" in ANSI.sub("", out.getvalue())
    out = io.StringIO()
    r = Renderer(Console(file=out, width=80))  # 파이프·StringIO 는 터미널이 아니다
    r.start()
    r.pause()
    assert out.getvalue() == ""


def test_app_draws_status_only_when_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _, _, _ = make_app(tmp_path, [])
    assert not app.status_line  # StringIO 콘솔
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    (tmp_path / "b").mkdir()
    app, _, _, _ = make_app(tmp_path / "b", [])
    app.console = Console(file=io.StringIO(), force_terminal=True)
    assert cli.App(app.settings, app.workspace, "manual", app.sessions, app.thread_id, app.console, input,
                   agent=app.agent).status_line


def test_streamed_tool_arguments_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenRouter 모양 청크로 스트리밍한 write_file 이 합쳐져 실제로 실행되고, 작성 중 상태가 보인다."""
    labels: list[tuple[str, str]] = []
    real_update = StatusLine.update

    def record(self: StatusLine, label: str, detail: str = "") -> None:
        labels.append((label, detail))
        real_update(self, label, detail)

    monkeypatch.setattr(StatusLine, "update", record)
    body = "q" * 3000
    model = StreamingChatModel(steps=[
        [thought("I'll write quizzes.js directly.\n\n"), writing(0, "", name="write_file"),
         writing(0, '{"file_path": "/quizzes.js", "content": "'), writing(0, body + '"}')],
        [openrouter(content="다 썼다")],
    ])
    (tmp_path / "ws").mkdir()
    agent = build_agent(Settings(data_dir=tmp_path / "data"), tmp_path / "ws", InMemorySaver(), model=model)
    r, out = renderer()
    r.start()
    run_turn(agent, {"messages": [{"role": "user", "content": "만들어"}]}, {"configurable": {"thread_id": "t"}},
             decide=lambda requests: [APPROVE for _ in requests], on_part=r.on_part)
    r.end_turn()
    assert (tmp_path / "ws" / "quizzes.js").read_text(encoding="utf-8") == body
    assert ("write_file /quizzes.js 작성 중", "3.0KB") in labels  # 인자 JSON 3,043바이트
    assert ("write_file /quizzes.js 실행 중", "") in labels
    assert plain(out) == ["I'll write quizzes.js directly.", "● write_file /quizzes.js", "  Updated file /quizzes.js",
                          "다 썼다"]


class Recorder:
    """cli.Renderer 대신 들어가 턴마다 만든 Renderer 를 남긴다."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.made: list[Renderer] = []
        real = cli.Renderer

        def make(*args, **kwargs) -> Renderer:
            self.made.append(real(*args, **kwargs))
            return self.made[-1]

        monkeypatch.setattr(cli, "Renderer", make)


def test_status_pauses_before_approval_prompt_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(monkeypatch)
    model = StreamingChatModel(steps=[
        [thought("Run the tests.\n\n"), writing(0, "", name="execute"), writing(0, '{"command": "echo hi"}')],
        [openrouter(content="통과")],
    ])
    app, _, out, script = make_app(tmp_path, [], "돌려", "y", "/exit", model=model)
    seen: list[tuple[str, bool, str]] = []

    def read_line(prompt: str) -> str:
        if recorder.made:
            status = recorder.made[-1].status
            seen.append((prompt, status.visible, status.label))
        return script(prompt)

    app.read_line = read_line
    app.run()
    assert ("[y] 승인 / [n] 거부 › ", False, "$ echo hi 실행 중") in seen  # 프롬프트 전에 멈췄다
    text = out.getvalue()
    assert text.index("Run the tests.") < text.index("승인 요청: execute") < text.index("통과")
    assert not recorder.made[-1].status.visible


def test_ctrl_c_stops_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(monkeypatch)

    def interrupt(messages: list) -> list[AIMessageChunk]:
        raise KeyboardInterrupt

    model = StreamingChatModel(steps=[[thought("Thinking a lot.")], interrupt])
    app, _, out, _ = make_app(tmp_path, [], "하나", "둘", "/exit", model=model)
    app.run()
    assert "중단했다" in out.getvalue()
    assert [r.status.visible for r in recorder.made] == [False, False]


def test_thinking_command_switches_full_reasoning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(monkeypatch)
    model = StreamingChatModel(steps=[[thought(LONG), openrouter(content="a")], [thought(LONG), openrouter(content="b")]])
    app, _, out, _ = make_app(tmp_path, [], "/help", "/thinking x", "/thinking", "원문", "/thinking off", "메모",
                              "/exit", model=model)
    app.run()
    text = out.getvalue()
    assert "/thinking [on|off]  생각(reasoning) 원문 표시" in text  # rich 마크업으로 사라지지 않는다
    assert text.count("/thinking [on|off]") == 2  # 도움말과 잘못된 인자 안내
    assert "생각 원문 표시 → 켬" in text and "생각 원문 표시 → 끔(문단마다 앞부분만 메모로)" in text
    assert [r.full_reasoning for r in recorder.made] == [True, False]
    assert text.count(LONG.strip()) == 1 and LONG[:MEMO_CHARS] + "…" in text


def test_cut_tool_call_status_and_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#6 상태 줄과 #7 잘린 호출 막기가 만나는 곳: 작성 중으로 보이던 잘린 write_file 은 '실행 중'이 되지 않고,
    막았다는 줄과 복구 안내가 나온 뒤 다시 '생각 중'으로 간다."""
    labels: list[str] = []
    real_update = StatusLine.update

    def record(self: StatusLine, label: str, detail: str = "") -> None:
        labels.append(label)
        real_update(self, label, detail)

    monkeypatch.setattr(StatusLine, "update", record)
    stop = AIMessageChunk(content="", response_metadata={"finish_reason": "length", "model_provider": "openrouter"})
    model = StreamingChatModel(steps=[
        [thought("Writing it all.\n\n"), writing(0, "", name="write_file"),
         writing(0, '{"file_path": "/q.js", "content": "const q = ['), stop],
        [openrouter(content="나눠서 쓰겠다")],
    ])
    (tmp_path / "ws").mkdir()
    agent = build_agent(Settings(data_dir=tmp_path / "data"), tmp_path / "ws", InMemorySaver(), model=model)
    r, out = renderer()
    r.start()
    run_turn(agent, {"messages": [{"role": "user", "content": "만들어"}]}, {"configurable": {"thread_id": "t"}},
             decide=lambda requests: [APPROVE for _ in requests], on_part=r.on_part)
    r.end_turn()
    assert not (tmp_path / "ws" / "q.js").exists()
    assert "write_file /q.js 작성 중" in labels and not any("실행 중" in label for label in labels)
    assert labels[-1] == "생각 중"  # 복구 안내 뒤 다시 모델을 기다렸다
    assert plain(out) == ["Writing it all.", "write_file 호출은 출력이 멈춰 인자가 잘렸을 수 있어 실행하지 않았다.",
                          "응답이 출력 한도에 걸려 잘렸다. 안내를 덧붙여 한 번 다시 요청한다.", "나눠서 쓰겠다"]
    assert not r.status.visible
