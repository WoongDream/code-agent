"""스트림 조각을 터미널에 그린다. AI 텍스트는 토큰 단위로, 도구 호출은 이름과 핵심 인자만, 도구 결과는 앞 몇 줄만 보인다.

messages 모드에서는 model 노드의 AI 메시지만, updates 모드에서는 도구 호출·결과·todo 와 예산 미들웨어의 종료 메시지,
잘림 자동 복구 안내만 쓴다(중복 출력 방지). HITL 이 승인 뒤 다시 내보내는 응답과 PatchToolCalls 가 다시 내보내는
기록 전체는 그리지 않는다.

생각(reasoning)은 흐리게 줄 단위로 찍는다. 기본은 줄(문단)마다 앞 MEMO_CHARS 자만 보이는 메모이고 /thinking 으로 원문을 켠다.
OpenRouter 는 생각을 청크의 additional_kwargs 에 담는다. reasoning_content 는 content_blocks 의 reasoning 블록이 되지만
reasoning_details 에만 담긴 청크는 블록이 없어 그 text·summary 를 쓴다.
모델과 도구를 기다리는 동안에는 상태 줄(StatusLine)에 단계·진행량·경과 시간을 보인다. 생각이 오래 걸리거나 큰 도구 인자를
쓰는 동안에도 멈춘 것처럼 보이지 않게 하려는 것이다(az-quiz 첫 호출 약 15분, quizzes.js 인자 작성 99초).
"""

from __future__ import annotations

import difflib
import json
import re
import time
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCallChunk
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

from code_agent.stops import budget_stop, recovery_notice

RESULT_LINES = 6
MEMO_CHARS = 100  # 메모 모드에서 생각 한 줄(문단)에 보이는 글자 수
STATUS_CALL_CHARS = 60  # 상태 줄에 보이는 도구 호출 설명 길이
_TODO_MARK = {"completed": "[green]✔[/]", "in_progress": "[yellow]▶[/]", "pending": "[dim]○[/]"}
_FILE_PATH = re.compile(r'"file_path"\s*:\s*"([^"]*)"')  # 작성 중인 도구 인자(JSON 조각)에서 경로만 꺼낸다


class StatusLine:
    """기다리는 동안 보이는 한 줄: 스피너, 단계(생각 중·작성 중·실행 중), 진행량, 경과 시간.

    rich Status(Spinner + transient Live)로 그리고, 새로 그릴 때마다 __rich__ 로 경과 시간을 다시 센다.
    Live 가 켜져 있어도 console.print 한 줄은 그 위에 찍히지만, 줄바꿈 없는 토큰 출력과 입력 프롬프트는 상태 줄과
    섞이므로 그 전에 hide() 한다. enabled 가 아니면(터미널이 아니면) 상태만 기록하고 그리지 않는다.
    """

    def __init__(self, console: Console, *, enabled: bool) -> None:
        self._status = Status(self, console=console) if enabled else None
        self.label = ""
        self.detail = ""
        self.visible = False
        self._started = time.monotonic()

    def phase(self, label: str, detail: str = "") -> None:
        """새 단계. 경과 시간을 0부터 센다."""
        self._started = time.monotonic()
        self.update(label, detail)

    def update(self, label: str, detail: str = "") -> None:
        self.label, self.detail = label, detail
        if not self.visible:
            self.visible = True
            if self._status is not None:
                self._status.start()

    def hide(self) -> None:
        if self.visible:
            self.visible = False
            if self._status is not None:
                self._status.stop()

    def text(self) -> str:
        seconds = int(time.monotonic() - self._started)
        elapsed = f"{seconds // 60}분 {seconds % 60}초" if seconds >= 60 else f"{seconds}초"
        return f"{self.label}{' ' + self.detail if self.detail else ''}… {elapsed}"

    def __rich__(self) -> Text:
        return Text(self.text(), style="dim")


class Renderer:
    def __init__(self, console: Console, *, status: bool | None = None, full_reasoning: bool = False) -> None:
        """status 는 상태 줄을 그릴지다(기본: 콘솔이 터미널일 때). full_reasoning 이면 생각 원문을 다 보인다."""
        self.console = console
        self.status = StatusLine(console, enabled=console.is_terminal if status is None else status)
        self.full_reasoning = full_reasoning
        self._streamed: set[str] = set()  # 토큰으로 이미 그린 AI 메시지 id
        self._mid_line = False  # 줄바꿈 없는 토큰 출력 뒤다. 이때 상태 줄은 꺼져 있다
        self.last_text = ""  # 메인 에이전트의 마지막 AI 텍스트. 턴의 최종 답변으로 트레이스에 남긴다
        self._reasoning = ""  # 아직 찍지 않은 생각의 현재 줄
        self._memo_shown = False  # 현재 줄의 메모를 이미 찍었다(줄이 끝날 때까지 나머지는 버린다)
        self._reasoning_chars = 0  # 이번 모델 호출의 생각 글자 수
        self._writing: dict[int, list[str]] = {}  # 이번 모델 호출에서 작성 중인 도구 호출: index → [이름, 인자 조각]
        self._pending: dict[str, str] = {}  # 결과를 기다리는 도구 호출 id → 설명

    def start(self) -> None:
        """턴 시작. 첫 모델 응답을 기다린다."""
        self._phase("생각 중")

    def pause(self) -> None:
        """승인 프롬프트처럼 입력을 받기 전에 상태 줄을 멈추고 줄을 끝낸다."""
        self._flush_reasoning()
        self.status.hide()
        self._newline()

    def resume(self) -> None:
        """승인 결정 뒤. 기다리던 도구가 있으면 그 실행을, 없으면 모델 응답을 다시 기다린다."""
        self._wait()

    def end_turn(self) -> None:
        self.pause()

    def on_part(self, part: tuple[tuple[str, ...], str, Any]) -> None:
        ns, mode, data = part
        prefix = "[magenta][sub][/] " if ns else ""
        if mode == "messages":
            message, meta = data
            if meta.get("langgraph_node") == "model":
                self._ai_text(message)
        elif mode == "updates":
            for node, update in data.items():
                if isinstance(update, dict):
                    self._update(node, update, prefix)

    def _ai_text(self, message: BaseMessage) -> None:
        """스트리밍 모델은 청크로, 스트리밍하지 않는 모델(대본 모델 등)은 완성된 메시지로 온다."""
        if isinstance(message, AIMessageChunk):
            if message.id:
                self._streamed.add(message.id)
            blocks = message.content_blocks
            reasoning = "".join(str(b.get("reasoning") or "") for b in blocks if b["type"] == "reasoning")
            if reasoning := reasoning or reasoning_details_text(message):
                self._think(reasoning)
            for block in blocks:
                if block["type"] == "text" and block.get("text"):
                    self._flush_reasoning()
                    self._write(block["text"])
            for chunk in message.tool_call_chunks:
                self._flush_reasoning()
                self._tool_writing(chunk)
        elif isinstance(message, AIMessage) and message.id not in self._streamed and message.text:
            self._write(message.text)

    def _update(self, node: str, update: dict, prefix: str) -> None:
        messages = update.get("messages")
        if not isinstance(messages, list) or rewrites_history(messages):
            messages = []  # 기록 전체를 다시 내보낸 것이라 새로 그릴 것이 없다
        for message in messages:
            if node == "model" and isinstance(message, AIMessage):
                self._flush_reasoning()
                self._reasoning_chars = 0
                self._writing.clear()
                if not prefix and message.text:
                    self.last_text = message.text
                for call in message.tool_calls:
                    self._line(f"{prefix}[cyan]●[/] {escape(describe_call(call))}")
                    self._pending[call["id"]] = describe_call(call)
                if message.tool_calls:
                    self._wait()
            elif recovery_notice(node, message):
                self._line(f"{prefix}[yellow]응답이 출력 한도에 걸려 잘렸다. 안내를 덧붙여 한 번 다시 요청한다.[/]")
                self._wait()
            elif budget_stop(node, message):
                # 예산 미들웨어의 종료 메시지는 model 이 아닌 노드에서 나온다
                self._line(f"{prefix}[yellow]{escape(message.text)}[/]")
                if not prefix:
                    self.last_text = message.text
            elif isinstance(message, ToolMessage):
                self._tool_result(message, prefix)
                if self._pending.pop(message.tool_call_id, None) is not None:
                    self._wait(new_phase=not self._pending)
        if isinstance(update.get("todos"), list):
            self._line(todos_markup(update["todos"]))

    def _think(self, text: str) -> None:
        """생각을 줄 단위로 찍는다. 메모 모드는 줄마다 앞 MEMO_CHARS 자만, 그만큼 모이면 줄이 끝나기 전에 찍는다."""
        self._reasoning_chars += len(text)
        self._show("생각 중", f"{self._reasoning_chars:,}자")
        *lines, self._reasoning = (self._reasoning + text).split("\n")
        for line in lines:
            self._reasoning_line(line)
        if not self.full_reasoning and not self._memo_shown and len(self._reasoning) >= MEMO_CHARS:
            self._reasoning_line(self._reasoning, done=False)

    def _reasoning_line(self, line: str, *, done: bool = True) -> None:
        if line.strip() and not self._memo_shown:
            shown = line if self.full_reasoning or len(line) <= MEMO_CHARS else line[:MEMO_CHARS] + "…"
            self._newline()
            self.console.print(Text(shown.strip(), style="dim italic"), soft_wrap=True)
        self._memo_shown = not done

    def _flush_reasoning(self) -> None:
        if self._reasoning:
            self._reasoning_line(self._reasoning)
        self._reasoning, self._memo_shown = "", False

    def _tool_writing(self, chunk: ToolCallChunk) -> None:
        """도구 인자가 스트리밍되는 동안(큰 write_file 등) 이름·경로·크기를 상태 줄에 보인다."""
        index = chunk.get("index") or 0
        new = index not in self._writing
        name_args = self._writing.setdefault(index, ["", ""])
        name_args[0] = chunk.get("name") or name_args[0]
        name_args[1] += chunk.get("args") or ""
        name, args = name_args
        path = _FILE_PATH.search(args)
        label = " ".join([name or "도구", *([path[1]] if path else []), "작성 중"])
        size = len(args.encode("utf-8"))
        (self._phase if new else self._show)(label, f"{size / 1024:.1f}KB" if size >= 1024 else f"{size}B")

    def _wait(self, *, new_phase: bool = True) -> None:
        """도구 결과를 기다리면 '실행 중', 다 받았으면 다음 모델 응답을 기다리는 '생각 중'."""
        if not self._pending:
            self._phase("생각 중")
            return
        calls = list(self._pending.values())
        call = " ".join(calls[0].split())
        call = call if len(call) <= STATUS_CALL_CHARS else call[: STATUS_CALL_CHARS - 1] + "…"
        label = f"{call} 실행 중" if len(calls) == 1 else f"도구 {len(calls)}개 실행 중"
        (self._phase if new_phase else self._show)(label)

    def _phase(self, label: str, detail: str = "") -> None:
        self._newline()
        self.status.phase(label, detail)

    def _show(self, label: str, detail: str = "") -> None:
        self._newline()
        self.status.update(label, detail)

    def _tool_result(self, message: ToolMessage, prefix: str) -> None:
        if message.name == "write_todos":
            return  # todo 는 updates 의 todos 로 따로 그린다
        lines = message.text.strip().splitlines() or ["(출력 없음)"]
        shown = "\n".join(f"  {line}" for line in lines[:RESULT_LINES])
        more = f"\n  … (+{len(lines) - RESULT_LINES}줄)" if len(lines) > RESULT_LINES else ""
        style = "red" if message.status == "error" else "dim"
        self._newline()
        self.console.print(Text(shown + more, style=style), soft_wrap=True)

    def _write(self, text: str, style: str | None = None) -> None:
        self.status.hide()  # 줄바꿈 없는 출력은 상태 줄과 섞인다
        self.console.print(Text(text, style=style or ""), end="", soft_wrap=True)
        self._mid_line = not text.endswith("\n")

    def _line(self, markup: str) -> None:
        self._newline()
        self.console.print(markup, soft_wrap=True)

    def _newline(self) -> None:
        if self._mid_line:
            self.console.print()
            self._mid_line = False


def reasoning_details_text(message: BaseMessage) -> str:
    """OpenRouter 청크의 reasoning_details 에 담긴 생각 글. 암호화된 항목(reasoning.encrypted)에는 글이 없다."""
    details = message.additional_kwargs.get("reasoning_details")
    items = details if isinstance(details, list) else []
    texts = [item.get("text") or item.get("summary") for item in items if isinstance(item, dict)]
    return "".join(t for t in texts if isinstance(t, str))


def rewrites_history(messages: list) -> bool:
    """[RemoveMessage(REMOVE_ALL), *기록] 형태인지. PatchToolCallsMiddleware 는 승인 중 끊긴 턴 다음 턴 시작에 이렇게
    기록 전체를 다시 내보낸다. 그대로 그리면 옛 도구 결과와 답변이 다시 찍힌다."""
    return bool(messages) and isinstance(messages[0], RemoveMessage) and messages[0].id == REMOVE_ALL_MESSAGES


def describe_call(call: dict) -> str:
    name, args = call["name"], call.get("args") or {}
    if name == "execute":
        return f"$ {args.get('command', '')}"
    if name in ("read_file", "write_file", "edit_file", "delete", "ls"):
        return f"{name} {args.get('file_path') or args.get('path') or '/'}"
    if name in ("glob", "grep"):
        return f"{name} {args.get('pattern', '')!r} {args.get('path') or ''}".rstrip()
    if name == "write_todos":
        return "write_todos"
    summary = json.dumps(args, ensure_ascii=False)
    return f"{name} {summary[:100]}{'…' if len(summary) > 100 else ''}"


def todos_markup(todos: list[dict]) -> str:
    rows = [f"  {_TODO_MARK.get(t.get('status', ''), '○')} {escape(str(t.get('content', '')))}" for t in todos]
    return "[bold]할 일[/]\n" + "\n".join(rows) if rows else "[dim]할 일 없음[/]"


def describe_action(action: dict) -> RenderableType:
    """승인 프롬프트에 보일 내용. execute 는 명령, edit_file 은 diff, write_file 은 경로·크기·앞부분."""
    name, args = action["name"], action.get("args") or {}
    title = Text(f"승인 요청: {name}", style="bold yellow")
    if name == "execute":
        return Group(title, Syntax(str(args.get("command", "")), "bash", word_wrap=True))
    if name == "edit_file":
        diff = difflib.unified_diff(str(args.get("old_string", "")).splitlines(),
                                    str(args.get("new_string", "")).splitlines(),
                                    fromfile=args.get("file_path", ""), tofile=args.get("file_path", ""), lineterm="")
        scope = " (모든 일치 항목)" if args.get("replace_all") else ""
        return Group(title, Text(f"{args.get('file_path', '')}{scope}"), Syntax("\n".join(diff), "diff"))
    if name == "write_file":
        content = str(args.get("content", ""))
        head = "\n".join(content.splitlines()[:20])
        return Group(title, Text(f"{args.get('file_path', '')} ({len(content.encode('utf-8'))} bytes)"),
                     Text(head, style="dim"))
    if name == "delete":
        return Group(title, Text(str(args.get("file_path", "")), style="red"))
    return Group(title, Text(json.dumps(args, ensure_ascii=False, indent=2)))
