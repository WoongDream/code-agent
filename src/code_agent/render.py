"""스트림 조각을 터미널에 그린다. AI 텍스트는 토큰 단위로, 도구 호출은 이름과 핵심 인자만, 도구 결과는 앞 몇 줄만 보인다.

messages 모드에서는 model 노드의 AI 메시지만, updates 모드에서는 도구 호출·결과·todo 와 model 이 아닌 노드가 남긴
AI 메시지(예산 상한 도달 안내), 잘림 자동 복구 안내만 쓴다(중복 출력 방지).
"""

from __future__ import annotations

import difflib
import json
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text

from code_agent.stops import is_recovery

RESULT_LINES = 6
_TODO_MARK = {"completed": "[green]✔[/]", "in_progress": "[yellow]▶[/]", "pending": "[dim]○[/]"}


class Renderer:
    def __init__(self, console: Console) -> None:
        self.console = console
        self._streamed: set[str] = set()  # 토큰으로 이미 그린 AI 메시지 id
        self._mid_line = False
        self.last_text = ""  # 메인 에이전트의 마지막 AI 텍스트. 턴의 최종 답변으로 트레이스에 남긴다

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

    def end_turn(self) -> None:
        self._newline()

    def _ai_text(self, message: BaseMessage) -> None:
        """스트리밍 모델은 청크로, 스트리밍하지 않는 모델(대본 모델 등)은 완성된 메시지로 온다."""
        if isinstance(message, AIMessageChunk):
            if message.id:
                self._streamed.add(message.id)
            for block in message.content_blocks:
                if block["type"] == "text" and block.get("text"):
                    self._write(block["text"])
                elif block["type"] == "reasoning" and block.get("reasoning"):
                    self._write(block["reasoning"], style="dim italic")
        elif isinstance(message, AIMessage) and message.id not in self._streamed and message.text:
            self._write(message.text)

    def _update(self, node: str, update: dict, prefix: str) -> None:
        messages = update.get("messages")
        for message in messages if isinstance(messages, list) else []:
            if node == "model" and isinstance(message, AIMessage):
                if not prefix and message.text:
                    self.last_text = message.text
                for call in message.tool_calls:
                    self._line(f"{prefix}[cyan]●[/] {escape(describe_call(call))}")
            elif is_recovery(message):
                self._line(f"{prefix}[yellow]응답이 출력 한도에 걸려 잘렸다. 결과물을 바로 쓰라고 안내하고 한 번 다시 요청한다.[/]")
            elif isinstance(message, AIMessage) and message.text:
                # 예산 미들웨어(exit_behavior="end")의 종료 메시지처럼 model 이 아닌 노드가 남긴 답변
                self._line(f"{prefix}[yellow]{escape(message.text)}[/]")
                if not prefix:
                    self.last_text = message.text
            elif isinstance(message, ToolMessage):
                self._tool_result(message, prefix)
        if isinstance(update.get("todos"), list):
            self._line(todos_markup(update["todos"]))

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
        self.console.print(Text(text, style=style or ""), end="", soft_wrap=True)
        self._mid_line = not text.endswith("\n")

    def _line(self, markup: str) -> None:
        self._newline()
        self.console.print(markup, soft_wrap=True)

    def _newline(self) -> None:
        if self._mid_line:
            self.console.print()
            self._mid_line = False


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
