"""터미널 REPL. 사용: code-agent [--workspace PATH] [--model ID] [--mode manual|auto-edit|yolo] [--resume ID]"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax

from code_agent.agent import build_agent, run_turn
from code_agent.approval import APPROVE, MODES, decide_all, reject
from code_agent.render import Renderer, describe_action, todos_markup
from code_agent.settings import Settings, load_settings
from code_agent.stops import TurnStops
from code_agent.tracing import Tracer, langfuse_installed, make_tracer
from code_agent.usage import Usage, UsageCallback
from code_agent.worklog import session_context

HELP = """\
[bold]슬래시 커맨드[/]
  /help          이 도움말
  /new           새 세션(대화) 시작
  /mode <모드>   승인 모드 변경: manual(전부 묻기) | auto-edit(파일 쓰기 자동, 셸은 묻기) | yolo(전부 자동)
  /diff          작업 폴더의 git diff
  /todos         현재 할 일 목록
  /cost          이 세션의 모델 호출·토큰·비용 합계
  /exit          종료
Ctrl+C 는 실행 중인 턴만 취소한다. 읽기 전용 셸 명령(git status, ls, cat 등)은 모드와 관계없이 자동 승인된다."""
YOLO_WARNING = "[bold red]yolo 모드: 파일 쓰기와 셸 명령을 묻지 않고 실행한다. 셸은 샌드박스가 아니다.[/]"

ReadLine = Callable[[str], str]
_USAGE_COLUMNS = [(f.name, "REAL" if f.type == "float" else "INTEGER") for f in fields(Usage)]


class Sessions:
    """sessions.db 에 체크포인트와 함께 thread → 작업 폴더를 남긴다. --resume 은 같은 작업 폴더로 이어 간다."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute("CREATE TABLE IF NOT EXISTS code_agent_threads "
                     "(thread_id TEXT PRIMARY KEY, workspace TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute(f"CREATE TABLE IF NOT EXISTS code_agent_usage (thread_id TEXT NOT NULL, at TEXT NOT NULL, "
                     f"{', '.join(f'{name} {kind} NOT NULL' for name, kind in _USAGE_COLUMNS)})")
        conn.commit()

    def create(self, workspace: Path) -> str:
        thread_id = uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO code_agent_threads VALUES (?, ?, ?)",
                          (thread_id, str(workspace), datetime.now(UTC).isoformat()))
        self.conn.commit()
        return thread_id

    def workspace_of(self, thread_id: str) -> Path | None:
        row = self.conn.execute("SELECT workspace FROM code_agent_threads WHERE thread_id = ?",
                                (thread_id,)).fetchone()
        return Path(row[0]) if row else None

    def add_usage(self, thread_id: str, usage: Usage) -> None:
        values = [getattr(usage, name) for name, _ in _USAGE_COLUMNS]
        self.conn.execute(f"INSERT INTO code_agent_usage VALUES (?, ?, {', '.join('?' * len(values))})",
                          (thread_id, datetime.now(UTC).isoformat(), *values))
        self.conn.commit()

    def usage_of(self, thread_id: str) -> tuple[Usage, int]:
        """(세션 합계, 기록된 턴 수)."""
        rows = self.conn.execute(f"SELECT {', '.join(name for name, _ in _USAGE_COLUMNS)} FROM code_agent_usage "
                                 "WHERE thread_id = ?", (thread_id,)).fetchall()
        total = Usage()
        for row in rows:
            total += Usage(**dict(zip((name for name, _ in _USAGE_COLUMNS), row, strict=True)))
        return total, len(rows)


class App:
    def __init__(self, settings: Settings, workspace: Path, mode: str, sessions: Sessions, thread_id: str,
                 console: Console, read_line: ReadLine, *, agent=None, tracer: Tracer | None = None) -> None:
        self.settings, self.workspace, self.mode = settings, workspace, mode
        self.sessions, self.thread_id = sessions, thread_id
        self.console, self.read_line = console, read_line
        self.agent = agent if agent is not None else build_agent(settings, workspace, SqliteSaver(sessions.conn))
        self.tracer = tracer

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def run(self) -> int:
        self.console.print(f"[bold]code-agent[/]  작업 폴더 [cyan]{escape(str(self.workspace))}[/]\n"
                           f"모델 {escape(self.settings.model)} · 승인 모드 {self.mode} · 세션 {self.thread_id}  "
                           "[dim](/help)[/]")
        if self.tracer is not None:
            self.console.print(f"[dim]Langfuse 트레이스 켜짐 → {escape(self.settings.langfuse_base_url)}[/]")
        if self.mode == "yolo":
            self.console.print(YOLO_WARNING)
        try:
            return self._loop()
        finally:
            if self.tracer is not None:
                self.tracer.shutdown()  # 남은 트레이스를 보낸다

    def _loop(self) -> int:
        while True:
            try:
                line = self.read_line("› ").strip()
            except KeyboardInterrupt:
                self.console.print("[dim]종료하려면 /exit[/]")
                continue
            except EOFError:
                return 0
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    return 0
            else:
                self.turn(line)

    def command(self, line: str) -> bool:
        """슬래시 커맨드를 처리한다. False 면 REPL 을 끝낸다."""
        name, _, arg = line.partition(" ")
        arg = arg.strip()
        if name in ("/exit", "/quit"):
            self.console.print(f"[dim]이어서 하려면: code-agent --resume {self.thread_id}[/]")
            return False
        if name == "/help":
            self.console.print(HELP)
        elif name == "/new":
            self.thread_id = self.sessions.create(self.workspace)
            self.console.print(f"새 세션 {self.thread_id}")
        elif name == "/mode":
            if arg not in MODES:
                self.console.print(f"[red]모드는 {', '.join(MODES)} 중 하나다. 지금은 {self.mode}.[/]")
            else:
                self.mode = arg
                self.console.print(f"승인 모드 → {arg}")
                if arg == "yolo":
                    self.console.print(YOLO_WARNING)
        elif name == "/diff":
            self.show_diff()
        elif name == "/todos":
            todos = self.agent.get_state(self.config).values.get("todos") or []
            self.console.print(todos_markup(todos))
        elif name == "/cost":
            total, turns = self.sessions.usage_of(self.thread_id)
            self.console.print(f"세션 {self.thread_id} · 턴 {turns}개 · {escape(total.summary())}" if turns
                               else "[dim]이 세션에 기록된 사용량이 아직 없다.[/]")
        else:
            self.console.print(f"[red]알 수 없는 커맨드: {escape(name)}[/] (/help)")
        return True

    def turn(self, text: str) -> None:
        renderer = Renderer(self.console)
        stops = TurnStops()
        meter = UsageCallback()
        tags = [self.workspace.name, f"mode:{self.mode}", f"model:{self.settings.model}"]
        trace_context: Any = nullcontext()
        if self.tracer is not None:
            previous = self.agent.get_state(self.config).values.get("messages", [])
            trace_context = self.tracer.turn(text, session_id=self.thread_id, tags=tags, workspace=self.workspace,
                                             context=session_context(previous, self.workspace))
        with trace_context as trace:
            def decide(action_requests: list[dict]) -> list[dict]:
                renderer.end_turn()
                return decide_all(action_requests, self.mode, self.ask,
                                  on_decision=trace.decision if trace is not None else None)

            def on_part(part: tuple) -> None:
                renderer.on_part(part)
                stops.on_part(part)
                if trace is not None:
                    trace.on_part(part)  # 작업 기록(도구 실행과 결과 요약)

            callbacks = [meter, *(trace.callbacks if trace is not None else [])]
            error = None
            try:
                run_turn(self.agent, {"messages": [{"role": "user", "content": text}]},
                         {**self.config, "callbacks": callbacks}, decide=decide, on_part=on_part)
            except KeyboardInterrupt:
                renderer.end_turn()
                self.console.print("[yellow]중단했다.[/]")
                error = "사용자가 중단"
            except Exception as e:  # noqa: BLE001 - 한 턴의 실패로 REPL 을 끝내지 않는다
                renderer.end_turn()
                self.console.print(f"[red]오류: {type(e).__name__}: {escape(str(e))}[/]")
                error = f"{type(e).__name__}: {e}"
            else:
                renderer.end_turn()
            warnings = stops.warnings()  # 잘린 응답·예산 상한 도달. 말없이 멈춘 것처럼 보이지 않게 한다
            for warning in warnings:
                self.console.print(f"[yellow]{escape(warning)}[/]", soft_wrap=True)
            if trace is not None:
                trace.end(output=renderer.last_text, usage=meter.usage, error=error, warnings=warnings)
        if meter.usage.calls:  # 중단·오류로 끝난 턴도 쓴 만큼은 남긴다
            self.sessions.add_usage(self.thread_id, meter.usage)
            self.console.print(f"[dim]{escape(meter.usage.summary())}[/]")

    def ask(self, action: dict) -> dict:
        self.console.print(describe_action(action))
        while True:
            answer = self.read_line("[y] 승인 / [n] 거부 › ").strip().lower()
            if answer in ("y", "yes", "ㅛ"):
                return APPROVE
            if answer in ("n", "no", "ㅜ"):
                return reject(self.read_line("거부 이유(선택, Enter 로 건너뜀) › ").strip())

    def show_diff(self) -> None:
        result = subprocess.run(["git", "diff"], cwd=self.workspace, capture_output=True, check=False)
        text = result.stdout.decode("utf-8", errors="replace")
        if result.returncode != 0:  # 저장소 밖이면 긴 usage 가 나오므로 첫 줄만 보인다
            first = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[:1]
            self.console.print(f"[red]git diff 실패: {escape(first[0] if first else str(result.returncode))}[/]")
        elif not text.strip():
            self.console.print("[dim]변경 없음[/]")
        else:
            self.console.print(Syntax(text, "diff"))


def _line_reader(history: Path) -> ReadLine:
    """Windows 콘솔이면 prompt_toolkit(기록·편집)을, 파이프나 mintty(Git Bash 창)면 input() 을 쓴다."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return input
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    try:
        session = PromptSession(history=FileHistory(str(history)))
    except Exception:  # noqa: BLE001 - NoConsoleScreenBufferError 등. 줄 편집만 포기한다
        return input
    return lambda prompt: session.prompt(prompt)


def _utf8_stdio() -> None:
    """파이프로 출력할 때 콘솔 코드페이지(cp949)로 한글이 깨지지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="code-agent", description="Deep Agents 기반 터미널 코딩 에이전트")
    parser.add_argument("--workspace", "-w", help="작업 폴더 (기본: 현재 폴더)")
    parser.add_argument("--model", help="anthropic:<id> 또는 openrouter:<id> (기본: CODE_AGENT_MODEL)")
    parser.add_argument("--mode", choices=MODES, default="manual", help="승인 모드 (기본: manual)")
    parser.add_argument("--resume", metavar="ID", help="이전 세션을 이어 간다")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = parse_args(argv)
    console = Console(highlight=False)
    try:
        settings = load_settings()
    except ValueError as e:
        console.print(f"[red]{escape(str(e))}[/]")
        return 2
    if args.model:
        settings = replace(settings, model=args.model)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    sessions = Sessions(sqlite3.connect(settings.data_dir / "sessions.db", check_same_thread=False))

    if args.resume:
        workspace = sessions.workspace_of(args.resume)
        if workspace is None:
            console.print(f"[red]세션 {escape(args.resume)} 을 찾지 못했다.[/]")
            return 2
        if args.workspace and Path(args.workspace).resolve() != workspace:
            console.print(f"[red]세션 {escape(args.resume)} 의 작업 폴더는 {escape(str(workspace))} 다.[/]")
            return 2
    else:
        workspace = Path(args.workspace or ".").resolve()
        if not workspace.is_dir():
            console.print(f"[red]작업 폴더가 없다: {escape(str(workspace))}[/]")
            return 2
    if workspace in (Path.home(), Path(workspace.anchor)):
        console.print(f"[yellow]주의: 작업 폴더가 {escape(str(workspace))} 다. 파일 도구가 이 폴더 전체에 접근한다.[/]")

    try:
        agent = build_agent(settings, workspace, SqliteSaver(sessions.conn))
    except (RuntimeError, ValueError) as e:  # API 키 미설정, 모델 이름 오류 등
        console.print(f"[red]{escape(str(e))}[/]")
        return 2
    if settings.langfuse_enabled and not langfuse_installed():
        console.print("[dim]Langfuse 키가 있지만 패키지가 없어 트레이스를 끈다. 켜려면: uv sync --extra langfuse[/]")
    tracer = make_tracer(settings)
    thread_id = args.resume or sessions.create(workspace)  # 시작에 성공한 뒤에만 세션을 남긴다
    app = App(settings, workspace, args.mode, sessions, thread_id, console,
              _line_reader(settings.data_dir / "history"), agent=agent, tracer=tracer)
    return app.run()
