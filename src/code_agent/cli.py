"""터미널 REPL.

사용: code-agent [--workspace PATH] [--model ID] [--mode manual|auto-edit|yolo] [--resume ID]
"""

from __future__ import annotations

import argparse
import os
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

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.syntax import Syntax

from code_agent.agent import build_agent, run_turn
from code_agent.approval import (
    APPROVE,
    MODES,
    PLAN_APPROVAL_TOOL,
    PLAN_REJECT_MESSAGE,
    PlanContext,
    decide_all,
    plan_reason,
    reject,
)
from code_agent.backend import process_shell_env, resolve_shell, safe_which
from code_agent.environment import probe, render
from code_agent.planning import PLAN_PREFIX, is_approved, plan_context, render_plan
from code_agent.render import Renderer, describe_action, todos_markup
from code_agent.resume_compat import SqlitePromptStore
from code_agent.settings import REPO_ROOT, Settings, load_settings
from code_agent.stops import TurnStops
from code_agent.tracing import Tracer, langfuse_installed, make_tracer
from code_agent.usage import Usage, UsageCallback
from code_agent.worklog import session_context

HELP = """\
[bold]슬래시 커맨드[/]
  /help          이 도움말
  /new           새 세션(대화) 시작
  /mode <모드>   승인 모드 변경: manual(전부 묻기) | auto-edit(파일 쓰기 자동, 셸은 묻기) \
| yolo(전부 자동)
  /plan <요청>   계획 모드로 요청: 조사 → 계획 → 리뷰어 검토 → 승인 → 구현. \
승인 전에는 쓰기가 막힌다
  /plan          현재 계획·상태·검토·범위 변경.  /plan off: 계획 모드와 계획을 지운다
  /diff          작업 폴더의 git diff
  /todos         현재 할 일 목록
  /cost          이 세션의 모델 호출·토큰·비용 합계
  /thinking \\[on|off]  생각(reasoning) 원문 표시. 끄면(기본) 문단마다 앞부분만 짧은 메모로 보인다
  /exit          종료
Ctrl+C 는 실행 중인 턴만 취소한다. \
읽기 전용 셸 명령(git status, ls, cat 등)은 모드와 관계없이 자동 승인된다."""
YOLO_WARNING = ("[bold red]yolo 모드: 파일 쓰기와 셸 명령을 묻지 않고 실행한다. "
                "셸은 샌드박스가 아니다.[/]")

ReadLine = Callable[[str], str]
_USAGE_COLUMNS = [(f.name, "REAL" if f.type == "float" else "INTEGER") for f in fields(Usage)]


class Sessions:
    """sessions.db 에 체크포인트와 함께 thread → 작업 폴더를 남긴다.
    --resume 은 같은 작업 폴더로 이어 간다.

    thread 마다 시스템 프롬프트의 '이 PC 의 도구' 절(environment.render)도 남긴다.
    --resume 때 다시 탐지하면 그사이 설치한 도구 때문에 system 이 바뀌어
    프롬프트 캐시와 Opus 5.5 의 보존된 thinking 이 깨지기 때문이다.
    code_agent_threads 에 열을 더하지 않고 표를 따로 둔다. 열을 더하면 아직 돌고 있는 옛 프로세스의
    'INSERT … VALUES (?, ?, ?)' 가 열 수가 맞지 않아 실패한다.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute("CREATE TABLE IF NOT EXISTS code_agent_threads "
                     "(thread_id TEXT PRIMARY KEY, workspace TEXT NOT NULL, "
                     "created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS code_agent_thread_env "
                     "(thread_id TEXT PRIMARY KEY, env_text TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS code_agent_usage "
                     "(thread_id TEXT NOT NULL, at TEXT NOT NULL, "
                     f"{', '.join(f'{name} {kind} NOT NULL' for name, kind in _USAGE_COLUMNS)})")
        conn.commit()

    def create(self, workspace: Path, env_text: str = "") -> str:
        """새 세션을 남기고 thread_id 를 돌려준다. 작업 폴더와 '이 PC 의 도구' 절을 함께 남긴다."""
        thread_id = uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO code_agent_threads VALUES (?, ?, ?)",
                          (thread_id, str(workspace), datetime.now(UTC).isoformat()))
        self.conn.execute("INSERT INTO code_agent_thread_env VALUES (?, ?)", (thread_id, env_text))
        self.conn.commit()
        return thread_id

    def env_text_of(self, thread_id: str) -> str:
        """저장한 '이 PC 의 도구' 절.

        그 절이 생기기 전의 옛 thread 는 빈 문자열(옛 system 과 같게).
        """
        row = self.conn.execute("SELECT env_text FROM code_agent_thread_env WHERE thread_id = ?",
                                (thread_id,)).fetchone()
        return row[0] if row else ""

    def workspace_of(self, thread_id: str) -> Path | None:
        """세션의 작업 폴더. 없는 세션이면 None."""
        row = self.conn.execute("SELECT workspace FROM code_agent_threads WHERE thread_id = ?",
                                (thread_id,)).fetchone()
        return Path(row[0]) if row else None

    def add_usage(self, thread_id: str, usage: Usage) -> None:
        """턴 하나의 사용량을 남긴다. /cost 가 세션 단위로 합친다."""
        values = [getattr(usage, name) for name, _ in _USAGE_COLUMNS]
        self.conn.execute("INSERT INTO code_agent_usage VALUES "
                          f"(?, ?, {', '.join('?' * len(values))})",
                          (thread_id, datetime.now(UTC).isoformat(), *values))
        self.conn.commit()

    def usage_of(self, thread_id: str) -> tuple[Usage, int]:
        """(세션 합계, 기록된 턴 수)."""
        rows = self.conn.execute(f"SELECT {', '.join(name for name, _ in _USAGE_COLUMNS)} "
                                 "FROM code_agent_usage WHERE thread_id = ?",
                                 (thread_id,)).fetchall()
        total = Usage()
        for row in rows:
            total += Usage(**dict(zip((name for name, _ in _USAGE_COLUMNS), row, strict=True)))
        return total, len(rows)


class App:
    """터미널 REPL 하나. 입력을 슬래시 커맨드나 에이전트 턴으로 처리하고 결과를 그린다.

    Args:
        settings: 설정(settings.load_settings). 모델 이름·셸·데이터 폴더를 쓴다.
        workspace: 작업 폴더.
        mode: 승인 모드(approval.MODES). /mode 로 바꾼다.
        sessions: sessions.db 의 세션·사용량 기록.
        thread_id: 지금 세션(대화)의 ID. /new 로 바뀐다.
        console: 출력할 rich 콘솔.
        read_line: 프롬프트를 받아 한 줄을 읽는 함수(input 또는 prompt_toolkit).
        agent: 쓸 그래프. None 이면 session_agent 로 만든다. 시험은 대본 모델 그래프를 넣는다.
        tracer: Langfuse 트레이서. None 이면 트레이스를 남기지 않는다.
        env_text: agent 의 시스템 프롬프트에 들어간 '이 PC 의 도구' 절이다.
            /new 로 만드는 thread 에도 같은 것을 남긴다.
        fresh_agent: 환경을 다시 탐지해 (env_text, agent) 를 새로 만드는 함수.
            절이 없던 옛 thread 를 이어 온 프로세스에서 /new 가 쓴다.
    """

    def __init__(self, settings: Settings, workspace: Path, mode: str, sessions: Sessions,
                 thread_id: str, console: Console, read_line: ReadLine, *, agent=None,
                 tracer: Tracer | None = None, env_text: str = "",
                 fresh_agent: Callable[[], tuple[str, Any]] | None = None) -> None:
        self.settings, self.workspace, self.mode = settings, workspace, mode
        self.sessions, self.thread_id = sessions, thread_id
        self.console, self.read_line = console, read_line
        self.env_text = env_text
        self.fresh_agent = fresh_agent  # (env_text, agent): 환경을 다시 탐지해 새로 만든 agent
        self.agent = (agent if agent is not None
                      else session_agent(settings, workspace, sessions, env_text))
        self.tracer = tracer
        # 파이프·테스트(StringIO)에서는 상태 줄을 그리지 않는다
        self.status_line = console.is_terminal and _interactive()
        self.full_thinking = False  # /thinking

    @property
    def config(self) -> RunnableConfig:
        """지금 세션의 langgraph 실행 설정(thread_id)."""
        return {"configurable": {"thread_id": self.thread_id}}

    def run(self) -> int:
        """시작 안내를 찍고 입력 루프를 돈다. 끝나면 남은 트레이스를 보내고 0 을 돌려준다."""
        self.console.print("[bold]code-agent[/]  작업 폴더 "
                           f"[cyan]{escape(str(self.workspace))}[/]\n"
                           f"모델 {escape(self.settings.model)} · 승인 모드 {self.mode} · "
                           f"세션 {self.thread_id}  [dim](/help)[/]")
        if self.tracer is not None:
            self.console.print("[dim]Langfuse 트레이스 켜짐 → "
                               f"{escape(self.settings.langfuse_base_url)}[/]")
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
            if not self.env_text and self.fresh_agent is not None:
                # 절이 없던 옛 thread 를 이어 온 프로세스다.
                # 새 thread 는 다시 탐지한 절과 그 system 으로 시작한다
                try:
                    self.env_text, self.agent = self.fresh_agent()
                except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001 - 지금 세션은 그대로 이어 간다
                    self.console.print(f"[red]새 세션을 시작하지 못했다: {type(e).__name__}: "
                                       f"{escape(str(e))}[/]\n"
                                       f"지금 세션 {self.thread_id} 을 이어 간다.")
                    return True
            self.thread_id = self.sessions.create(self.workspace, self.env_text)
            self.console.print(f"새 세션 {self.thread_id}")
        elif name == "/mode":
            if arg not in MODES:
                self.console.print(f"[red]모드는 {', '.join(MODES)} 중 하나다. "
                                   f"지금은 {self.mode}.[/]")
            else:
                self.mode = arg
                self.console.print(f"승인 모드 → {arg}")
                if arg == "yolo":
                    self.console.print(YOLO_WARNING)
        elif name == "/plan":
            if not arg:
                self.show_plan()
            elif arg == "off":
                self.plan_off()
            else:
                self.turn(arg, plan_mode=True)
        elif name == "/diff":
            self.show_diff()
        elif name == "/todos":
            todos = self.agent.get_state(self.config).values.get("todos") or []
            self.console.print(todos_markup(todos))
        elif name == "/cost":
            total, turns = self.sessions.usage_of(self.thread_id)
            self.console.print(f"세션 {self.thread_id} · 턴 {turns}개 · {escape(total.summary())}"
                               if turns else "[dim]이 세션에 기록된 사용량이 아직 없다.[/]")
        elif name == "/thinking":
            if arg not in ("", "on", "off"):
                self.console.print(f"[red]{escape('/thinking [on|off]')}[/]")
            else:
                self.full_thinking = not self.full_thinking if not arg else arg == "on"
                self.console.print("생각 원문 표시 → " + ("켬" if self.full_thinking
                                                    else "끔(문단마다 앞부분만 메모로)"))
        else:
            self.console.print(f"[red]알 수 없는 커맨드: {escape(name)}[/] (/help)")
        return True

    def turn(self, text: str, *, plan_mode: bool = False) -> None:
        """사용자 메시지 하나로 에이전트 턴을 돌린다.

        스트림 조각을 화면·멈춤 감지·트레이스에 나눠 주고, 승인 요청은 모드에 따라 정하거나 묻는다.
        중단(Ctrl+C)과 오류는 화면과 트레이스에 남기고 REPL 은 이어 간다.

        Args:
            text: 사용자가 입력한 요청.
            plan_mode: 계획 모드(/plan <요청>)로 보낼지. 참이면 입력에 plan_mode 를 넣고
                메시지 앞에 순서 안내(planning.PLAN_PREFIX)를 붙인다.

        Returns:
            None. 턴의 사용량은 sessions.db 에 남기고 한 줄로 보인다.
        """
        renderer = Renderer(self.console, status=self.status_line,
                            full_reasoning=self.full_thinking)
        stops = TurnStops()
        meter = UsageCallback()
        tags = [self.workspace.name, f"mode:{self.mode}", f"model:{self.settings.model}"]
        trace_context: Any = nullcontext()
        if self.tracer is not None:
            previous = self.agent.get_state(self.config).values.get("messages", [])
            trace_context = self.tracer.turn(text, session_id=self.thread_id, tags=tags,
                                             workspace=self.workspace,
                                             context=session_context(previous, self.workspace))
        with trace_context as trace:
            def decide(action_requests: list[dict]) -> list[dict]:
                # 결정 시점의 계획(검토·승인 여부, 범위)으로 정한다.
                # interrupt 로 멈춘 동안이라 state 가 최신이다
                plan = plan_context(self.agent.get_state(self.config).values)

                def ask(action: dict) -> dict:
                    renderer.pause()  # 상태 줄이 승인 프롬프트와 입력 화면을 깨지 않게 멈춘다
                    return self.ask(action, plan)

                decisions = decide_all(action_requests, self.mode, ask,
                                       on_decision=trace.decision if trace is not None else None,
                                       plan=plan)
                renderer.resume()
                return decisions

            def on_part(part: tuple) -> None:
                renderer.on_part(part)
                stops.on_part(part)
                if trace is not None:
                    trace.on_part(part)  # 작업 기록(도구 실행과 결과 요약)

            callbacks = [meter, *(trace.callbacks if trace is not None else [])]
            error = None
            try:
                renderer.start()
                run_turn(self.agent, self._turn_input(text, plan_mode),
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
            # 잘린 응답·예산 상한 도달. 말없이 멈춘 것처럼 보이지 않게 한다
            warnings = stops.warnings()
            for warning in warnings:
                self.console.print(f"[yellow]{escape(warning)}[/]", soft_wrap=True)
            if trace is not None:
                trace.end(output=renderer.last_text, usage=meter.usage, error=error,
                          warnings=warnings)
        if meter.usage.calls:  # 중단·오류로 끝난 턴도 쓴 만큼은 남긴다
            self.sessions.add_usage(self.thread_id, meter.usage)
            self.console.print(f"[dim]{escape(meter.usage.summary())}[/]")

    def _turn_input(self, text: str, plan_mode: bool) -> dict[str, Any]:
        """턴 입력. 계획 모드면 plan_mode 를 켜고 메시지 앞에 순서 안내를 붙인다.

        승인된 계획이 남아 있으면 지운다.
        끝난 요청의 계획이라, 두면 새 요청의 쓰기가 막히지 않는다.
        """
        if not plan_mode:
            return {"messages": [{"role": "user", "content": text}]}
        inp: dict[str, Any] = {"messages": [{"role": "user", "content": PLAN_PREFIX + text}],
                               "plan_mode": True}
        if is_approved(self.agent.get_state(self.config).values.get("plan")):
            inp["plan"] = None
        return inp

    def ask(self, action: dict, plan: PlanContext | None = None) -> dict:
        """승인 요청 하나를 보여 주고 y/n 을 받는다. 거부면 이유(선택)도 받아 모델에 넘긴다.

        계획 승인 요청은 계획과 검토 의견을 보인다(ask_plan).
        승인된 계획 밖 파일 쓰기는 그 이유(approval.plan_reason)를 함께 보인다.
        """
        if action["name"] == PLAN_APPROVAL_TOOL:
            return self.ask_plan(action)
        self.console.print(describe_action(action, reason=plan_reason(action, plan)))
        if self._yes("[y] 승인 / [n] 거부 › "):
            return APPROVE
        return reject(self.read_line("거부 이유(선택, Enter 로 건너뜀) › ").strip())

    def ask_plan(self, action: dict) -> dict:
        """계획 승인 요청. state 의 계획(단계·완료 조건)과 검토 의견을 보이고 y/n 을 받는다.

        거부 의견은 모델에 가는 거부 메시지(approval.PLAN_REJECT_MESSAGE) 끝에 붙는다.
        """
        plan = self.agent.get_state(self.config).values.get("plan")
        self.console.print(Markdown(render_plan(plan)) if plan else "[red]계획이 없다.[/]")
        if note := str((action.get("args") or {}).get("note") or "").strip():
            self.console.print(f"[bold]에이전트 메모[/] {escape(note)}")
        self.console.print("[bold yellow]승인 요청: 이 계획대로 구현할까?[/] "
                           "승인하면 단계가 할 일이 되고 쓰기 차단이 풀린다.")
        if self._yes("[y] 승인 / [n] 거부(의견) › "):
            return APPROVE
        return reject(self.read_line("거부 의견(선택, Enter 로 건너뜀) › ").strip(),
                      base=PLAN_REJECT_MESSAGE)

    def _yes(self, prompt: str) -> bool:
        """y 면 참, n 이면 거짓. 다른 입력이면 다시 묻는다."""
        while True:
            answer = self.read_line(prompt).strip().lower()
            if answer in ("y", "yes", "ㅛ"):
                return True
            if answer in ("n", "no", "ㅜ"):
                return False

    def show_plan(self) -> None:
        """/plan: 현재 계획(단계·완료 조건·검토·범위 변경)과 상태를 보인다."""
        values = self.agent.get_state(self.config).values
        plan = values.get("plan")
        if plan:
            self.console.print(Markdown(render_plan(plan)))
        elif values.get("plan_mode"):
            self.console.print("[dim]계획 없음. 계획 모드가 켜져 있어 계획부터 세운다.[/]")
        else:
            self.console.print("[dim]계획 없음.[/]")

    def plan_off(self) -> None:
        """/plan off: 계획 모드와 계획을 지운다(체크포인트에 남는다).

        이후 쓰기는 승인 모드대로 처리한다.
        """
        values = self.agent.get_state(self.config).values
        if not values.get("plan") and not values.get("plan_mode"):
            self.console.print("[dim]지울 계획이 없다.[/]")
            return
        self.agent.update_state(self.config, {"plan_mode": False, "plan": None})
        self.console.print("계획 모드와 계획을 지웠다. 이후 쓰기는 승인 모드대로 처리한다.")

    def show_diff(self) -> None:
        """작업 폴더의 git diff 를 보인다(/diff)."""
        # 이름만 주면 현재 폴더에 심은 git.exe 가 먼저 불릴 수 있다
        git = safe_which("git", self.workspace)
        if git is None:
            self.console.print("[red]git 을 찾지 못했다.[/]")
            return
        result = subprocess.run([git, "diff"], cwd=self.workspace, capture_output=True,
                                check=False)
        text = result.stdout.decode("utf-8", errors="replace")
        if result.returncode != 0:  # 저장소 밖이면 긴 usage 가 나오므로 첫 줄만 보인다
            first = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[:1]
            self.console.print("[red]git diff 실패: "
                               f"{escape(first[0] if first else str(result.returncode))}[/]")
        elif not text.strip():
            self.console.print("[dim]변경 없음[/]")
        else:
            self.console.print(Syntax(text, "diff"))


def _interactive() -> bool:
    """입출력이 모두 터미널인지. 파이프나 mintty(Git Bash 창)는 아니다."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _line_reader(history: Path) -> ReadLine:
    """Windows 콘솔이면 prompt_toolkit(기록·편집)을,
    파이프나 mintty(Git Bash 창)면 input() 을 쓴다.
    """
    if not _interactive():
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
        # TextIO 타입에는 없다. 실제 콘솔은 TextIOWrapper 다
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def repo_warning(workspace: Path) -> str | None:
    """작업 폴더가 code-agent 저장소(실제 키가 든 .env 가 있다)이거나
    그것을 품은 폴더면 경고 문구.
    """
    root = REPO_ROOT.resolve()
    if not (root / ".env").is_file() or not root.is_relative_to(workspace.resolve()):
        return None
    where = "code-agent 저장소" if root == workspace.resolve() else "code-agent 저장소를 품은 폴더"
    return f"작업 폴더가 {where}다. 실제 키가 든 .env 가 있다. 다른 폴더를 -w 로 지정하라."


def _confirm(prompt: str) -> bool:
    """대화형이면 y/N 을 묻는다. 파이프 입력이면 첫 요청을 먹어 버리므로 묻지 않고 진행한다."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return True
    return input(prompt).strip().lower() in ("y", "yes", "ㅛ")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """명령줄 인자(--workspace, --model, --mode, --resume)를 읽는다."""
    parser = argparse.ArgumentParser(prog="code-agent",
                                     description="Deep Agents 기반 터미널 코딩 에이전트")
    parser.add_argument("--workspace", "-w", help="작업 폴더 (기본: 현재 폴더)")
    parser.add_argument("--model",
                        help="anthropic:<id> 또는 openrouter:<id> (기본: CODE_AGENT_MODEL)")
    parser.add_argument("--mode", choices=MODES, default="manual", help="승인 모드 (기본: manual)")
    parser.add_argument("--resume", metavar="ID", help="이전 세션을 이어 간다")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """code-agent 명령의 진입점. 설정·작업 폴더·세션을 준비하고 REPL 을 돈다.

    Args:
        argv: 명령줄 인자. None 이면 sys.argv[1:] 을 쓴다.

    Returns:
        종료 코드. 정상 종료는 0, 설정·작업 폴더·세션·모델 준비에 실패해 시작하지 못하면 2.
    """
    if sys.platform == "win32":
        # 이것이 없으면 shutil.which 가 현재 폴더를 먼저 찾아, 악성 저장소 폴더에서 띄우기만 해도
        # 그 안의 git.cmd 등이 불릴 수 있다. Python 3.12+ 의 which 는 이 변수를 따른다.
        # 셸 백엔드도 이 환경을 물려받는다
        os.environ.setdefault("NoDefaultCurrentDirectoryInExePath", "1")
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
    sessions = Sessions(sqlite3.connect(settings.data_dir / "sessions.db",
                                        check_same_thread=False))

    if args.resume:
        workspace = sessions.workspace_of(args.resume)
        if workspace is None:
            console.print(f"[red]세션 {escape(args.resume)} 을 찾지 못했다.[/]")
            return 2
        if args.workspace and Path(args.workspace).resolve() != workspace:
            console.print(f"[red]세션 {escape(args.resume)} 의 작업 폴더는 "
                          f"{escape(str(workspace))} 다.[/]")
            return 2
    else:
        workspace = Path(args.workspace or ".").resolve()
        if not workspace.is_dir():
            console.print(f"[red]작업 폴더가 없다: {escape(str(workspace))}[/]")
            return 2
    if workspace in (Path.home().resolve(), Path(workspace.anchor)):
        console.print(f"[yellow]주의: 작업 폴더가 {escape(str(workspace))} 다. "
                      "파일 도구가 이 폴더 전체에 접근한다.[/]")
    if warning := repo_warning(workspace):
        console.print(f"[bold yellow]주의: {escape(warning)}[/]")
        # -w 를 빼먹어 저장소 폴더에서 띄운 경우가 많아 한 번 확인한다.
        # 명시했거나 이어 가는 세션이면 묻지 않는다
        if (not args.workspace and not args.resume
                and not _confirm("그래도 이 폴더에서 시작할까? [y/N] › ")):
            console.print("[dim]시작하지 않았다. 작업할 폴더를 -w 로 지정하라.[/]")
            return 2

    try:
        # --resume 은 그 thread 에 저장한 절을 그대로 쓴다(다시 탐지하면 system 이 바뀔 수 있다)
        env_text = (sessions.env_text_of(args.resume) if args.resume
                    else detect_environment(settings.shell, workspace))
        agent = session_agent(settings, workspace, sessions, env_text)
    except (RuntimeError, ValueError) as e:  # API 키 미설정, 모델 이름 오류, Git Bash 없음 등
        console.print(f"[red]{escape(str(e))}[/]")
        return 2
    if settings.langfuse_enabled and not langfuse_installed():
        console.print("[dim]Langfuse 키가 있지만 패키지가 없어 트레이스를 끈다. "
                      "켜려면: uv sync --extra langfuse[/]")
    tracer = make_tracer(settings)
    # 시작에 성공한 뒤에만 세션을 남긴다
    thread_id = args.resume or sessions.create(workspace, env_text)

    def fresh_agent() -> tuple[str, Any]:
        text = detect_environment(settings.shell, workspace)
        return text, session_agent(settings, workspace, sessions, text)

    app = App(settings, workspace, args.mode, sessions, thread_id, console,
              _line_reader(settings.data_dir / "history"), agent=agent, tracer=tracer,
              env_text=env_text, fresh_agent=fresh_agent)
    return app.run()


def session_agent(settings: Settings, workspace: Path, sessions: Sessions, env_text: str) -> Any:
    """sessions.db 에 체크포인트와 프롬프트 지문(resume_compat)을 남기는 agent.

    지문 저장소는 그래프 실행 중(모델 호출 직전)에 쓰고, 그때 SqliteSaver 는 앞 단계의 체크포인트를
    백그라운드에서 같은 연결로 저장할 수 있다. 그래서 두 쪽이 같은 잠금을 쓴다.
    """
    saver = SqliteSaver(sessions.conn)
    return build_agent(settings, workspace, saver, env_text=env_text,
                       prompt_store=SqlitePromptStore(sessions.conn, saver.lock))


def detect_environment(shell_preference: str, workspace: Path | None = None) -> str:
    """셸 백엔드가 쓸 환경(process_shell_env)으로 도구를 확인해 '이 PC 의 도구' 절을 만든다.

    약 1초 걸린다.
    """
    shell = resolve_shell(shell_preference, workspace)[0]
    return render(probe(process_shell_env(workspace), workspace=workspace), shell)
