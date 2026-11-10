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
from rich.text import Text

from code_agent.agent import build_agent, run_turn
from code_agent.approval import (
    APPROVE,
    MODES,
    PLAN_APPROVAL_TOOL,
    PLAN_REJECT_MESSAGE,
    PlanContext,
    ask_reason,
    decide_all,
    reject,
)
from code_agent.backend import process_shell_env, resolve_shell, safe_which
from code_agent.environment import probe, render
from code_agent.localtrace import LocalTracer, run_dir, trace_view
from code_agent.memory import (
    MEMORY_FILE,
    PROJECT_MEMORY,
    REMEMBER_DONE,
    WARN_CHARS,
    MemoryTrust,
    decode_memory,
    library_logs,
    loaded_chars,
    memory_report,
    pending_chars,
    project_skill_files,
    read_memory,
    remember_prompt,
    skills_sha256,
    trust_key,
)
from code_agent.planning import (
    PLAN_PREFIX,
    SCOPE_KEY,
    is_approved,
    plan_context,
    render_plan,
)
from code_agent.render import Renderer, describe_action, strip_controls, todos_markup
from code_agent.resume_compat import SqlitePromptStore
from code_agent.settings import REPO_ROOT, Settings, load_settings
from code_agent.stops import TurnStops
from code_agent.tracing import (
    CompositeTracer,
    Tracer,
    combine_tracers,
    langfuse_installed,
    make_tracer,
    tracer_notices,
)
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
  /trace \\[n]     마지막(또는 n번째) 턴의 타임라인(모델·도구·승인, 소요 시간, 상태)
  /trace errors  이 세션의 실패한 도구·오류·경고
  /remember \\[메모]  이번 대화에서 확인된 규칙·사실을 메모리(AGENTS.md)에 적게 한다. \
다음 /new 부터 반영된다
  /memory        이 세션이 읽은 메모리 파일(글자 수, 그 뒤 바뀌었는지)과 스킬
  /thinking \\[on|off]  생각(reasoning) 원문 표시. 끄면(기본) 문단마다 앞부분만 짧은 메모로 보인다
  /exit          종료
Ctrl+C 는 실행 중인 턴만 취소한다. \
읽기 전용 셸 명령(git status, ls, cat 등)은 모드와 관계없이 자동 승인된다."""
YOLO_WARNING = ("[bold red]yolo 모드: 파일 쓰기와 셸 명령을 묻지 않고 실행한다. "
                "셸은 샌드박스가 아니다.[/]")

MEMORY_PREVIEW_LINES = 15  # 신뢰를 물을 때 보일 AGENTS.md 앞부분
MEMORY_PREVIEW_WIDTH = 120  # 그 한 줄의 최대 글자 수
NOT_REMEMBERED = "메모리에 쓴 것이 없다."

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

    프로젝트 AGENTS.md 를 메모리로 읽을지 정한 결정도 남긴다(작업 폴더와 내용 해시마다,
    code_agent_trusted_memory). 내용이 바뀌면 해시가 달라 다시 묻는다.
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
        conn.execute("CREATE TABLE IF NOT EXISTS code_agent_trusted_memory "
                     "(workspace TEXT NOT NULL, sha256 TEXT NOT NULL, trusted INTEGER NOT NULL, "
                     "decided_at TEXT NOT NULL, PRIMARY KEY (workspace, sha256))")
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

    @staticmethod
    def _workspace_key(workspace: Path) -> str:
        """결정을 묶는 작업 폴더 이름. 같은 폴더를 다른 철자(대소문자·상대 경로)로 열어도 같다."""
        return os.path.normcase(str(Path(workspace).resolve()))

    def memory_decision(self, workspace: Path, sha256: str) -> bool | None:
        """이 작업 폴더의 이 내용(sha256)인 AGENTS.md 를 신뢰했는지. 정한 적이 없으면 None."""
        row = self.conn.execute("SELECT trusted FROM code_agent_trusted_memory "
                                "WHERE workspace = ? AND sha256 = ?",
                                (self._workspace_key(workspace), sha256)).fetchone()
        return bool(row[0]) if row else None

    def trusted_before(self, workspace: Path) -> bool:
        """이 작업 폴더의 AGENTS.md 를 (다른 내용일 때라도) 신뢰한 적이 있는지."""
        return self.conn.execute("SELECT 1 FROM code_agent_trusted_memory "
                                 "WHERE workspace = ? AND trusted = 1",
                                 (self._workspace_key(workspace),)).fetchone() is not None

    def decide_memory(self, workspace: Path, sha256: str, trusted: bool) -> None:
        """프로젝트 AGENTS.md 를 메모리로 읽을지 정한 결정을 남긴다."""
        self.conn.execute("INSERT OR REPLACE INTO code_agent_trusted_memory VALUES (?, ?, ?, ?)",
                          (self._workspace_key(workspace), sha256, int(trusted),
                           datetime.now(UTC).isoformat()))
        self.conn.commit()


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
        tracer: 트레이서(Langfuse·로컬, 둘이면 CompositeTracer).
            None 이면 트레이스를 남기지 않는다.
        env_text: agent 의 시스템 프롬프트에 들어간 '이 PC 의 도구' 절이다.
            /new 로 만드는 thread 에도 같은 것을 남긴다.
        fresh_agent: 환경을 다시 탐지해 (env_text, agent) 를 새로 만드는 함수.
            절이 없던 옛 thread 를 이어 온 프로세스에서 /new 가 쓴다.
        memory_trust: agent 와 나눠 쓰는 프로젝트 AGENTS.md 신뢰(memory.MemoryTrust).
            새 thread 를 시작할 때 사람에게 묻고 고친다. None 이면 새로 만든다(agent 를 주면 그
            agent 는 이것을 보지 않는다).
        interactive: 사람에게 y/N 을 물을 수 있는지. None 이면 입출력이 터미널인지로 정한다.
            파이프 입력에서 물으면 첫 요청을 답으로 먹으므로 묻지 않는다.
    """

    def __init__(self, settings: Settings, workspace: Path, mode: str, sessions: Sessions,
                 thread_id: str, console: Console, read_line: ReadLine, *, agent=None,
                 tracer: Tracer | None = None, env_text: str = "",
                 fresh_agent: Callable[[], tuple[str, Any]] | None = None,
                 memory_trust: MemoryTrust | None = None,
                 interactive: bool | None = None) -> None:
        self.settings, self.workspace, self.mode = settings, workspace, mode
        self.sessions, self.thread_id = sessions, thread_id
        self.console, self.read_line = console, read_line
        self.env_text = env_text
        self.fresh_agent = fresh_agent  # (env_text, agent): 환경을 다시 탐지해 새로 만든 agent
        self.memory_trust = memory_trust if memory_trust is not None else MemoryTrust()
        self.agent = (agent if agent is not None
                      else session_agent(settings, workspace, sessions, env_text,
                                         memory_trust=self.memory_trust))
        self.tracer = tracer
        self.interactive = _interactive() if interactive is None else interactive
        # 파이프·테스트(StringIO)에서는 상태 줄을 그리지 않는다
        self.status_line = console.is_terminal and _interactive()
        self.full_thinking = False  # /thinking
        self._refused_memory: set[str] = set()  # 이 프로세스에서 거절한 AGENTS.md·스킬(trust_key)
        self._memory_written = False  # 이번 턴에 메모리 파일 쓰기가 실행됐는지
        self._project_written = False  # 그중 프로젝트 /AGENTS.md 쓰기가 있었는지
        self._own_write_ok = False  # 이 thread 의 프로젝트 메모리가 신뢰했거나 없던 것인지

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
        tracers = (self.tracer.tracers if isinstance(self.tracer, CompositeTracer)
                   else [self.tracer] if self.tracer is not None else [])
        if any(not isinstance(tracer, LocalTracer) for tracer in tracers):
            self.console.print("[dim]Langfuse 트레이스 켜짐 → "
                               f"{escape(self.settings.langfuse_base_url)}[/]")
        for tracer in tracers:
            if isinstance(tracer, LocalTracer):
                self.console.print(f"[dim]로컬 트레이스 → "
                                   f"{escape(str(tracer.folder(self.thread_id)))} (/trace)[/]")
        if self.mode == "yolo":
            self.console.print(YOLO_WARNING)
        # deepagents 의 경고(없는 스킬 폴더 등)가 stderr 로 나가 상태 줄을 깨지 않게 파일로 보낸다
        with library_logs(self.settings.data_dir):
            try:
                self.start_memory()
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
            self.start_memory()
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
            elif arg.lower() == "off":
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
        elif name == "/trace":
            self.show_trace(arg)
        elif name == "/remember":
            self.turn(remember_prompt(arg))
            done = REMEMBER_DONE if self._memory_written else NOT_REMEMBERED
            self.console.print(f"[dim]{done}[/]")
        elif name == "/memory":
            self.show_memory()
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
        self._memory_written = self._project_written = False
        tags = [self.workspace.name, f"mode:{self.mode}", f"model:{self.settings.model}"]
        trace_context: Any = nullcontext()
        if self.tracer is not None:
            trace_context = self.tracer.turn(text, session_id=self.thread_id, tags=tags,
                                             workspace=self.workspace,
                                             context=self._trace_context())
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
                self._note_memory_write(part)
                renderer.on_part(part)
                stops.on_part(part)
                if trace is not None:
                    trace.on_part(part)  # 작업 기록(도구 실행과 결과 요약)

            callbacks = [meter, *(trace.callbacks if trace is not None else [])]
            error, interrupted = None, False
            try:
                inp = self._turn_input(text, plan_mode)
                renderer.start()
                run_turn(self.agent, inp, {**self.config, "callbacks": callbacks},
                         decide=decide, on_part=on_part)
            except KeyboardInterrupt:
                renderer.end_turn()
                self.console.print("[yellow]중단했다.[/]")
                error, interrupted = "사용자가 중단", True
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
                          warnings=warnings, retries=stops.recoveries, interrupted=interrupted)
        if self._project_written:
            self.trust_own_write()
        for notice in tracer_notices(self.tracer):  # 로컬 트레이스 기록 실패 등(한 번만)
            self.console.print(f"[dim]{escape(notice)}[/]", soft_wrap=True)
        if meter.usage.calls:  # 중단·오류로 끝난 턴도 쓴 만큼은 남긴다
            self.sessions.add_usage(self.thread_id, meter.usage)
            self.console.print(f"[dim]{escape(meter.usage.summary())}[/]")

    def _note_memory_write(self, part: tuple) -> None:
        """스트림 조각이 memory_write 사건이면 이번 턴의 쓰기 기록에 남긴다."""
        if (len(part) == 3 and part[1] == "custom" and isinstance(part[2], dict)
                and part[2].get("type") == "memory_write"):
            self._memory_written = True
            self._project_written |= part[2].get("scope") == "project"

    def _trace_context(self) -> str:
        """Langfuse 평가기에 줄 이전 턴 요약(worklog.session_context).

        쓰는 트레이서가 있을 때만 state 를 읽는다(로컬 트레이스는 쓰지 않는다).
        읽다가 실패해도 트레이스 맥락만 비우고 턴은 돈다.
        """
        if not getattr(self.tracer, "uses_context", True):
            return ""
        try:
            previous = self.agent.get_state(self.config).values.get("messages", [])
            return session_context(previous, self.workspace)
        except Exception:  # noqa: BLE001 - 트레이스 맥락 때문에 턴을 멈추지 않는다
            return ""

    def _turn_input(self, text: str, plan_mode: bool) -> dict[str, Any]:
        """턴 입력.

        계획 모드면 plan_mode 를 켜고 메시지 앞에 순서 안내를 붙인다. 남은 계획은 승인 여부와
        관계없이 지워 새 계획으로 시작한다. 이전 요청의 검토를 물려받으면 새 계획이 검토 없이
        승인 요청까지 갈 수 있고, 승인된 계획을 두면 새 요청의 쓰기가 막히지 않는다.

        보통 요청이면, 승인된 계획의 할 일이 모두 끝났을 때 계획을 지우고 흐린 안내를 보인다.
        끝난 계획의 범위가 다음 요청의 쓰기마다 '계획에 없는 파일' 재승인을 묻지 않게 한다.
        할 일이 남았으면 그대로 둔다(그 재승인 문구가 /plan off 를 알려 준다).
        """
        if plan_mode:
            return {"messages": [{"role": "user", "content": PLAN_PREFIX + text}],
                    "plan_mode": True, "plan": None, SCOPE_KEY: None}
        inp: dict[str, Any] = {"messages": [{"role": "user", "content": text}]}
        values = self.agent.get_state(self.config).values
        plan, todos = values.get("plan"), values.get("todos") or []
        if (is_approved(plan) and todos
                and all(todo.get("status") == "completed" for todo in todos)):
            inp.update({"plan": None, SCOPE_KEY: None, "plan_mode": False})
            self.console.print(f"[dim]계획 v{plan.get('version')} 이 끝나 계획 범위를 "
                               "해제했다.[/]")
        return inp

    def ask(self, action: dict, plan: PlanContext | None = None) -> dict:
        """승인 요청 하나를 보여 주고 y/n 을 받는다. 거부면 이유(선택)도 받아 모델에 넘긴다.

        계획 승인 요청은 계획과 검토 의견을 보인다(ask_plan).
        모드와 관계없이 묻는 이유(사용자 공통 메모리 쓰기, 승인된 계획 밖 파일 쓰기,
        approval.ask_reason)가 있으면 함께 보인다.
        """
        if action["name"] == PLAN_APPROVAL_TOOL:
            return self.ask_plan(action)
        self.console.print(describe_action(action, reason=ask_reason(action, plan)))
        if self._yes("[y] 승인 / [n] 거부 › "):
            return APPROVE
        return reject(self.read_line("거부 이유(선택, Enter 로 건너뜀) › ").strip())

    def ask_plan(self, action: dict) -> dict:
        """계획 승인 요청. state 의 계획(단계·완료 조건)과 검토 의견을 보이고 y/n 을 받는다.

        거부 의견은 모델에 가는 거부 메시지(approval.PLAN_REJECT_MESSAGE) 끝에 붙는다.
        마지막 검토의 VERDICT 를 읽지 못했으면(unknown) 노란 줄로 알린다(원문은 계획에 붙는다).
        """
        values = self.agent.get_state(self.config).values
        plan = values.get("plan")
        self.console.print(Markdown(render_plan(plan, values.get(SCOPE_KEY))) if plan
                           else "[red]계획이 없다.[/]")
        reviews = (plan or {}).get("reviews") or []
        if reviews and reviews[-1].get("verdict") == "unknown":
            self.console.print("[yellow]리뷰 결과를 해석하지 못했다(원문 표시).[/]")
        note = strip_controls(str((action.get("args") or {}).get("note") or "")).strip()
        if note:
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

    def start_memory(self) -> None:
        """thread 를 시작할 때(시작, /new) 메모리를 준비한다.

        이미 메모리를 읽은 thread(--resume)는 그대로 쓴다(state 의 memory_contents). 아직 읽지 않은
        thread 는 프로젝트 AGENTS.md·스킬 신뢰를 정한다(trust_project_memory).
        첫 요청 때 읽힐 합계가 WARN_CHARS 를 넘으면 노란 줄로 알린다.
        매 요청 system 에 들어가 캐시 읽기 비용이 늘기 때문이다.
        """
        values = self.agent.get_state(self.config).values
        total = loaded_chars(values)
        absent = read_memory(self.workspace / MEMORY_FILE) is None
        if total is None:
            self.trust_project_memory()
            total = pending_chars(workspace=self.workspace, data_dir=self.settings.data_dir,
                                  trust=self.memory_trust)
            self._own_write_ok = self.memory_trust.sha256 is not None or absent
        else:
            contents = values.get("memory_contents")
            self._own_write_ok = absent or (isinstance(contents, dict)
                                            and PROJECT_MEMORY in contents)
        if total > WARN_CHARS:
            self.console.print(f"[yellow]메모리(AGENTS.md)가 합쳐 {total:,}자다"
                               f"({WARN_CHARS:,}자 초과). 매 요청 시스템 프롬프트에 들어가 "
                               "비용이 는다. 줄이려면 /memory 로 위치를 확인하라.[/]")

    def trust_project_memory(self) -> None:
        """프로젝트 AGENTS.md 와 프로젝트 스킬을 읽을지 정해 memory_trust 에 담는다.

        저장소가 심은 지시가 system 에 들어가지 않게, 처음 보는(또는 바뀐) 내용이면 y/N 을 묻는다
        (S11). AGENTS.md 와 스킬 설명은 둘 다 system 에 들어가므로 한 번에 묻고, 하나라도 바뀌면
        다시 묻는다. 신뢰한 결정은 sessions.db 에 남아 같은 내용이면 다시 묻지 않는다. 거절은 이
        프로세스 동안만 기억하고 다음 실행 때 다시 묻는다. 대화형이 아니면 묻지 않고 읽지 않는다.
        내용이 빈 AGENTS.md 는 스킬이 없으면 묻지 않는다.
        """
        trust = self.memory_trust
        trust.sha256 = trust.skills_sha256 = None
        data = read_memory(self.workspace / MEMORY_FILE)
        skills = skills_sha256(self.workspace)
        if data is not None and not decode_memory(data).strip() and skills is None:
            trust.sha256 = trust_key(data, None)  # 읽어도 system 에 넣을 것이 없다
            return
        key = trust_key(data, skills)
        if key is None:
            return
        if self.sessions.memory_decision(self.workspace, key) or self._ask_trust(
                key, data, skills):
            trust.skills_sha256 = skills
            trust.sha256 = None if data is None else trust_key(data, None)

    def _ask_trust(self, key: str, data: bytes | None, skills: str | None) -> bool:
        """프로젝트 AGENTS.md·스킬을 신뢰할지 사람에게 묻는다. 물을 수 없으면 거짓."""
        text = "" if data is None else decode_memory(data)
        count = len(project_skill_files(self.workspace)) if skills is not None else 0
        extra = f", 스킬 {count}개 포함" if count else ""
        what = f"프로젝트 AGENTS.md({'없음' if data is None else f'{len(text):,}자'}{extra})"
        if key in self._refused_memory:
            self.console.print(f"[dim]{what}는 이번 실행에서 거절해 메모리로 읽지 않는다.[/]")
            return False
        if not self.interactive:
            self.console.print(f"[dim]{what}는 신뢰를 확인하지 않아 메모리로 읽지 않는다"
                               "(대화형으로 실행하면 묻는다).[/]")
            return False
        changed = ("내용이 바뀌었다. " if self.sessions.trusted_before(self.workspace) else "")
        self.console.print(f"[bold yellow]{changed}이 프로젝트에 AGENTS.md 나 스킬이 있다.[/] "
                           "메모리로 읽으면 매 요청 시스템 프롬프트에 들어가 에이전트가 지시로 "
                           "참고한다.")
        self.console.print(Text(self._trust_preview(text, skills is not None), style="dim"))
        trusted = self._yes_no(f"이 {what}를 메모리로 읽을까? [y/N] › ")
        self.sessions.decide_memory(self.workspace, key, trusted)
        if not trusted:
            self._refused_memory.add(key)
            self.console.print("[dim]읽지 않는다. 이 세션 메모리에서 뺐다.[/]")
        return trusted

    def _trust_preview(self, text: str, has_skills: bool) -> str:
        """신뢰를 물을 때 보일 앞부분. 제어 문자를 빼고 긴 줄과 긴 파일을 줄인다.

        저장소가 심은 파일이라 터미널 제어 문자로 질문 화면을 지우거나 가리지 못하게 한다.
        """
        def clip(line: str) -> str:
            return (line if len(line) <= MEMORY_PREVIEW_WIDTH
                    else line[:MEMORY_PREVIEW_WIDTH] + "…")

        lines = strip_controls(text).splitlines()
        shown = [clip(line) for line in lines[:MEMORY_PREVIEW_LINES]]
        if len(lines) > MEMORY_PREVIEW_LINES:
            shown.append(f"… 나머지 {len(lines) - MEMORY_PREVIEW_LINES}줄 생략")
        if has_skills:
            for path in project_skill_files(self.workspace):
                name = strip_controls(path.relative_to(self.workspace).as_posix())
                shown.append(clip(f"스킬 {name}{_description(path)}"))
        return "\n".join(shown)

    def trust_own_write(self) -> None:
        """에이전트가 프로젝트 /AGENTS.md 를 쓴 뒤의 내용을 이 작업 폴더에서 신뢰한 것으로 남긴다.

        실행된 쓰기(승인 모드를 거친 것)라 다음 /new 가 방금 쓴 내용을 다시 묻지 않게 한다.
        thread 를 시작할 때 신뢰했거나 없던 파일일 때만 남긴다. 거절한 파일에 덧붙인 것을 신뢰하면
        거절한 내용까지 읽게 된다. 스킬은 신뢰한 그대로일 때만 함께 남긴다.
        """
        data = read_memory(self.workspace / MEMORY_FILE)
        skills = skills_sha256(self.workspace)
        key = trust_key(data, skills)
        if (key is None or data is None or not self._own_write_ok
                or (skills is not None and not self.memory_trust.allows_skills(self.workspace))):
            return
        self.sessions.decide_memory(self.workspace, key, True)

    def _yes_no(self, prompt: str) -> bool:
        """y 면 참. 그냥 Enter 나 n 이면 거짓(기본 No). 다른 입력이면 다시 묻는다."""
        while True:
            answer = self.read_line(prompt).strip().lower()
            if answer in ("y", "yes", "ㅛ"):
                return True
            if answer in ("", "n", "no", "ㅜ"):
                return False

    def show_memory(self) -> None:
        """/memory: 이 세션의 메모리 파일(위치, 읽은 글자 수, 그 뒤 바뀌었는지)과 스킬."""
        values = self.agent.get_state(self.config).values
        for line in memory_report(values, workspace=self.workspace,
                                  data_dir=self.settings.data_dir, trust=self.memory_trust):
            self.console.print(Text(line))

    def show_plan(self) -> None:
        """/plan: 현재 계획(단계·완료 조건·검토·범위 변경)과 상태를 보인다."""
        values = self.agent.get_state(self.config).values
        plan = values.get("plan")
        if plan:
            self.console.print(Markdown(render_plan(plan, values.get(SCOPE_KEY))))
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
        self.agent.update_state(self.config, {"plan_mode": False, "plan": None, SCOPE_KEY: None})
        self.console.print("계획 모드와 계획을 지웠다. 이후 쓰기는 승인 모드대로 처리한다.")

    def show_trace(self, arg: str) -> None:
        """이 세션의 로컬 트레이스를 보인다(/trace [n|errors], localtrace.trace_view)."""
        view = trace_view(run_dir(self.settings.data_dir, self.thread_id), arg)
        if view is not None:
            self.console.print(view)
        elif not self.settings.local_trace:
            self.console.print("[dim]로컬 트레이스가 꺼져 있다(CODE_AGENT_LOCAL_TRACE=0).[/]")
        else:
            self.console.print("[dim]이 세션의 로컬 트레이스가 아직 없다. 턴을 마치면 생긴다"
                               "(로컬 트레이스가 생기기 전에 시작한 세션의 옛 턴은 없다).[/]")

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


def _description(path: Path) -> str:
    """스킬 파일의 description 줄(없으면 빈 글). 신뢰를 물을 때 보인다."""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:20]:
            if line.startswith("description:"):
                return " — " + strip_controls(line.removeprefix("description:").strip())
    except OSError:
        pass
    return ""


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

    # 프로젝트 AGENTS.md 신뢰. App 이 thread 를 시작할 때 정하고 agent 들이 나눠 본다
    trust = MemoryTrust()
    try:
        # --resume 은 그 thread 에 저장한 절을 그대로 쓴다(다시 탐지하면 system 이 바뀔 수 있다)
        env_text = (sessions.env_text_of(args.resume) if args.resume
                    else detect_environment(settings.shell, workspace))
        agent = session_agent(settings, workspace, sessions, env_text, memory_trust=trust)
    except (RuntimeError, ValueError) as e:  # API 키 미설정, 모델 이름 오류, Git Bash 없음 등
        console.print(f"[red]{escape(str(e))}[/]")
        return 2
    if settings.langfuse_enabled and not langfuse_installed():
        console.print("[dim]Langfuse 키가 있지만 패키지가 없어 트레이스를 끈다. "
                      "켜려면: uv sync --extra langfuse[/]")
    # 로컬 트레이스는 Langfuse 와 상관없이 켠다(CODE_AGENT_LOCAL_TRACE=0 으로 끈다).
    # 로컬을 앞에 둔다. 네트워크를 쓰는 Langfuse 보다 먼저 남고, 나중에 닫힌다
    tracer = combine_tracers([LocalTracer(settings.data_dir) if settings.local_trace else None,
                              make_tracer(settings)])
    # 시작에 성공한 뒤에만 세션을 남긴다
    thread_id = args.resume or sessions.create(workspace, env_text)

    def fresh_agent() -> tuple[str, Any]:
        text = detect_environment(settings.shell, workspace)
        return text, session_agent(settings, workspace, sessions, text, memory_trust=trust)

    app = App(settings, workspace, args.mode, sessions, thread_id, console,
              _line_reader(settings.data_dir / "history"), agent=agent, tracer=tracer,
              env_text=env_text, fresh_agent=fresh_agent, memory_trust=trust)
    return app.run()


def session_agent(settings: Settings, workspace: Path, sessions: Sessions, env_text: str, *,
                  memory_trust: MemoryTrust | None = None) -> Any:
    """sessions.db 에 체크포인트와 프롬프트 지문(resume_compat)을 남기는 agent.

    지문 저장소는 그래프 실행 중(모델 호출 직전)에 쓰고, 그때 SqliteSaver 는 앞 단계의 체크포인트를
    백그라운드에서 같은 연결로 저장할 수 있다. 그래서 두 쪽이 같은 잠금을 쓴다.
    memory_trust 는 App 과 나눠 쓰는 프로젝트 AGENTS.md 신뢰다(build_agent).
    """
    saver = SqliteSaver(sessions.conn)
    return build_agent(settings, workspace, saver, env_text=env_text,
                       prompt_store=SqlitePromptStore(sessions.conn, saver.lock),
                       memory_trust=memory_trust)


def detect_environment(shell_preference: str, workspace: Path | None = None) -> str:
    """셸 백엔드가 쓸 환경(process_shell_env)으로 도구를 확인해 '이 PC 의 도구' 절을 만든다.

    약 1초 걸린다.
    """
    shell = resolve_shell(shell_preference, workspace)[0]
    return render(probe(process_shell_env(workspace), workspace=workspace), shell)
