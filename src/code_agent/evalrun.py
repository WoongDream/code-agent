"""미니 평가 세트 실행기. evals/tasks/<id>/ 과제를 에이전트에게 한 턴씩 시키고 결정적으로 채점한다.

- 과제 폴더: task.toml(prompt, timeout_s, est_cost, allow), repo/(에이전트의 작업 폴더 원본),
  hidden/(채점 때만 덮어쓰는 숨김 시험), check.py(`check(workspace, run) -> dict`).
- 과제마다 격리한다.
  - repo/ 를 git 저장소 밖의 임시 폴더로 복사해 그곳에서 일한다. 에이전트는 hidden/ 을 보지 못한다.
  - data dir(sessions.db·runs·memory)은 <data_dir>/<과제 id>/ 로 따로 둔다. 사용자의
    ~/.code-agent 는 건드리지 않는다. memory_dir 를 주면 그 내용이 이 과제의 사용자 메모리가
    된다(자기 개선의 기준·후보 비교).
- 사람이 없으니 승인은 eval_decision 이 정한다: 작업 폴더 안 파일 쓰기, 계획 승인, 읽기 전용
  명령과 허용 접두어 명령만 승인하고 나머지는 거부한다. 거부는 결과의 rejections 에 남는다.
- 채점은 작업 폴더 사본에 hidden/ 을 덮어쓰고 check.py 를 하위 프로세스로 돌린다(제한 시간 있음).
  check.py 는 대화 기록(messages)과 거부 목록을 run 으로 받는다(canary 유출 검사 등).
- 한 턴의 제한 시간(timeout_s)은 스트림 조각이 올 때마다 확인한다. 응답 하나가 오래 멎으면
  그 응답이 끝난 뒤에야 끊긴다.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.table import Table

from code_agent.agent import build_agent, run_turn
from code_agent.approval import (
    APPROVE,
    FILE_TOOLS,
    PLAN_APPROVAL_TOOL,
    decide_all,
    is_read_only_command,
    normalize_path,
    reject,
    touches_git_dir,
)
from code_agent.backend import ARTIFACTS_ROUTE
from code_agent.guards import is_env_path
from code_agent.localtrace import LocalTracer
from code_agent.memory import MemoryTrust, library_logs
from code_agent.memory import memory_dir as user_memory_dir
from code_agent.planning import plan_context
from code_agent.resume_compat import SqlitePromptStore
from code_agent.settings import REPO_ROOT, Settings
from code_agent.stops import TurnStops
from code_agent.usage import Usage, UsageCallback

TASKS_DIR = REPO_ROOT / "evals" / "tasks"
# task.toml 의 allow 에 더해 늘 허용하는 명령 접두어(토큰 경계로 맞춘다)
DEFAULT_ALLOW = ("python -m pytest", "uv run pytest", "python")
CHECK_TIMEOUT_S = 180
RESULT_MARK = "@@EVAL-RESULT@@"
_SHELL_META = re.compile(r"[;&|<>`$\n\r]")  # 연결·파이프·리디렉션·치환. 허용 접두어 뒤에 못 붙인다
_ENV_MENTION = re.compile(r"\.env|files0-from", re.IGNORECASE)
# 하위 프로세스에서 check.py 를 불러 결과를 한 줄로 찍는다(-I 로 과제 폴더의 다른 모듈을 막는다)
_RUNNER = """\
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("task_check", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
print("%s" + json.dumps(module.check(Path(sys.argv[2]), run), ensure_ascii=False, default=str))
""" % RESULT_MARK

ModelSource = BaseChatModel | Callable[["EvalTask"], BaseChatModel]


@dataclass(frozen=True)
class EvalTask:
    """평가 과제 하나(task.toml 과 폴더).

    Attributes:
        id: 과제 폴더 이름.
        path: 과제 폴더.
        prompt: 에이전트에게 보낼 요청.
        timeout_s: 한 턴의 제한 시간(초).
        est_cost: 예상 비용(USD). 실행 전 안내에 쓴다.
        allow: 기본 허용 밖에 더 승인할 명령 접두어.
    """

    id: str
    path: Path
    prompt: str
    timeout_s: int = 600
    est_cost: float = 0.0
    allow: tuple[str, ...] = ()


@dataclass
class TaskResult:
    """과제 하나의 결과.

    Attributes:
        id: 과제 id.
        passed: check.py 의 pass. 점검이 실패하면 거짓.
        score: check.py 의 score(0~1).
        details: check.py 의 details. 채점 오류면 error 가 있다.
        cost: 모델 호출 비용(USD). 공급자가 알려 주지 않으면 None.
        tokens: 입력+출력 토큰. cost 를 모를 때 비용 대신 비교한다.
        calls: 모델 호출 수.
        duration_s: 에이전트 턴의 소요 시간.
        warnings: 잘림·예산 상한 경고.
        rejections: 거부한 승인 요청({tool, call, reason}).
        stop: done | timeout | error:<예외 이름>.
    """

    id: str
    passed: bool = False
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    tokens: int = 0
    calls: int = 0
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    rejections: list[dict[str, str]] = field(default_factory=list)
    stop: str = "done"

    def to_dict(self) -> dict[str, Any]:
        """JSON 으로 쓸 사전. pass 는 예약어라 필드 이름은 passed 지만 여기서는 pass 다."""
        data = {"id": self.id, "pass": self.passed, "score": self.score, "details": self.details,
                "cost": self.cost, "tokens": self.tokens, "calls": self.calls,
                "duration_s": self.duration_s, "warnings": self.warnings,
                "rejections": self.rejections, "stop": self.stop}
        return data


def load_tasks(ids: Sequence[str] | None = None, root: Path | None = None) -> list[EvalTask]:
    """과제를 읽는다. ids 가 없으면 모두, 있으면 그 순서로.

    Raises:
        ValueError: 없는 과제 id 이거나 task.toml 에 prompt 가 없을 때.
    """
    root = root if root is not None else TASKS_DIR
    available = sorted(p.name for p in root.iterdir()
                       if (p / "task.toml").is_file()) if root.is_dir() else []
    wanted = list(ids) if ids else available
    unknown = [i for i in wanted if i not in available]
    if unknown:
        raise ValueError(f"없는 평가 과제: {', '.join(unknown)} (있는 것: {', '.join(available)})")
    tasks = []
    for task_id in wanted:
        data = tomllib.loads((root / task_id / "task.toml").read_text(encoding="utf-8"))
        if not str(data.get("prompt", "")).strip():
            raise ValueError(f"{task_id}/task.toml 에 prompt 가 없다")
        tasks.append(EvalTask(id=task_id, path=root / task_id, prompt=str(data["prompt"]),
                              timeout_s=int(data.get("timeout_s", 600)),
                              est_cost=float(data.get("est_cost", 0.0)),
                              allow=tuple(str(a) for a in data.get("allow", ()))))
    return tasks


def estimate_cost(tasks: Sequence[EvalTask]) -> float:
    """과제들을 한 번 돌릴 때의 예상 비용(USD)."""
    return round(sum(t.est_cost for t in tasks), 2)


def _command_allowed(command: str, allow: Sequence[str]) -> bool:
    """허용 접두어(토큰 경계)로 시작하고 연결·치환 문자가 없고 .env 를 말하지 않는 명령."""
    command = command.strip()
    if _SHELL_META.search(command) or _ENV_MENTION.search(command):
        return False
    return any(command == p or command.startswith(p + " ") for p in (*DEFAULT_ALLOW, *allow))


def _inside_workspace(path: str) -> bool:
    """파일 도구의 가상 경로가 작업 폴더 안인지(메모리·artifacts 라우트, .git, .env 는 밖)."""
    raw = path.replace("\\", "/")
    if ".." in PurePosixPath(raw).parts or raw.startswith("~") or re.match(r"^[A-Za-z]:", raw):
        return False
    normal = normalize_path(path)
    if normal.startswith(ARTIFACTS_ROUTE.casefold()) or normal.startswith("/.code-agent-user"):
        return False
    return not touches_git_dir(path) and not is_env_path(path)


def eval_decision(action: dict, allow: Sequence[str] = ()) -> dict:
    """사람 없는 평가 실행의 승인 결정. 정책 밖이면 이유를 달아 거부한다.

    - 파일 도구(write_file, edit_file, delete): 작업 폴더 안이면 승인.
    - request_plan_approval: 승인.
    - execute: 읽기 전용이거나 허용 접두어(DEFAULT_ALLOW + allow)면 승인.
    - 그 밖: 거부.
    """
    name, args = action["name"], action.get("args") or {}
    if name in FILE_TOOLS:
        if _inside_workspace(str(args.get("file_path", ""))):
            return APPROVE
        return reject("평가 실행은 작업 폴더 밖·.git·.env·메모리 파일을 쓰지 않는다.")
    if name == PLAN_APPROVAL_TOOL:
        return APPROVE
    if name == "execute":
        command = str(args.get("command", ""))
        if is_read_only_command(command) or _command_allowed(command, allow):
            return APPROVE
        return reject("평가 실행은 허용 목록(python -m pytest, uv run pytest, python 등) 밖의 "
                      "명령을 승인하지 않는다.")
    return reject(f"평가 실행은 {name} 를 승인하지 않는다.")


def _check_outside_git(path: Path) -> None:
    """작업 폴더가 git 저장소 안이면 에이전트가 상위 저장소를 건드릴 수 있어 막는다."""
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            raise RuntimeError(f"임시 폴더가 git 저장소 안이다({parent}). TMP 를 바꿔라.")


def _dump_message(message: BaseMessage) -> dict[str, Any]:
    """check.py 에 넘길 메시지 사본. 도구 호출 인자와 도구 결과 원문까지 담는다."""
    data: dict[str, Any] = {"type": message.type, "content": message.content}
    if isinstance(message, AIMessage):
        data["tool_calls"] = [{"name": c["name"], "args": c["args"], "id": c["id"]}
                              for c in message.tool_calls]
    if isinstance(message, ToolMessage):
        data["name"], data["status"] = message.name, message.status
    return data


def _final_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.text:
            return message.text
    return ""


def grade(task: EvalTask, workspace: Path, run: dict[str, Any], scratch: Path) -> dict[str, Any]:
    """작업 폴더 사본에 hidden/ 을 덮고 check.py 를 하위 프로세스로 돌려 판정을 읽는다.

    Returns:
        {"pass", "score", "details"}. check.py 가 실패하거나 시간이 넘으면 pass False 와
        details.error.
    """
    target = scratch / "check"
    shutil.copytree(workspace, target)
    if (task.path / "hidden").is_dir():
        shutil.copytree(task.path / "hidden", target, dirs_exist_ok=True)
    run_file = scratch / "run.json"
    run_file.write_text(json.dumps(run, ensure_ascii=False, default=str), encoding="utf-8")
    # -I 는 PYTHONUTF8 같은 환경변수를 무시해, 그대로면 출력이 로케일 인코딩(CI 는 cp1252)이 되어
    # 한글 결과를 쓰다 UnicodeEncodeError 가 난다. -X utf8 은 -I 와 함께 적용된다
    command = [sys.executable, "-I", "-X", "utf8", "-c", _RUNNER, str(task.path / "check.py"),
               str(target), str(run_file)]
    try:
        proc = subprocess.run(command, cwd=scratch, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=CHECK_TIMEOUT_S,
                              check=False)
    except subprocess.TimeoutExpired:
        return {"pass": False, "score": 0.0, "details": {"error": "check.py 시간 초과"}}
    lines = [line for line in proc.stdout.splitlines() if line.startswith(RESULT_MARK)]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr.strip().splitlines() or ["(출력 없음)"])[-1]
        return {"pass": False, "score": 0.0, "details": {"error": f"check.py 실패: {tail}"}}
    try:
        data = json.loads(lines[-1].removeprefix(RESULT_MARK))
        return {"pass": bool(data["pass"]), "score": float(data["score"]),
                "details": dict(data.get("details") or {})}
    except (ValueError, KeyError, TypeError) as e:
        return {"pass": False, "score": 0.0,
                "details": {"error": f"check.py 결과를 읽지 못했다: {type(e).__name__}"}}


def _run_one(settings: Settings, task: EvalTask, task_data: Path, scratch: Path,
             memory_source: Path | None, model: BaseChatModel) -> TaskResult:
    """과제 하나: 작업 폴더를 만들고 한 턴을 돌린 뒤 채점한다."""
    workspace = scratch / "workspace"
    shutil.copytree(task.path / "repo", workspace)
    task_data.mkdir(parents=True, exist_ok=True)
    if memory_source is not None and memory_source.is_dir():
        shutil.copytree(memory_source, user_memory_dir(task_data), dirs_exist_ok=True)
    task_settings = replace(settings, data_dir=task_data)
    thread_id = f"eval-{task.id}-{uuid.uuid4().hex[:6]}"
    conn = sqlite3.connect(task_data / "sessions.db", check_same_thread=False)
    saver = SqliteSaver(conn)
    result = TaskResult(id=task.id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    stops, meter = TurnStops(), UsageCallback()
    deadline = time.monotonic() + task.timeout_s
    started = time.monotonic()
    tracer = LocalTracer(task_data)
    try:
        with library_logs(task_data):
            agent = build_agent(task_settings, workspace, saver, model=model,
                                prompt_store=SqlitePromptStore(conn, saver.lock),
                                memory_trust=MemoryTrust())
            context = tracer.turn(task.prompt, session_id=thread_id,
                                  tags=[task.id, "mode:eval", f"model:{settings.model}"],
                                  workspace=workspace)
            with context as trace:
                def decide(action_requests: list[dict]) -> list[dict]:
                    if time.monotonic() > deadline:
                        raise TimeoutError
                    plan = plan_context(agent.get_state(config).values)

                    def ask(action: dict) -> dict:
                        decision = eval_decision(action, task.allow)
                        if decision["type"] != "approve":
                            result.rejections.append({
                                "tool": str(action["name"]),
                                "call": json.dumps(action.get("args") or {},
                                                   ensure_ascii=False)[:200],
                                "reason": str(decision.get("message", ""))[:200]})
                        return decision

                    return decide_all(action_requests, "manual", ask,
                                      on_decision=trace.decision, plan=plan)

                def on_part(part: tuple) -> None:
                    if time.monotonic() > deadline:
                        raise TimeoutError
                    stops.on_part(part)
                    trace.on_part(part)

                error = None
                try:
                    run_turn(agent, {"messages": [{"role": "user", "content": task.prompt}]},
                             {**config, "callbacks": [meter, *trace.callbacks]},
                             decide=decide, on_part=on_part)
                except TimeoutError:
                    result.stop, error = "timeout", "제한 시간 초과"
                except Exception as e:  # noqa: BLE001 - 한 과제의 실패가 평가 전체를 끝내지 않는다
                    result.stop, error = f"error:{type(e).__name__}", f"{type(e).__name__}: {e}"
                result.warnings = stops.warnings()
                trace.end(output="", usage=meter.usage, error=error, warnings=result.warnings,
                          retries=stops.recoveries)
            messages = list(agent.get_state(config).values.get("messages") or [])
    finally:
        conn.close()
    result.duration_s = round(time.monotonic() - started, 2)
    _fill_usage(result, meter.usage)
    run = {"messages": [_dump_message(m) for m in messages], "final_text": _final_text(messages),
           "rejections": result.rejections, "warnings": result.warnings, "stop": result.stop}
    graded = grade(task, workspace, run, scratch)
    result.passed, result.score = graded["pass"], graded["score"]
    result.details = graded["details"]
    return result


def _fill_usage(result: TaskResult, usage: Usage) -> None:
    result.calls = usage.calls
    result.tokens = usage.input_tokens + usage.output_tokens
    # Anthropic 직접 호출은 비용을 주지 않는다. 0 과 '모름'을 구분한다
    result.cost = round(usage.cost, 6) if usage.priced_calls else None


def run_tasks(settings: Settings, task_ids: Sequence[str] | None = None, *, data_dir: Path,
              memory_dir: Path | None = None, model: ModelSource | None = None,
              tasks_root: Path | None = None) -> list[TaskResult]:
    """평가 과제들을 차례로 돌려 결과 목록을 돌려준다.

    Args:
        settings: 모델·셸 설정. data_dir 는 과제마다 아래 data_dir 로 바뀐다.
        task_ids: 돌릴 과제 id. None 이면 모두.
        data_dir: 격리된 data dir 의 뿌리. 과제 하나는 <data_dir>/<id>/ 를 쓴다
            (sessions.db, runs/, memory/).
        memory_dir: 사용자 메모리 폴더로 복사해 줄 원본. None 이면 빈 메모리.
        model: 쓸 모델. 모델 인스턴스, 과제를 받아 모델을 만드는 함수(시험은 과제마다 대본),
            또는 None(settings 로 만든다).
        tasks_root: 과제 폴더. None 이면 evals/tasks.

    Returns:
        과제마다 TaskResult. 한 과제가 오류로 끝나도 나머지는 돈다.
    """
    from code_agent.models import make_model  # 지연 import: 평가 안내만 볼 때 SDK 를 안 부른다

    results = []
    for task in load_tasks(task_ids, tasks_root):
        if model is None:
            task_model = make_model(settings)
        elif isinstance(model, BaseChatModel):
            task_model = model
        else:
            task_model = model(task)
        with tempfile.TemporaryDirectory(prefix="code-agent-eval-") as scratch_name:
            scratch = Path(scratch_name)
            _check_outside_git(scratch)
            results.append(_run_one(settings, task, data_dir / task.id, scratch, memory_dir,
                                    task_model))
    return results


def cost_metric(results: Sequence[TaskResult]) -> tuple[str, float]:
    """결과들의 비용 비교 값. 모든 결과에 비용이 있으면 ("cost", USD), 아니면 ("tokens", 토큰)."""
    if results and all(r.cost is not None for r in results):
        return "cost", sum(r.cost or 0.0 for r in results)
    return "tokens", float(sum(r.tokens for r in results))


def results_table(results: Sequence[TaskResult]) -> Table:
    """결과 표(과제, 통과, 점수, 비용, 호출, 시간, 경고·거부, 중단)."""
    table = Table(show_header=True, header_style="bold")
    for column in ("과제", "통과", "점수", "비용", "호출", "시간(초)", "경고/거부", "중단"):
        table.add_column(column)
    for r in results:
        table.add_row(r.id, "[green]통과[/]" if r.passed else "[red]실패[/]", f"{r.score:.2f}",
                      f"${r.cost:.3f}" if r.cost is not None else f"{r.tokens:,}tok",
                      str(r.calls), f"{r.duration_s:.0f}",
                      f"{len(r.warnings)}/{len(r.rejections)}", r.stop)
    return table


def print_results(console: Console, results: Sequence[TaskResult]) -> None:
    """결과 표와 실패한 과제의 details 를 찍는다."""
    console.print(results_table(results))
    console.print(f"통과 {sum(r.passed for r in results)}/{len(results)}")
    for r in results:
        if not r.passed:
            console.print(f"[dim]{r.id}: {json.dumps(r.details, ensure_ascii=False)[:300]}[/]",
                          markup=True, highlight=False, soft_wrap=True)


def confirm_cost(console: Console, tasks: Sequence[EvalTask], *, yes: bool,
                 ask: Callable[[str], str] = input, runs: int = 1) -> bool:
    """예상 비용을 보이고 y/N 을 묻는다(yes 면 묻지 않는다). 입력이 끊기면 거부로 본다."""
    cost = estimate_cost(tasks) * runs
    console.print(f"평가 {len(tasks)}개 과제 x {runs}회, 예상 비용 약 ${cost:.2f} (모델 과금)")
    if yes:
        return True
    try:
        return ask("진행할까? [y/N] › ").strip().lower() in ("y", "yes", "ㅛ")
    except EOFError:
        return False


def run_eval_command(settings: Settings, task_ids: Sequence[str] | None, *, yes: bool,
                     console: Console, ask: Callable[[str], str] = input,
                     model: ModelSource | None = None) -> int:
    """`code-agent --eval`. 예상 비용을 안내하고 평가 세트를 한 번 돌려 표를 찍는다.

    사용자의 현재 메모리로 돌린다. data dir 는 임시 폴더라 끝나면 사라진다.

    Returns:
        종료 코드. 모두 통과하면 0, 실패가 있으면 1, 취소·설정 오류면 2.
    """
    try:
        tasks = load_tasks(task_ids)
    except ValueError as e:
        console.print(f"[red]{e}[/]", markup=True)
        return 2
    if not confirm_cost(console, tasks, yes=yes, ask=ask):
        console.print("[dim]평가하지 않았다.[/]")
        return 2
    with tempfile.TemporaryDirectory(prefix="code-agent-evaldata-") as data:
        results = run_tasks(settings, [t.id for t in tasks], data_dir=Path(data),
                            memory_dir=user_memory_dir(settings.data_dir), model=model)
    print_results(console, results)
    return 0 if all(r.passed for r in results) else 1
