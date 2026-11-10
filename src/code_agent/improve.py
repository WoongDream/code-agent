"""자기 개선 루프(/improve, code-agent --improve).

흐름: 최근 턴의 로컬 트레이스에서 신호를 모으고(collect_signals) → 개선 모델이 메모리·스킬 개선안을
만들고(propose) → 정적 검사(static_check)와 평가 세트 기준·후보 비교(evaluate)를 거쳐 →
사람이 diff 를 보고 승인하면 쓴다(apply_proposal). 평가를 못 하는 개선안(프로젝트 대상,
--no-eval)은 사람의 승인만으로 반영하고, --yes 로는 자동 반영하지 않는다.

- 개선안은 메모리(AGENTS.md)와 스킬(SKILL.md)만 고친다. 코드·설정은 건드리지 않는다.
  경로는 target·skill_name 으로만 정해 화이트리스트 밖을 쓸 수 없다(target_path).
- 개선 모델은 JSON 배열을 ```json 블록으로 답한다. with_structured_output 은 쓰지 않는다
  (강제 도구 선택이 OpenRouter Opus 의 thinking 과 충돌한다).
- 모든 단계는 improve_signals·improve_proposal·improve_eval·improve_decision 사건으로
  <data_dir>/runs/improve-<id>/trace.jsonl 에, 개선안·평가·결정은 <data_dir>/improvements.jsonl 에
  남는다. Langfuse 트레이서가 있으면 같은 사건을 보낸다(실패해도 개선은 이어 간다).
- 신호 수집의 Langfuse Judge 점수는 키가 있을 때만 쓰는 보조 신호다. 어떤 오류든 조용히 넘긴다.
"""

from __future__ import annotations

import base64
import difflib
import json
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from rich.console import Console
from rich.markup import escape

from code_agent.evalrun import (
    EvalTask,
    ModelSource,
    TaskResult,
    cost_metric,
    estimate_cost,
    load_tasks,
    results_table,
    run_tasks,
)
from code_agent.localtrace import METRICS_FILE, RUNS_DIR, TRACE_FILE, append_jsonl, read_jsonl
from code_agent.memory import (
    MEMORY_FILE,
    MEMORY_SOURCES,
    PROJECT_SKILLS,
    USER_SKILLS,
    decode_memory,
    disk_path,
    memory_dir,
    read_memory,
)
from code_agent.settings import Settings
from code_agent.tracing import Tracer
from code_agent.usage import Usage

IMPROVEMENTS_FILE = "improvements.jsonl"
SKILL_FILE = "SKILL.md"
MAX_CHARS = 8_000  # AGENTS.md 합계·SKILL.md 상한(memory.WARN_CHARS 와 같다)
MAX_PROPOSALS = 3
COST_LIMIT = 1.3  # 후보 비용이 기준의 이 배수를 넘으면 거부
LOW_SCORE = 0.5  # Langfuse Judge 점수가 이보다 낮으면 신호
TARGETS = ("user_memory", "project_memory", "user_skill", "project_skill")
EVALUATED_TARGETS = ("user_memory", "user_skill")  # 평가 세트가 읽는 것은 사용자 메모리뿐이다
NOT_EVALUATED = "평가 대상 아님: 평가 과제는 프로젝트 파일을 못 본다. 사람이 직접 승인해야 한다."
_SKILL_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_BULLET = re.compile(r"^\s*[-*]\s+")
_PYTEST = re.compile(r"\bpytest\b")
_SECRETS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "API 키 모양(sk-…)"),
    (re.compile(r"\b[A-Za-z][A-Za-z0-9_]*_(?:KEY|TOKEN|SECRET|PASSWORD)\s*[=:]\s*\S", re.I),
     "비밀값 대입 모양(*_KEY=…)"),
    (re.compile(r"CANARY_[0-9A-Za-z]+"), "canary 값"),
    (re.compile(r"\b(?:ghp|gho|xox[abp]|AKIA)[A-Za-z0-9_-]{12,}"), "토큰 모양"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "개인 키"),
)

Ask = Callable[[str], bool]  # y/N 을 묻는 함수


# --- 신호 ---

@dataclass(frozen=True)
class Signal:
    """개선 근거가 되는 신호 하나.

    Attributes:
        kind: warning | retry | rejection | repeated_error | failed_tests | plan_review |
            judge_low.
        thread_id: 신호가 난 세션.
        turn: 그 세션 안의 턴 번호(모르면 0).
        text: 짧은 증거 글.
    """

    kind: str
    thread_id: str
    turn: int
    text: str

    @property
    def evidence(self) -> str:
        """근거로 인용할 열쇠(thread:turn)."""
        return f"{self.thread_id}:{self.turn}"


def _short(text: Any, limit: int = 160) -> str:
    return " ".join(str(text).split())[:limit]


def _turn_signals(thread: str, turn: int, metrics: Mapping[str, Any],
                  events: Sequence[Mapping[str, Any]]) -> list[Signal]:
    """한 턴의 trace 사건과 지표에서 신호를 뽑는다."""
    found: list[Signal] = []

    def add(kind: str, text: Any) -> None:
        found.append(Signal(kind, thread, turn, _short(text)))

    for event in events:
        kind = event.get("type")
        if kind == "warning":
            add("warning", event.get("message", ""))
        elif kind == "error":
            add("warning", f"오류: {event.get('message', '')}")
        elif kind == "approval" and event.get("decision") == "reject" \
                and event.get("by") == "user":
            add("rejection", f"{event.get('call', '')} / 이유: {event.get('reason') or '(없음)'}")
        elif kind == "plan_reviewed":
            high = [f for f in event.get("findings") or []
                    if isinstance(f, Mapping) and f.get("severity") == "high"]
            if high:
                add("plan_review", "; ".join(str(f.get("issue", "")) for f in high))
    if int(metrics.get("retries") or 0) > 0:
        add("retry", f"잘린 응답 자동 복구 {metrics.get('retries')}회")
    tool_events = [e for e in events if e.get("type") == "tool"]
    errors: dict[tuple[str, str, str], int] = {}
    for event in tool_events:
        if event.get("status") == "error":
            key = (str(event.get("name")), str(event.get("call")), str(event.get("error")))
            errors[key] = errors.get(key, 0) + 1
    for (name, call, error), count in errors.items():
        if count >= 2:
            add("repeated_error", f"{name} {call} 이 {count}번 같은 오류: {error}")
    last_failed: Mapping[str, Any] | None = None
    for event in tool_events:  # 통과한 pytest 가 뒤따르면 실패는 해결된 것이다
        if event.get("name") == "execute" and _PYTEST.search(str(event.get("call", ""))):
            code = event.get("exit_code")
            if code not in (None, 0):
                last_failed = event
            elif code == 0:
                last_failed = None
    if last_failed is not None:
        add("failed_tests", f"{last_failed.get('call')} exit {last_failed.get('exit_code')}: "
                            f"{last_failed.get('error') or ''}")
    return found


def judge_signals(settings: Settings,
                  fetch: Callable[[str, dict[str, str]], Any] | None = None) -> list[Signal]:
    """Langfuse 의 낮은 Judge 점수 관측을 신호로 돌려준다. 키가 없거나 실패하면 빈 목록.

    GET {base}/api/public/v2/observations 의 data 항목 중 scores 에 LOW_SCORE 미만 숫자가 있는
    것을 본다. 응답 모양은 서버 버전마다 다를 수 있어 모르는 모양은 건너뛴다.
    """
    if not settings.langfuse_enabled:
        return []
    url = (settings.langfuse_base_url.rstrip("/") + "/api/public/v2/observations?"
           + urllib.parse.urlencode({"limit": 50, "fields": "core,scores"}))
    token = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()).decode()
    headers = {"Authorization": f"Basic {token}"}
    try:
        if fetch is None:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - https
                body = json.loads(response.read().decode("utf-8"))
        else:
            body = fetch(url, headers)
        signals = []
        for item in body.get("data") or []:
            for score in item.get("scores") or []:
                value = score.get("value")
                if isinstance(value, int | float) and value < LOW_SCORE:
                    thread = str(item.get("sessionId") or item.get("traceId") or "langfuse")
                    signals.append(Signal("judge_low", thread, 0,
                                          _short(f"{score.get('name', 'judge')}={value}")))
        return signals
    except Exception:  # noqa: BLE001 - 보조 신호라 어떤 실패든 조용히 넘긴다
        return []


def collect_signals(data_dir: Path, limit_turns: int = 20, *,
                    settings: Settings | None = None,
                    fetch: Callable[[str, dict[str, str]], Any] | None = None) -> list[Signal]:
    """최근 limit_turns 개 턴의 로컬 트레이스에서 신호를 모은다.

    <data_dir>/runs/*/metrics.jsonl 의 턴을 시작 시각순으로 최근 것부터 고르고, 같은 폴더의
    trace.jsonl 사건과 합쳐 본다. improve- 로 시작하는 폴더(개선 실행 자신)와 eval- 세션은
    건너뛴다.

    Args:
        data_dir: 사용자 데이터 폴더.
        limit_turns: 볼 최근 턴 수.
        settings: 있으면 Langfuse 키로 낮은 Judge 점수도 본다(judge_signals).
        fetch: judge_signals 의 HTTP 대역(시험용).

    Returns:
        신호 목록(오래된 턴부터). 각 신호에 thread_id·turn·짧은 증거가 있다.
    """
    turns: list[tuple[str, str, int, Mapping[str, Any], Path]] = []
    runs = data_dir / RUNS_DIR
    folders = sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []
    for folder in folders:
        if folder.name.startswith(("improve-", "eval-")):
            continue
        for row in read_jsonl(folder / METRICS_FILE):
            number = row.get("turn")
            if isinstance(number, int) and not isinstance(number, bool):
                turns.append((str(row.get("started_at") or ""), str(row.get("thread_id")
                                                                      or folder.name),
                              number, row, folder))
    turns.sort(key=lambda t: (t[0], t[2]))
    signals: list[Signal] = []
    cache: dict[Path, list[dict[str, Any]]] = {}
    for _, thread, number, metrics, folder in turns[-limit_turns:] if limit_turns > 0 else []:
        events = cache.setdefault(folder, read_jsonl(folder / TRACE_FILE))
        signals += _turn_signals(thread, number, metrics,
                                 [e for e in events if e.get("turn") == number])
    if settings is not None:
        signals += judge_signals(settings, fetch)
    return signals


# --- 개선안 ---

class Proposal(BaseModel):
    """개선안 하나.

    Attributes:
        target: user_memory(사용자 AGENTS.md) | project_memory(작업 폴더 AGENTS.md) |
            user_skill | project_skill.
        skill_name: 스킬 대상이면 스킬 폴더 이름(소문자·숫자·하이픈).
        patch: 메모리 대상이면 덧붙일 항목(줄 앞 '- '), 스킬 대상이면 SKILL.md 전문.
        evidence: 근거(thread:turn).
        expected_effect: 기대 효과 한 줄.
    """

    target: Literal["user_memory", "project_memory", "user_skill", "project_skill"]
    skill_name: str | None = None
    patch: str
    evidence: list[str] = Field(default_factory=list)
    expected_effect: str = ""

    @field_validator("patch")
    @classmethod
    def _patch_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("patch 가 비었다")
        return value

    @model_validator(mode="after")
    def _skill_name_rules(self) -> Proposal:
        is_skill = self.target.endswith("_skill")
        if is_skill and not (self.skill_name and _SKILL_NAME.fullmatch(self.skill_name)
                             and len(self.skill_name) <= 64):
            raise ValueError("스킬 대상은 skill_name(소문자·숫자·하이픈, 64자 이하)이 있어야 한다")
        if not is_skill and self.skill_name:
            raise ValueError("메모리 대상은 skill_name 을 쓰지 않는다")
        return self


def memory_snapshot(data_dir: Path, workspace: Path) -> dict[str, str]:
    """현재 메모리·스킬 내용. 키는 가상 경로(/.code-agent-user/AGENTS.md, /AGENTS.md,
    …/skills/<이름>/SKILL.md)다. 없는 파일은 넣지 않는다.
    """
    snapshot: dict[str, str] = {}
    for source in MEMORY_SOURCES:
        if (data := read_memory(disk_path(source, workspace=workspace, data_dir=data_dir))):
            snapshot[source] = decode_memory(data)
    for source in (USER_SKILLS, PROJECT_SKILLS):
        root = disk_path(source, workspace=workspace, data_dir=data_dir)
        for skill in sorted(root.glob(f"*/{SKILL_FILE}")) if root.is_dir() else []:
            if (data := read_memory(skill)):
                snapshot[f"{source}{skill.parent.name}/{SKILL_FILE}"] = decode_memory(data)
    return snapshot


PROPOSE_SYSTEM = """\
너는 코딩 에이전트(code-agent)의 메모리와 스킬을 개선하는 편집자다. 아래 신호(최근 세션에서 \
관찰된 문제)와 현재 메모리를 보고, 같은 문제를 줄일 개선안을 최대 3개 만든다.

규칙
- 개선안은 메모리 항목이나 스킬만이다. 코드·설정은 고치지 않는다.
- target: user_memory(모든 프로젝트 공통 AGENTS.md) | project_memory(이 프로젝트 AGENTS.md) \
| user_skill | project_skill.
- 메모리 patch 는 덧붙일 짧은 항목들이다(줄마다 '- '). 이미 있는 항목은 되풀이하지 않는다.
- 스킬 patch 는 SKILL.md 전문이다. 맨 앞에 YAML frontmatter(name 은 skill_name 과 같게, \
description 은 언제 쓰는지 한 줄)를 둔다. skill_name 은 소문자·숫자·하이픈이다.
- 비밀값(키, 토큰, .env 내용)과 일회성 사실은 쓰지 않는다. 신호에서 일반 규칙을 뽑을 수 없으면 \
빈 배열 []을 답한다.
- evidence 는 신호의 'thread:turn' 열쇠 목록이다.

답은 아래 모양의 JSON 배열 하나를 ```json 코드 블록에 담아서만 한다.
```json
[{"target": "user_memory", "skill_name": null, "patch": "- 항목", \
"evidence": ["thread:3"], "expected_effect": "기대 효과 한 줄"}]
```"""


def _propose_prompt(signals: Sequence[Signal], snapshot: Mapping[str, str]) -> str:
    lines = [f"- [{s.kind}] {s.evidence} {s.text}" for s in signals] or ["(신호 없음)"]
    memory = [f"### {path}\n{text}" for path, text in snapshot.items()] or ["(메모리·스킬 없음)"]
    return "## 신호\n" + "\n".join(lines) + "\n\n## 현재 메모리와 스킬\n" + "\n\n".join(memory)


def parse_proposals(text: str) -> list[Proposal]:
    """모델 답의 ```json 블록(없으면 답 전체)을 개선안 목록으로 읽는다.

    Raises:
        ValueError: JSON 이 아니거나 배열이 아니거나 항목이 스키마(Proposal)에 안 맞을 때.
    """
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    raw = (match.group(1) if match else text).strip()
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"JSON 을 읽지 못했다: {e}") from e
    if not isinstance(data, list):
        raise ValueError("최상위가 배열이 아니다")
    try:
        return [Proposal.model_validate(item) for item in data]
    except ValidationError as e:
        raise ValueError(f"개선안이 스키마에 안 맞는다: {e.errors()[0]['msg']}") from e


def propose(signals: Sequence[Signal], snapshot: Mapping[str, str],
            model: BaseChatModel) -> list[Proposal]:
    """개선 모델에게 개선안을 받는다. 읽지 못하면 한 번 다시 묻고, 또 실패하면 빈 목록이다.

    Args:
        signals: collect_signals 의 신호.
        snapshot: 현재 메모리·스킬(memory_snapshot).
        model: 개선 모델.

    Returns:
        검증된 개선안(최대 MAX_PROPOSALS).
    """
    messages: list[Any] = [SystemMessage(PROPOSE_SYSTEM),
                           HumanMessage(_propose_prompt(signals, snapshot))]
    for attempt in range(2):
        reply = model.invoke(messages)
        try:
            return parse_proposals(reply.text)[:MAX_PROPOSALS]
        except ValueError as e:
            if attempt == 1:
                return []
            messages += [reply, HumanMessage(
                f"답을 읽지 못했다({e}). 지시한 모양의 JSON 배열 하나만 ```json 블록에 담아 "
                "다시 답하라.")]
    return []


# --- 경로·적용 ---

def target_path(proposal: Proposal, *, memory_root: Path, workspace: Path) -> Path:
    """개선안이 쓸 파일. target 과 검증된 skill_name 으로만 정해 화이트리스트 밖으로 못 나간다.

    Args:
        proposal: 개선안.
        memory_root: 사용자 메모리 폴더(<data_dir>/memory).
        workspace: 작업 폴더(프로젝트 대상).
    """
    name = proposal.skill_name or ""
    match proposal.target:
        case "user_memory":
            return memory_root / MEMORY_FILE
        case "project_memory":
            return workspace / MEMORY_FILE
        case "user_skill":
            return memory_root / "skills" / name / SKILL_FILE
        case _:
            return workspace / ".agents" / "skills" / name / SKILL_FILE


def new_text(proposal: Proposal, old: str) -> str:
    """개선안을 적용한 파일 내용. 메모리는 끝에 덧붙이고 스킬은 patch 전문이다."""
    if proposal.target.endswith("_skill"):
        return proposal.patch.strip("\n") + "\n"
    patch = proposal.patch.strip("\n") + "\n"
    return old.rstrip("\n") + "\n" + patch if old.strip() else patch


def apply_patch(proposal: Proposal, *, memory_root: Path, workspace: Path) -> Path:
    """개선안을 파일에 쓴다(UTF-8, BOM 없음, LF). 승인 확인은 부르는 쪽 책임이다.

    Returns:
        쓴 파일 경로.
    """
    path = target_path(proposal, memory_root=memory_root, workspace=workspace)
    old = decode_memory(data) if (data := read_memory(path)) is not None else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(new_text(proposal, old))
    return path


def render_diff(proposal: Proposal, *, memory_root: Path, workspace: Path) -> str:
    """승인 전에 보일 unified diff."""
    path = target_path(proposal, memory_root=memory_root, workspace=workspace)
    old = decode_memory(data) if (data := read_memory(path)) is not None else ""
    diff = difflib.unified_diff(old.splitlines(), new_text(proposal, old).splitlines(),
                                f"{path.name} (현재)", f"{path.name} (개선안)", lineterm="")
    return "\n".join(diff)


# --- 정적 검사 ---

def _bullets(text: str) -> set[str]:
    return {" ".join(_BULLET.sub("", line).split()).casefold()
            for line in text.splitlines() if _BULLET.match(line)}


def static_check(proposal: Proposal, snapshot: Mapping[str, str] | None = None) -> list[str]:
    """정적 검사. 문제 목록을 돌려주고 비어 있으면 통과다.

    - 비밀값 모양(sk-…, *_KEY=…, canary 등)이 patch 에 없다.
    - 메모리는 AGENTS.md 합계(현재 사용자·프로젝트 메모리 + patch)가 MAX_CHARS 이하,
      같은 항목(bullet)이 이미 없거나 patch 안에서 겹치지 않는다.
    - 스킬은 BOM 이 없고 frontmatter(name 이 skill_name 과 같음, description)가 맞고
      MAX_CHARS 이하다.
    - 쓸 경로는 target·skill_name 으로 정해지는 화이트리스트(target_path)뿐이다.

    Args:
        proposal: 개선안.
        snapshot: 현재 메모리·스킬(memory_snapshot). None 이면 비어 있다고 본다.
    """
    snapshot = snapshot or {}
    problems = [f"비밀값 의심: {label}" for pattern, label in _SECRETS
                if pattern.search(proposal.patch)]
    patch = proposal.patch
    if patch.startswith("﻿"):
        problems.append("BOM 으로 시작한다")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in patch):
        problems.append("제어 문자가 있다")
    if proposal.target.endswith("_skill"):
        problems += _skill_problems(proposal)
    else:
        source = MEMORY_SOURCES[0] if proposal.target == "user_memory" else MEMORY_SOURCES[1]
        total = sum(len(snapshot.get(s, "")) for s in MEMORY_SOURCES) + len(patch)
        if total > MAX_CHARS:
            problems.append(f"AGENTS.md 합계가 {total:,}자로 {MAX_CHARS:,}자를 넘는다")
        existing = _bullets(snapshot.get(source, ""))
        seen: set[str] = set()
        for line in patch.splitlines():
            if _BULLET.match(line):
                key = " ".join(_BULLET.sub("", line).split()).casefold()
                if key in existing or key in seen:
                    problems.append(f"중복 항목: {_short(line, 60)}")
                seen.add(key)
    return problems


def _skill_problems(proposal: Proposal) -> list[str]:
    patch = proposal.patch
    problems = []
    if len(patch) > MAX_CHARS:
        problems.append(f"SKILL.md 가 {len(patch):,}자로 {MAX_CHARS:,}자를 넘는다")
    match = _FRONTMATTER.match(patch)
    if match is None:
        return problems + ["SKILL.md 맨 앞에 YAML frontmatter(---)가 없다"]
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return problems + ["frontmatter 가 YAML 이 아니다"]
    if not isinstance(meta, dict):
        return problems + ["frontmatter 가 name·description 사전이 아니다"]
    if meta.get("name") != proposal.skill_name:
        problems.append(f"frontmatter name({meta.get('name')!r}) 이 skill_name 과 다르다")
    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        problems.append("frontmatter description 이 없다")
    elif len(description) > 1024:
        problems.append("frontmatter description 이 1024자를 넘는다")
    return problems


# --- 평가 ---

@dataclass
class Evaluation:
    """개선안의 평가 결과.

    Attributes:
        evaluated: 평가 세트를 돌렸는지(--no-eval·프로젝트 대상이면 False).
        accepted: 채택 규칙을 통과했는지. evaluated 가 False 면 항상 False 다.
        reason: 채택·거부 이유.
        baseline: 기준(현재 메모리)의 과제별 결과(TaskResult.to_dict).
        candidate: 후보(개선안 적용)의 과제별 결과.
    """

    evaluated: bool = False
    accepted: bool = False
    reason: str = ""
    baseline: list[dict[str, Any]] = field(default_factory=list)
    candidate: list[dict[str, Any]] = field(default_factory=list)


def accept_rule(baseline: Sequence[TaskResult],
                candidate: Sequence[TaskResult]) -> tuple[bool, str]:
    """채택 규칙: 후보의 통과 수가 기준 이상이고, 통과→실패로 바뀐 과제가 없고,
    비용이 기준의 COST_LIMIT 배 이하(비용을 모르면 토큰 수)여야 채택한다.

    Returns:
        (채택 여부, 이유).
    """
    base_pass, cand_pass = sum(r.passed for r in baseline), sum(r.passed for r in candidate)
    if cand_pass < base_pass:
        return False, f"통과 수가 줄었다({base_pass} → {cand_pass})"
    after = {r.id: r.passed for r in candidate}
    flipped = [r.id for r in baseline if r.passed and not after.get(r.id, False)]
    if flipped:
        return False, f"통과에서 실패로 바뀐 과제: {', '.join(flipped)}"
    unit, base_value = cost_metric(baseline)
    unit2, cand_value = cost_metric(candidate)
    if unit != unit2:  # 한쪽만 비용을 알면 토큰으로 맞춘다
        unit = "tokens"
        base_value, cand_value = (float(sum(r.tokens for r in rs)) for rs in (baseline, candidate))
    if cand_value > base_value * COST_LIMIT:
        return False, (f"{unit} 이 기준의 {COST_LIMIT:.1f}배를 넘었다"
                       f"({base_value:g} → {cand_value:g})")
    return True, f"통과 {base_pass} → {cand_pass}, {unit} {base_value:g} → {cand_value:g}"


def run_arm(arm: str, settings: Settings, tasks: Sequence[EvalTask], *, data_dir: Path,
            memory_source: Path, models: Callable[[str, EvalTask], BaseChatModel] | None
            ) -> list[TaskResult]:
    """평가 arm(baseline·candidate) 하나를 임시 data dir 에서 돌린다."""
    model: ModelSource | None = None if models is None else (lambda t: models(arm, t))
    return run_tasks(settings, [t.id for t in tasks], data_dir=data_dir / arm,
                     memory_dir=memory_source, model=model)


def evaluate(proposal: Proposal, settings: Settings, tasks: Sequence[EvalTask], *,
             workspace: Path | None = None, baseline: Sequence[TaskResult] | None = None,
             models: Callable[[str, EvalTask], BaseChatModel] | None = None) -> Evaluation:
    """평가 세트를 기준(현재 메모리)과 후보(개선안 적용)로 각각 돌려 채택 규칙으로 가른다.

    arm 마다 임시 data dir 와 메모리 사본에서 돌아 사용자 폴더는 바뀌지 않는다.
    프로젝트 대상 개선안은 평가하지 않고 "평가 대상 아님"으로 둔다(사람 승인 필요).

    Args:
        proposal: 개선안.
        settings: 설정. settings.data_dir 의 memory 가 현재(기준) 메모리다.
        tasks: 돌릴 과제.
        workspace: 작업 폴더. 평가하는 개선안은 사용자 대상뿐이라 없어도 된다.
        baseline: 이미 돌린 기준 결과. 주면 기준을 다시 돌리지 않는다(개선안이 여러 개일 때).
        models: (arm, 과제) → 모델. None 이면 settings 로 만든다. 시험은 대본을 넣는다.

    Returns:
        Evaluation.
    """
    if proposal.target not in EVALUATED_TARGETS:
        return Evaluation(reason=NOT_EVALUATED)
    current = memory_dir(settings.data_dir)
    with tempfile.TemporaryDirectory(prefix="code-agent-improve-") as base_name:
        base = Path(base_name)
        base_results = list(baseline) if baseline is not None else run_arm(
            "baseline", settings, tasks, data_dir=base, memory_source=current, models=models)
        candidate_memory = base / "candidate-memory"
        if current.is_dir():
            shutil.copytree(current, candidate_memory)
        candidate_memory.mkdir(exist_ok=True)
        apply_patch(proposal, memory_root=candidate_memory, workspace=workspace or Path.cwd())
        cand_results = run_arm("candidate", settings, tasks, data_dir=base,
                               memory_source=candidate_memory, models=models)
    accepted, reason = accept_rule(base_results, cand_results)
    return Evaluation(evaluated=True, accepted=accepted, reason=reason,
                      baseline=[r.to_dict() for r in base_results],
                      candidate=[r.to_dict() for r in cand_results])


# --- 기록 ---

def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ImproveTrace:
    """개선 실행의 사건 기록. 그래프 밖이라 get_stream_writer 대신 파일에 직접 덧붙인다.

    <data_dir>/runs/improve-<id>/trace.jsonl 에 localtrace 와 같은 모양(ts, turn, type, t)으로
    쓴다. tracer(Langfuse 등)가 있으면 같은 사건을 보내고, 실패해도 무시한다.
    """

    def __init__(self, data_dir: Path, tracer: Tracer | None = None) -> None:
        self.run_id = f"improve-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        self.folder = data_dir / RUNS_DIR / self.run_id
        self.started = time.perf_counter()
        self._cm: AbstractContextManager[Any] | None = None
        self._turn: Any = None
        if tracer is not None:
            try:
                manager = tracer.turn("improve", session_id=self.run_id, tags=["improve"],
                                      workspace=None, context="")
                self._turn = manager.__enter__()
                self._cm = manager
            except Exception:  # noqa: BLE001 - 보조 기록은 개선을 막지 않는다
                self._cm = self._turn = None

    def emit(self, kind: str, **fields: Any) -> None:
        """사건 한 줄. 파일 쓰기와 트레이서 전송은 실패해도 삼킨다."""
        event = {"ts": _now(), "turn": 0, "type": kind,
                 "t": round(time.perf_counter() - self.started, 3), **fields}
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.folder / TRACE_FILE, event)
        except (OSError, TypeError, ValueError):
            pass
        if self._turn is not None:
            try:
                self._turn.on_part(((), "custom", {"type": kind, **fields}))
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        """트레이서 턴을 닫는다."""
        if self._cm is not None:
            try:
                self._turn.end(output="", usage=Usage())
                self._cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._cm = None


@dataclass
class Outcome:
    """개선안 하나의 처리 결과.

    Attributes:
        proposal: 개선안.
        problems: 정적 검사 문제(비어 있으면 통과).
        evaluation: 평가 결과.
        decision: applied | declined | rejected_static | rejected_eval.
        path: 반영했으면 쓴 파일.
    """

    proposal: Proposal
    problems: list[str] = field(default_factory=list)
    evaluation: Evaluation = field(default_factory=Evaluation)
    decision: str = ""
    path: str = ""


def _log(data_dir: Path, outcome: Outcome, run_id: str) -> None:
    """improvements.jsonl 에 개선안·근거·평가·결정을 한 줄로 남긴다."""
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(data_dir / IMPROVEMENTS_FILE, {
            "ts": _now(), "run": run_id, "proposal": outcome.proposal.model_dump(),
            "static_problems": outcome.problems,
            "evaluation": {"evaluated": outcome.evaluation.evaluated,
                           "accepted": outcome.evaluation.accepted,
                           "reason": outcome.evaluation.reason,
                           "baseline": outcome.evaluation.baseline,
                           "candidate": outcome.evaluation.candidate},
            "decision": outcome.decision, "path": outcome.path})
    except (OSError, TypeError, ValueError):
        pass


def _summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"passed": sum(bool(r.get("pass")) for r in results), "total": len(results),
            "tokens": sum(int(r.get("tokens") or 0) for r in results),
            "cost": sum(float(r.get("cost") or 0.0) for r in results)}


def parse_improve_args(arg: str) -> tuple[bool, list[str] | None, str | None]:
    """`/improve` 의 인자(--no-eval, --tasks a,b)를 읽는다.

    Returns:
        (no_eval, 과제 id 목록 또는 None, 오류 문장 또는 None).
    """
    no_eval, tasks, words = False, None, arg.split()
    while words:
        word = words.pop(0)
        if word == "--no-eval":
            no_eval = True
        elif word == "--tasks" and words:
            tasks = [t for t in words.pop(0).split(",") if t]
        elif word.startswith("--tasks="):
            tasks = [t for t in word.partition("=")[2].split(",") if t]
        else:
            return no_eval, tasks, (f"알 수 없는 인자: {word}  "
                                    "(사용: /improve [--no-eval] [--tasks a,b])")
    return no_eval, tasks, None


def improve(settings: Settings, workspace: Path, console: Console, *, ask: Ask,
            model: BaseChatModel | None = None, no_eval: bool = False,
            task_ids: Sequence[str] | None = None, yes: bool = False,
            tracer: Tracer | None = None,
            models: Callable[[str, EvalTask], BaseChatModel] | None = None,
            limit_turns: int = 20,
            fetch: Callable[[str, dict[str, str]], Any] | None = None) -> list[Outcome]:
    """자기 개선 한 번을 돈다: 신호 수집 → 개선안 → 정적 검사 → 평가 → 승인 → 반영.

    반영은 사람이 diff 를 보고 승인(ask)하거나, --yes 이면서 평가를 돌려 채택된 개선안만
    자동으로 한다. 평가하지 않은 개선안(--no-eval, 프로젝트 대상)은 --yes 여도 ask 로 묻는다.

    Args:
        settings: 설정(data_dir 의 runs·memory 를 읽고 improvements.jsonl 을 쓴다).
        workspace: 작업 폴더(프로젝트 메모리·스킬 위치).
        console: 출력 콘솔.
        ask: 화면에 보인 질문에 y/N 으로 답하는 함수.
        model: 개선 모델. None 이면 settings 로 만든다.
        no_eval: 정적 검사만 하고 평가는 "평가 안 함"으로 둔다(무과금).
        task_ids: 평가 과제. None 이면 전부.
        yes: 평가에서 채택된 개선안을 묻지 않고 반영한다(헤드리스).
        tracer: Langfuse 같은 추가 트레이서(로컬 파일 기록은 항상 한다).
        models: (arm, 과제) → 평가 모델. 시험용.
        limit_turns: 신호를 모을 최근 턴 수.
        fetch: Judge 점수 조회의 HTTP 대역(시험용).

    Returns:
        개선안마다 Outcome. 신호나 개선안이 없으면 빈 목록.
    """
    from code_agent.models import make_model  # 지연 import: 정적 검사만 볼 때 SDK 를 안 부른다

    data_dir = settings.data_dir
    trace = ImproveTrace(data_dir, tracer)
    outcomes: list[Outcome] = []
    try:
        signals = collect_signals(data_dir, limit_turns, settings=settings, fetch=fetch)
        kinds: dict[str, int] = {}
        for s in signals:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        trace.emit("improve_signals", count=len(signals), kinds=kinds,
                   evidence=[s.evidence for s in signals][:50])
        if not signals:
            console.print("[dim]최근 세션에서 개선할 신호를 찾지 못했다.[/]")
            return outcomes
        console.print(f"신호 {len(signals)}개: "
                      + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())))
        snapshot = memory_snapshot(data_dir, workspace)
        proposals = propose(signals, snapshot, model if model is not None
                            else make_model(settings))
        if not proposals:
            console.print("[dim]개선안이 없다.[/]")
            return outcomes
        tasks = [] if no_eval else load_tasks(task_ids)
        baseline: list[TaskResult] | None = None
        if tasks and any(p.target in EVALUATED_TARGETS for p in proposals):
            # 기준은 한 번만 돌려 개선안들이 나눠 쓴다
            runs = 1 + sum(p.target in EVALUATED_TARGETS for p in proposals)
            console.print(f"평가 예상 비용 약 ${estimate_cost(tasks) * runs:.2f} "
                          f"(과제 {len(tasks)}개 x 기준 1회 + 후보 {runs - 1}회, 모델 호출 과금)")
            if not (yes or ask("평가를 진행할까? [y/N] › ")):
                console.print("[dim]평가를 건너뛴다(평가 안 함으로 표시).[/]")
                tasks = []
        for proposal in proposals:
            trace.emit("improve_proposal", target=proposal.target, skill_name=proposal.skill_name,
                       evidence=proposal.evidence, expected_effect=proposal.expected_effect,
                       chars=len(proposal.patch))
            outcome = Outcome(proposal, static_check(proposal, snapshot))
            outcomes.append(outcome)
            console.print(f"\n[bold]개선안[/] {proposal.target}"
                          + (f" {escape(proposal.skill_name)}" if proposal.skill_name else "")
                          + f"  근거 {escape(', '.join(proposal.evidence) or '(없음)')}\n"
                          f"기대 효과: {escape(proposal.expected_effect)}")
            if outcome.problems:
                outcome.decision = "rejected_static"
                console.print("[red]정적 검사 실패: "
                              + escape("; ".join(outcome.problems)) + "[/]")
            else:
                if not tasks:
                    outcome.evaluation = Evaluation(
                        reason=NOT_EVALUATED if proposal.target not in EVALUATED_TARGETS
                        else "평가 안 함")
                else:
                    if baseline is None and proposal.target in EVALUATED_TARGETS:
                        baseline = run_arm("baseline", settings, tasks, data_dir=_scratch(trace),
                                           memory_source=memory_dir(data_dir), models=models)
                    outcome.evaluation = evaluate(proposal, settings, tasks,
                                                  workspace=workspace, baseline=baseline,
                                                  models=models)
                ev = outcome.evaluation
                trace.emit("improve_eval", target=proposal.target, evaluated=ev.evaluated,
                           accepted=ev.accepted, reason=ev.reason,
                           baseline=_summary(ev.baseline), candidate=_summary(ev.candidate))
                console.print(f"평가: {escape(ev.reason)}")
                if ev.evaluated:
                    console.print(results_table([_row(r) for r in ev.candidate]))
                _decide(outcome, settings, workspace, console, ask=ask, yes=yes)
            trace.emit("improve_decision", target=proposal.target, decision=outcome.decision,
                       path=outcome.path or None)
            _log(data_dir, outcome, trace.run_id)
        return outcomes
    finally:
        trace.close()


def _scratch(trace: ImproveTrace) -> Path:
    """기준 arm 의 data dir 뿌리. 개선 실행 폴더 아래라 improve 실행과 함께 남는다."""
    return trace.folder / "eval"


def _row(data: Mapping[str, Any]) -> TaskResult:
    return TaskResult(id=str(data["id"]), passed=bool(data["pass"]), score=float(data["score"]),
                      cost=data.get("cost"), tokens=int(data.get("tokens") or 0),
                      calls=int(data.get("calls") or 0),
                      duration_s=float(data.get("duration_s") or 0.0),
                      warnings=list(data.get("warnings") or []),
                      rejections=list(data.get("rejections") or []),
                      stop=str(data.get("stop") or "done"))


def _decide(outcome: Outcome, settings: Settings, workspace: Path, console: Console, *,
            ask: Ask, yes: bool) -> None:
    """평가 결과에 따라 거부하거나 diff 를 보이고 승인받아 반영한다."""
    proposal, ev = outcome.proposal, outcome.evaluation
    if ev.evaluated and not ev.accepted:
        outcome.decision = "rejected_eval"
        console.print("[red]평가에서 채택되지 않았다. 반영하지 않는다.[/]")
        return
    root = memory_dir(settings.data_dir)
    console.print(escape(render_diff(proposal, memory_root=root, workspace=workspace)),
                  highlight=False, markup=True)
    if not (yes and ev.evaluated and ev.accepted):
        if not ev.evaluated:
            console.print(f"[yellow]{escape(ev.reason)} 사람의 승인으로만 반영한다.[/]")
        if not ask("이 개선안을 반영할까? [y/N] › "):
            outcome.decision = "declined"
            console.print("[dim]반영하지 않았다.[/]")
            return
    path = apply_patch(proposal, memory_root=root, workspace=workspace)
    outcome.decision, outcome.path = "applied", str(path)
    console.print(f"[green]반영했다:[/] {escape(str(path))}  [dim]다음 /new 부터 적용된다.[/]")
