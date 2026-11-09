"""계획 모드. 조사(읽기 전용) → write_plan → plan-reviewer 검토 → 반영 → 사람 승인
→ 단계대로 구현 → 완료 조건 확인 보고.

에이전트 state 에 두 키를 더한다(PlanModeMiddleware.state_schema).
- plan_mode: /plan 으로 보낸 요청이면 참이다. 계획이 아직 없어도 쓰기를 막는다.
- plan: {version, status, data, reviews, scope_changes}. 체크포인트에 남아 --resume 해도 이어진다.
  status 는 draft(작성) → reviewed(검토 받음) → revised(검토 뒤 고침) → approved(사람 승인) 다.

PlanModeMiddleware 가 하는 일
- 도구 write_plan 과 request_plan_approval 을 준다. request_plan_approval 은 HITL 대상이라 사람이
  승인해야 실행되고, 실행되면 status=approved 와 단계별 todos 를 한 번에 쓴다.
- 쓰기 차단: plan_mode 이거나 승인되지 않은 계획이 있으면 파일 쓰기·수정·삭제와 읽기 전용이 아닌
  execute 를 오류 ToolMessage 로 돌려준다(deepagents _tool_exclusion.py 방식).
  메모리 파일(approval.is_memory_file)은 코드 변경이 아니라서 차단과 범위 변경에서 뺀다.
- 검토 기록: task(plan-reviewer) 결과를 plan.reviews 에 넣는다. 계획 하나에 MAX_REVIEWS 회까지다.
- 범위 변경: 승인된 계획 밖 파일 쓰기는 approval.auto_decision 이 모든 모드에서 사람에게 묻고,
  실행되면 scope_changes 에 남겨 계획 범위에 더한다.
  execute 로 바꾸는 파일(sed -i 등)은 추적하지 않는다.
- 사건: plan_written, plan_reviewed, plan_approval, scope_change 를 custom 스트림으로 보낸다
  (트레이스와 작업 기록이 받는다).

리뷰어 출력은 구조화 출력(response_format) 대신 첫 줄이 'VERDICT: ok|revise' 인 글이고
parse_review 가 읽는다.
response_format 은 대본 모델에서는 되지만(ToolStrategy) 실제 설정에서 믿기 어렵다.
- OpenRouter 의 Opus 5.5 는 프로필에 structured_output 이 없어 ToolStrategy 가 되고,
  ToolStrategy 는 tool_choice 를 강제한다.
  Anthropic 은 thinking 과 강제 tool_choice 를 함께 받지 않는다.
- Anthropic 직결은 langchain-anthropic 이 강제 tool_choice 를 경고와 함께 버려
  호출이 보장되지 않는다.
글 형식은 공급자와 관계없다. 읽지 못하면 verdict 를 unknown 으로 남기고 원문을 사람에게 보인다.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, NotRequired, cast

from deepagents import FilesystemMiddleware, SubAgent
from deepagents.backends import BackendProtocol
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallLimitMiddleware,
    ToolCallRequest,
)
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.types import Command
from pydantic import BaseModel, Field

from code_agent.approval import (
    FILE_TOOLS,
    PLAN_APPROVAL_TOOL,
    PLAN_REVIEW_FIRST,
    PlanContext,
    in_plan,
    is_memory_file,
    is_write_action,
    normalize_path,
)
from code_agent.caching import OpenRouterCacheMiddleware
from code_agent.guards import DotenvGuardMiddleware
from code_agent.stops import TruncationRecoveryMiddleware
from code_agent.tool_text import TOOL_DESCRIPTIONS

PLAN_TOOL = "write_plan"
REVIEWER = "plan-reviewer"
MAX_REVIEWS = 2  # 계획 하나(승인 전 고친 버전들)의 검토 횟수
REVIEWER_MODEL_CALLS = 20  # 리뷰어 한 번의 예산
REVIEWER_TOOL_CALLS = 60
REVIEW_CHARS = 4000  # state 에 남기는 검토 원문 길이
REVIEWED = frozenset({"reviewed", "revised", "approved"})
STATUS_LABEL = {"draft": "작성", "reviewed": "검토 받음", "revised": "검토 뒤 고침",
                "approved": "승인됨"}
# /plan <요청> 의 사용자 메시지 앞에 붙인다. system 은 바꾸지 않는다(캐시와 보존된 thinking)
PLAN_PREFIX = ("[계획 모드] 이 요청은 바로 고치지 말고 계획부터 세운다. "
               "순서: 읽기 전용으로 조사 → write_plan → task(subagent_type=\"plan-reviewer\") "
               "로 검토(description 에 아래 요청 원문) → 지적 반영 → request_plan_approval 로 "
               "사용자 승인 → 승인되면 단계대로 구현하고 완료 조건마다 확인 결과를 보고한다."
               "\n\n요청: ")

Middleware = AgentMiddleware[Any, Any, Any]

# 도구 설명은 deepagents 의 다른 도구 설명처럼 영어로 둔다(tool_text.py)
WRITE_PLAN_DESCRIPTION = """\
Save the implementation plan for the current request: a new plan, or a revision of the current \
one. Until the user approves the plan, file writes and non-read-only shell commands are blocked.
After saving, have it reviewed with task(subagent_type="plan-reviewer"), revise it with this tool \
if needed, then call request_plan_approval. Each step lists the files it changes; after approval, \
writing a file outside the plan needs the user's approval again."""

APPROVAL_DESCRIPTION = """\
Ask the user to approve the current plan, after plan-reviewer has reviewed it and you have \
revised it. The user sees the plan and the review findings and approves, or rejects with a \
comment. On approval the plan's steps become the todo list and writing is allowed; on rejection, \
revise the plan following the comment and ask again."""

REVIEWER_DESCRIPTION = (
    "Reviews the current plan (the one saved with write_plan; it is attached automatically) "
    "against the code, read-only, and returns `VERDICT: ok|revise` with findings. Call it after "
    "write_plan, at most 2 times per plan. In `description`, quote the user's original request "
    "verbatim and name anything specific to check.")

REVIEWER_PROMPT = """\
너는 코딩 에이전트가 세운 구현 계획을 검토하는 리뷰어다. 답변은 한국어로 한다.
검토할 계획은 이 메시지 끝에 있고, 사용자의 원래 요청과 확인할 점은 사용자 메시지에 있다.
파일 도구(ls, read_file, glob, grep)로 저장소를 읽어 계획이 실제 코드와 맞는지 확인한다. \
파일을 고치거나 명령을 실행할 수는 없다. 필요한 파일만 읽는다.

# 검토 기준
- 요구사항 누락: 사용자 요청의 요구사항이 모두 계획에 있는가.
- 완료 조건: 조건마다 명령이나 구체적인 확인으로 검증할 수 있는가.
- 단계: 단계마다 바꿀 파일과 확인 방법이 있는가. 빠진 파일(호출하는 쪽, 시험, 설정)은 없는가.
- 범위: 요청과 관계없는 변경이 끼어 있지 않은가.
- 위험: 깨질 수 있는 기존 동작, 보안(비밀값, 주입), 되돌리기 어려운 작업.
사소한 문체는 지적하지 않는다. 지적마다 근거(파일:줄)를 든다.
이전 검토가 붙어 있으면 그 지적이 반영됐는지도 본다.

# 출력 형식 (반드시 지킨다)
첫 줄은 `VERDICT: ok` 또는 `VERDICT: revise` 다. 고쳐야 할 high·medium 지적이 있으면 revise 다.
그다음 지적을 한 줄에 하나씩 쓴다: `- [high|medium|low] 문제 :: 고칠 방법`
지적이 없으면 `- 없음` 한 줄만 쓴다. 다른 글은 쓰지 않는다."""

_VERDICT = re.compile(r"VERDICT\W{0,6}(ok|revise)\b", re.IGNORECASE)
_FINDING = re.compile(r"^\s*[-*]\s*\[(high|medium|low)\]\s*(.+)$", re.IGNORECASE)
_SUGGESTION = re.compile(r"\s*(?:::|→|->)\s*")


class PlanStep(BaseModel):
    """계획의 한 단계: 할 일, 바꿀 파일, 확인 방법."""

    title: str = Field(min_length=1, description="What this step does, in one line.")
    files: list[str] = Field(
        default_factory=list,
        description=("Workspace files this step creates, edits or deletes, as virtual paths like "
                     "/src/app.py. End a path with / to cover a whole folder."))
    test: str = Field(description="How to verify this step: a command to run or a concrete check.")


class Plan(BaseModel):
    """write_plan 의 인자이자 state 에 남는 계획 내용(plan["data"])."""

    goal: str = Field(min_length=1,
                      description="The user's request summarized in one or two sentences.")
    requirements: list[str] = Field(default_factory=list,
                                    description="Each requirement in the user's request.")
    in_scope: list[str] = Field(default_factory=list, description="What this plan changes.")
    out_of_scope: list[str] = Field(default_factory=list,
                                    description="What this plan deliberately leaves unchanged.")
    done_criteria: list[str] = Field(
        min_length=1,
        description="Checkable conditions that show the work is done, each with how to check it.")
    steps: list[PlanStep] = Field(min_length=1, description="Implementation steps in order.")
    risks: list[str] = Field(default_factory=list,
                             description="Risks and open questions (optional).")


class PlanApprovalArgs(BaseModel):
    """request_plan_approval 의 인자. note 는 승인 프롬프트에서 사용자에게 보인다."""

    note: str = Field(default="", description=("Optional note for the user, e.g. review findings "
                                               "you did not apply and why."))


def _latest(_old: Any, new: Any) -> Any:
    """plan 의 reducer. 같은 단계에서 둘 이상 바꿔도(병렬 호출) 실패하지 않고 마지막 값을 쓴다."""
    return new


class PlanModeState(AgentState):
    """PlanModeMiddleware 가 더하는 state.

    두 키 모두 입력으로 받는다(/plan 이 plan_mode 를 넣는다).
    """

    plan_mode: NotRequired[bool]
    plan: NotRequired[Annotated[dict[str, Any] | None, _latest]]


class _ReviewerState(AgentState):
    """리뷰어가 부모에게서 받는 plan. 선언한 키만 서브에이전트 입력으로 받는다."""

    plan: NotRequired[Annotated[dict[str, Any] | None, _latest]]


def _plan_of(state: Any) -> dict[str, Any] | None:
    """state 의 plan. 없거나 dict 가 아니면 None."""
    plan = state.get("plan") if isinstance(state, Mapping) else None
    return plan if isinstance(plan, dict) else None


def plan_files(plan: Mapping[str, Any] | None) -> frozenset[str]:
    """계획 범위: 단계의 files 와 범위 변경(scope_changes)의 합집합(approval.normalize_path)."""
    if not plan:
        return frozenset()
    steps = (plan.get("data") or {}).get("steps") or []
    paths = [str(path) for step in steps for path in step.get("files") or []]
    paths += [str(change.get("file", "")) for change in plan.get("scope_changes") or []]
    return frozenset(normalize_path(path) for path in paths if path.strip())


def is_reviewed(plan: Mapping[str, Any] | None) -> bool:
    """plan-reviewer 검토를 받았는지(검토 뒤 고친 것과 승인된 것 포함)."""
    return plan is not None and plan.get("status") in REVIEWED


def is_approved(plan: Mapping[str, Any] | None) -> bool:
    """사람이 승인한 계획인지."""
    return plan is not None and plan.get("status") == "approved"


def block_message(plan_mode: bool, plan: Mapping[str, Any] | None) -> str:
    """쓰기를 막는 이유와 다음에 할 일. 막지 않으면 ''(계획이 승인됐거나 계획 모드가 아니다)."""
    if is_approved(plan) or (not plan and not plan_mode):
        return ""
    if not plan:
        why = "계획 모드인데 계획이 아직 없어"
        step = "읽기 전용으로 조사한 뒤 write_plan 으로 계획을 써라"
    else:
        why = f"계획(v{plan.get('version')})이 아직 승인되지 않아"
        step = ("request_plan_approval 로 사용자 승인을 받아라" if is_reviewed(plan)
                else "task(subagent_type=\"plan-reviewer\") 로 검토를 받고 지적을 반영한 뒤 "
                     "request_plan_approval 로 사용자 승인을 받아라")
    return (f"{why} 파일 쓰기·수정·삭제와 읽기 전용이 아닌 명령은 실행하지 않았다. 다음: {step}. "
            "파일 읽기와 읽기 전용 명령(git status, ls, cat 등)은 된다.")


def plan_context(values: Mapping[str, Any]) -> PlanContext | None:
    """agent state 값으로 승인 결정용 계획 상태(approval.PlanContext)를 만든다.

    Args:
        values: agent.get_state(config).values.

    Returns:
        계획 모드도 계획도 없으면 None.
    """
    plan, plan_mode = _plan_of(values), bool(values.get("plan_mode"))
    if plan is None and not plan_mode:
        return None
    return PlanContext(block_message=block_message(plan_mode, plan), reviewed=is_reviewed(plan),
                       approved=is_approved(plan), files=plan_files(plan))


def next_plan(previous: Mapping[str, Any] | None, data: dict[str, Any]) -> dict[str, Any]:
    """write_plan 이 저장할 새 plan.

    승인 전 계획을 고치면 같은 계획의 다음 버전이다.
    검토 기록을 이어 받고, 검토를 받았으면 revised 다.
    계획이 없거나 승인된 계획을 다시 쓰면 새 계획이라 검토 횟수와 범위 변경을 새로 센다.
    버전은 thread 안에서 계속 올라 사건과 기록에서 계획을 구별한다.
    """
    version = int((previous or {}).get("version") or 0) + 1
    if previous is None or is_approved(previous):
        return {"version": version, "status": "draft", "data": data, "reviews": [],
                "scope_changes": []}
    reviews = list(previous.get("reviews") or [])
    return {"version": version, "status": "revised" if reviews else "draft", "data": data,
            "reviews": reviews, "scope_changes": list(previous.get("scope_changes") or [])}


def parse_review(text: str) -> tuple[str, list[dict[str, str]]]:
    """리뷰어 답의 (verdict, findings). verdict 는 ok·revise 이고, 읽지 못하면 unknown 이다.

    지적 줄은 '- [high|medium|low] 문제 :: 고칠 방법' 이다(:: 대신 →, -> 도 받는다).
    """
    match = _VERDICT.search(text)
    findings = []
    for line in text.splitlines():
        if found := _FINDING.match(line):
            issue, *suggestion = _SUGGESTION.split(found[2], maxsplit=1)
            findings.append({"severity": found[1].lower(), "issue": issue.strip(" *`"),
                             "suggestion": suggestion[0].strip(" *`") if suggestion else ""})
    return (match[1].lower() if match else "unknown"), findings


def _bullets(title: str, items: Any) -> list[str]:
    return ["", f"## {title}", *(f"- {item}" for item in items)] if items else []


def render_plan(plan: Mapping[str, Any]) -> str:
    """계획 마크다운. 범위 변경과 검토 기록도 붙인다(승인 프롬프트, /plan, 리뷰어 system)."""
    data = plan.get("data") or {}
    status = str(plan.get("status", ""))
    lines = [f"# 계획 v{plan.get('version')} ({STATUS_LABEL.get(status, status)})", "",
             f"**목표**: {data.get('goal', '')}"]
    lines += _bullets("요구사항", data.get("requirements"))
    lines += _bullets("범위", data.get("in_scope"))
    lines += _bullets("범위 밖", data.get("out_of_scope"))
    lines += ["", "## 단계"]
    for number, step in enumerate(data.get("steps") or [], 1):
        files = ", ".join(f"`{path}`" for path in step.get("files") or []) or "(없음)"
        lines += [f"{number}. {step.get('title', '')}", f"   - 파일: {files}",
                  f"   - 확인: {step.get('test', '')}"]
    lines += _bullets("완료 조건", data.get("done_criteria"))
    lines += _bullets("위험", data.get("risks"))
    lines += _bullets("범위 변경 (승인 뒤 더한 파일)",
                      [f"`{c.get('file')}` ({c.get('tool')})"
                       for c in plan.get("scope_changes") or []])
    reviews = plan.get("reviews") or []
    if reviews:
        lines += ["", f"## 검토 ({len(reviews)}/{MAX_REVIEWS}회)"]
    for review in reviews:
        findings = review.get("findings") or []
        lines.append(f"- v{review.get('version')} 검토: {review.get('verdict')} "
                     f"(지적 {len(findings)}개)")
        lines += [f"  - [{f.get('severity')}] {f.get('issue')}"
                  + (f" → {f['suggestion']}" if f.get("suggestion") else "") for f in findings]
        if review.get("verdict") == "unknown" and review.get("text"):
            # 형식을 읽지 못했으면 원문을 그대로 보인다
            lines += [f"  > {line}" for line in str(review["text"]).splitlines()[:20]]
    return "\n".join(lines)


def _emit(event: dict[str, Any]) -> None:
    """custom 스트림 사건(트레이스·작업 기록용). 그래프 밖(단위 시험)이면 아무것도 하지 않는다."""
    try:
        get_stream_writer()(event)
    except RuntimeError:
        pass


def _reply(runtime: ToolRuntime, name: str, text: str, *, error: bool = False,
           **update: Any) -> Command:
    message = ToolMessage(text, tool_call_id=runtime.tool_call_id or "", name=name,
                          status="error" if error else "success")
    return Command(update={**update, "messages": [message]})


def _write_plan(runtime: ToolRuntime, **fields: Any) -> Command:
    """write_plan: 계획을 저장한다(next_plan). 인자는 Plan 이다."""
    data = Plan.model_validate(fields).model_dump()
    plan = next_plan(_plan_of(runtime.state), data)
    files = sorted(plan_files(plan))
    _emit({"type": "plan_written", "version": plan["version"], "files": files,
           "goal": data["goal"], "status": plan["status"]})
    reviews = len(plan["reviews"])
    if plan["status"] == "revised":
        step = ("request_plan_approval 로 사용자 승인을 요청하라. "
                "반영하지 않은 지적은 note 에 이유를 적는다.")
        if reviews < MAX_REVIEWS:
            step += (f" 크게 바꿨으면 plan-reviewer 검토를 한 번 더 받을 수 있다"
                     f"(검토 {reviews}/{MAX_REVIEWS}회).")
    else:
        step = ("task(subagent_type=\"plan-reviewer\") 로 검토를 받아라. description 에 사용자 "
                "요청 원문을 넣는다.")
    text = (f"계획(v{plan['version']})을 저장했다"
            f"(단계 {len(data['steps'])}개, 파일 {len(files)}개). "
            f"승인 전까지 쓰기는 막힌다. 다음: {step}")
    return _reply(runtime, PLAN_TOOL, text, plan=plan)


def _request_plan_approval(runtime: ToolRuntime, note: str = "") -> Command:
    """request_plan_approval: HITL 에서 사람이 승인해야 불린다.

    note 는 승인 프롬프트(cli)가 사용자에게 보이는 인자라 여기서는 쓰지 않는다.
    승인 결정이 검토를 확인하지만(approval.auto_decision) 다른 결정 함수로 돌 때를 위해 다시 본다.
    """
    plan = _plan_of(runtime.state)
    if plan is None:
        return _reply(runtime, PLAN_APPROVAL_TOOL,
                      "Error: 승인받을 계획이 없다. write_plan 으로 계획을 먼저 써라.", error=True)
    if is_approved(plan):
        return _reply(runtime, PLAN_APPROVAL_TOOL,
                      f"계획(v{plan.get('version')})은 이미 승인됐다. 단계대로 진행하라.")
    if not is_reviewed(plan):
        return _reply(runtime, PLAN_APPROVAL_TOOL, f"Error: {PLAN_REVIEW_FIRST}", error=True)
    data = plan.get("data") or {}
    todos = [{"content": step.get("title", ""), "status": "pending"}
             for step in data.get("steps") or []]
    _emit({"type": "plan_approval", "version": plan.get("version"), "steps": len(todos)})
    criteria = "\n".join(f"  {number}. {item}"
                         for number, item in enumerate(data.get("done_criteria") or [], 1))
    text = (f"사용자가 계획(v{plan.get('version')})을 승인했다. "
            f"단계 {len(todos)}개를 할 일(todos)로 넣었다.\n"
            "- 단계 순서대로 구현하고, 단계를 시작하거나 끝낼 때 write_todos 로 상태를 갱신한다.\n"
            "- 계획에 없는 파일을 바꿔야 하면 왜 필요한지 먼저 짧게 말한다. "
            "그 쓰기는 사용자에게 다시 묻는다.\n"
            "- 다 끝나면 완료 조건마다 충족 여부와 확인 방법(실행한 명령과 결과)을 보고한다:\n"
            f"{criteria}")
    return _reply(runtime, PLAN_APPROVAL_TOOL, text, plan={**plan, "status": "approved"},
                  todos=todos)


def _error(call: ToolCall, text: str) -> ToolMessage:
    return ToolMessage(content=f"Error: {text}", tool_call_id=call["id"] or "",
                       name=call["name"], status="error")


def _result_message(result: ToolMessage | Command) -> ToolMessage | None:
    """도구 결과의 ToolMessage. Command 면 update 의 messages 에서 찾는다."""
    if isinstance(result, ToolMessage):
        return result
    update = result.update if isinstance(result.update, dict) else {}
    return next((m for m in reversed(update.get("messages") or []) if isinstance(m, ToolMessage)),
                None)


def _failed(message: ToolMessage | None) -> bool:
    """오류로 끝난 도구 결과인지. deepagents 파일 도구는 오류를 'Error: ...' 글로도 돌려준다."""
    return message is None or message.status == "error" or message.text.startswith("Error")


def _with_update(result: ToolMessage | Command, update: dict[str, Any],
                 message: ToolMessage | None = None) -> Command:
    """도구 결과에 state 갱신을 합친 Command.

    message 를 주면 결과의 ToolMessage(같은 tool_call_id)를 그것으로 바꾼다.
    """
    if isinstance(result, ToolMessage):
        return Command(update={**update, "messages": [message or result]})
    base = dict(result.update) if isinstance(result.update, dict) else {}
    messages = list(base.get("messages") or [])
    if message is not None:
        messages = [message if isinstance(m, ToolMessage)
                    and m.tool_call_id == message.tool_call_id else m for m in messages]
    return replace(result, update={**base, **update, "messages": messages})


class PlanModeMiddleware(AgentMiddleware):
    """계획 모드의 state·도구·쓰기 차단·검토 기록·범위 변경 기록(모듈 docstring)."""

    state_schema = PlanModeState

    def __init__(self) -> None:
        super().__init__()
        self.tools = [
            StructuredTool.from_function(name=PLAN_TOOL, description=WRITE_PLAN_DESCRIPTION,
                                         func=_write_plan, args_schema=Plan, infer_schema=False),
            StructuredTool.from_function(name=PLAN_APPROVAL_TOOL, description=APPROVAL_DESCRIPTION,
                                         func=_request_plan_approval,
                                         args_schema=PlanApprovalArgs, infer_schema=False),
        ]

    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command],
                       ) -> ToolMessage | Command:
        """승인 전 쓰기를 막고, 리뷰어 결과와 계획 밖 파일 쓰기를 plan 에 남긴다."""
        call = request.tool_call
        name, args = call["name"], call.get("args") or {}
        if name == "task" and args.get("subagent_type") == REVIEWER:
            return self._review(request, handler)
        state = request.state if isinstance(request.state, Mapping) else {}
        plan = _plan_of(state)
        if (blocked := block_message(bool(state.get("plan_mode")), plan)) and is_write_action(
                name, args):
            return _error(call, blocked)
        path = str(args.get("file_path", ""))
        # 메모리 파일은 코드 변경이 아니라서 차단(is_write_action)과 범위 변경에서 뺀다
        if (plan is not None and is_approved(plan) and name in FILE_TOOLS
                and not is_memory_file(path) and not in_plan(path, plan_files(plan))):
            # 여기까지 왔으면 사람이 승인했다(approval.auto_decision 이 모든 모드에서 묻는다)
            result = handler(request)
            if _failed(_result_message(result)):
                return result
            change = {"file": path, "tool": name, "version": plan.get("version")}
            _emit({"type": "scope_change", **change})
            return _with_update(result, {"plan": {
                **plan, "scope_changes": [*(plan.get("scope_changes") or []), change]}})
        return handler(request)

    def _review(self, request: ToolCallRequest,
                handler: Callable[[ToolCallRequest], ToolMessage | Command],
                ) -> ToolMessage | Command:
        """task(plan-reviewer) 를 부르고 결과를 plan.reviews 에 넣는다.

        계획이 없거나, 이미 승인됐거나, 검토를 MAX_REVIEWS 회 받았으면 부르지 않고 오류로 답한다.
        """
        call = request.tool_call
        plan = _plan_of(request.state)
        if plan is None:
            return _error(call, "검토할 계획이 없다. write_plan 으로 계획을 먼저 써라.")
        if is_approved(plan):
            return _error(call, "승인된 계획은 다시 검토하지 않는다. "
                                "계획을 바꾸려면 write_plan 으로 새로 쓴다.")
        reviews = list(plan.get("reviews") or [])
        if len(reviews) >= MAX_REVIEWS:
            return _error(call, f"계획 하나의 검토는 {MAX_REVIEWS}회까지라 "
                                "리뷰어를 부르지 않았다. 남은 지적은 직접 판단해 반영하고 "
                                "request_plan_approval 로 사용자 승인을 요청하라"
                                "(반영하지 않은 지적은 note 에 이유를 적는다).")
        result = handler(request)
        message = _result_message(result)
        if message is None or message.status == "error":
            return result
        verdict, findings = parse_review(message.text)
        reviews.append({"version": plan.get("version"), "verdict": verdict, "findings": findings,
                        "text": message.text[:REVIEW_CHARS]})
        _emit({"type": "plan_reviewed", "version": plan.get("version"), "verdict": verdict,
               "findings": findings})
        step = {"ok": "request_plan_approval 로 사용자 승인을 요청하라.",
                "revise": ("지적을 반영해 write_plan 으로 계획을 고친 뒤 request_plan_approval 로 "
                           "사용자 승인을 요청하라.")}.get(
            verdict, "VERDICT 줄을 읽지 못했다. 위 내용을 직접 판단해 반영한 뒤 "
                     "request_plan_approval 로 사용자 승인을 요청하라.")
        note = f"(plan-reviewer 검토 {len(reviews)}/{MAX_REVIEWS}회. {step})"
        return _with_update(result, {"plan": {**plan, "status": "reviewed", "reviews": reviews}},
                            message.model_copy(update={"content": f"{message.text}\n\n{note}"}))


class PlanBriefMiddleware(AgentMiddleware):
    """리뷰어 system 끝에 검토할 계획과 이전 검토를 붙인다.

    서브에이전트 state 는 부모 state 에서 messages·todos 등만 뺀 것이라
    (deepagents subagents.py _EXCLUDED_STATE_KEYS) 부모의 plan 이 넘어온다.
    넘어온 키는 state_schema 에 선언해야 받는다.
    리뷰어 대화는 호출마다 새것이라 system 에 계획을 넣어도 캐시와 보존된 thinking 문제가 없다.
    """

    state_schema = _ReviewerState

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """요청의 system 에 '# 검토할 계획' 절을 덧붙인다."""
        plan = _plan_of(request.state)
        brief = ("\n\n# 검토할 계획\n" + (render_plan(plan) if plan is not None else
                                         "(계획이 없다. 'VERDICT: revise' 와 계획이 없다는 지적 "
                                         "하나만 답한다.)"))
        system = request.system_message
        blocks = [*(system.content_blocks if system is not None else []),
                  {"type": "text", "text": brief}]
        return handler(request.override(
            system_message=SystemMessage(content=cast("list[str | dict[str, Any]]", blocks))))


def plan_reviewer(model: BaseChatModel, backend: BackendProtocol, workspace: Path, *,
                  summarization: Middleware) -> SubAgent:
    """plan-reviewer 서브에이전트 spec.

    isolated 서브에이전트는 메인 middleware 를 물려받지 않으므로(deepagents graph.py:771-775,
    task8 0절) 필요한 것을 모두 명시한다.
    - 읽기 전용 파일 도구: FilesystemMiddleware(tools=ls·read_file·glob·grep) 가 기본을 이름으로
      대체해 write_file·edit_file·delete·execute 가 아예 만들어지지 않는다.
    - .env 가드, OpenRouter 캐시, 잘림 복구, 계획 주입(PlanBriefMiddleware),
      예산(모델 20회, 도구 60회).
    - 요약(summarization): agent.summarization_middleware 로 리뷰어 모델에 맞춰 만든 것을 받는다
      (task3, 도구 인자 잘림 끔). agent.py 가 이 모듈을 쓰므로 거꾸로 import 하지 않는다.
    interrupt_on 은 비워 메인의 승인 설정을 물려받지 않는다(승인 대상 도구가 없다).

    Args:
        model: 리뷰어 모델(settings.reviewer_model 로 만든 인스턴스).
        backend: 메인과 같은 파일 백엔드. 작업 폴더 밖은 볼 수 없다.
        workspace: 작업 폴더(.env 가드가 실제 파일 이름을 본다).
        summarization: 리뷰어용 요약 미들웨어.

    Returns:
        create_deep_agent(subagents=[...]) 에 넣을 SubAgent.
    """
    return {
        "name": REVIEWER,
        "description": REVIEWER_DESCRIPTION,
        "system_prompt": REVIEWER_PROMPT,
        "model": model,
        "tools": [],
        "middleware": [
            FilesystemMiddleware(backend=backend, tools=["ls", "read_file", "glob", "grep"],
                                 custom_tool_descriptions=TOOL_DESCRIPTIONS),
            TruncationRecoveryMiddleware(), DotenvGuardMiddleware(workspace),
            OpenRouterCacheMiddleware(), PlanBriefMiddleware(),
            ModelCallLimitMiddleware(run_limit=REVIEWER_MODEL_CALLS, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=REVIEWER_TOOL_CALLS, exit_behavior="end"),
            summarization,
        ],
        "interrupt_on": {},
    }
