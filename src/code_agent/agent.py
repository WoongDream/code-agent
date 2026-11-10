"""에이전트 조립과 한 턴 실행.

deepagents·langgraph API 와 결합되는 곳은 여기와 guards.py 뿐이다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)

# 0.7.32 고정. 모델 인스턴스로 공급자를 얻는 공개 API 가 없다.
from deepagents._models import get_model_provider
from deepagents.backends import CompositeBackend
from deepagents.middleware import SummarizationMiddleware
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from code_agent.approval import PLAN_APPROVAL_TOOL
from code_agent.backend import USER_MEMORY_ROUTE, WindowsShellBackend, make_backend
from code_agent.caching import OpenRouterCacheMiddleware
from code_agent.guards import DotenvGuardMiddleware
from code_agent.memory import (
    MEMORY_SOURCES,
    SKILL_LABELS,
    SKILL_SOURCES,
    AgentMemoryMiddleware,
    MemoryTrust,
    QuietSkillsMiddleware,
    memory_dir,
)
from code_agent.models import make_model
from code_agent.planning import PlanModeMiddleware, plan_reviewer
from code_agent.prompts import coding_prompt
from code_agent.resume_compat import MemoryPromptStore, PromptFingerprintMiddleware, PromptStore
from code_agent.settings import Settings
from code_agent.stops import TruncationRecoveryMiddleware
from code_agent.tool_text import TODO_PROMPT, TOOL_DESCRIPTIONS, ToolArgsMiddleware

# 파일·셸 승인 대상. 계획 승인(request_plan_approval)은 build_agent 가 따로 더한다
# (scripts/backfill_langfuse.py 는 이 목록을 파일·셸 승인으로 읽는다)
INTERRUPT_TOOLS = ("execute", "write_file", "edit_file", "delete")
MODEL_CALLS = 100  # 사용자 메시지 하나(재개 포함)의 모델 호출 상한
TOOL_CALLS = 300
RECURSION_LIMIT = 1000  # 그래프 단계 수. 예산 미들웨어가 먼저 끊도록 넉넉히 둔다

StreamPart = tuple[tuple[str, ...], str, Any]
# 미들웨어마다 state 타입이 다르다(PlanningState 등).
# AgentMiddleware 는 state 에 불변이라 한 목록에 담으려면 Any 로 묶는다.
Middleware = AgentMiddleware[Any, Any, Any]


def budget_middleware(model_calls: int = MODEL_CALLS,
                      tool_calls: int = TOOL_CALLS) -> list[Middleware]:
    """exit_behavior="end" 라 초과하면 오류 대신 안내 메시지로 턴을 끝낸다
    (pjt Day08 guardlab/budget.py).
    """
    return [
        ModelCallLimitMiddleware(run_limit=model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=tool_calls, exit_behavior="end"),
    ]


def summarization_middleware(model: BaseChatModel, backend: CompositeBackend,
                             context_tokens: int) -> Middleware:
    """deepagents 의 기본 요약을 같은 이름(SummarizationMiddleware)으로 core 자리에서 바꾼다.

    - 도구 인자 잘림을 끈다(truncate_args_settings=None).
      잘림은 중간 메시지를 바꿔 프롬프트 캐시를 깨고 Opus 5.5 의 보존된 thinking 을 무효로
      만든다(서명이 앞쪽 기록 전체에 묶인다).
    - 프로필의 max_input_tokens(models.with_context_limit)의 85% 에서 요약하고 10% 를 남긴다.
      프로필이 없는 모델(대본 모델 등)은 context_tokens 를 쓴다.
      fraction 은 프로필이 없으면 생성자가 ValueError 를 내서 tokens 로 준다.
    - trim_tokens_to_summarize=None: 생성자 기본값(4,000)이면 요약기가 버릴 기록의
      마지막 4K 만 본다. deepagents 팩토리(create_summarization_middleware)도 None 을 넘긴다.
    """
    limit = (model.profile or {}).get("max_input_tokens") or context_tokens
    return SummarizationMiddleware(model=model, backend=backend,
                                   trigger=("tokens", int(limit * 0.85)),
                                   keep=("tokens", int(limit * 0.10)),
                                   trim_tokens_to_summarize=None, truncate_args_settings=None)


def _register_harness_profile(model: BaseChatModel) -> None:
    """프로필은 공급자 단위로 등록되고 내장 프로필과 병합된다.

    - general-purpose 서브에이전트는 끈다. 메인 문맥을 복제해 비싸고, 서브에이전트는
      plan-reviewer(planning.py)만 둔다.
    - 기본 도구 설명 중 우리 환경과 다른 것(execute 의 'sandbox'·절대 경로·&& 연결,
      ls 의 'ALWAYS', task 의 동시 실행 권유 등)을 고친다(tool_text).
      등록은 프로세스 전역이고 같은 키에 병합된다(deepagents _merge_profiles).
    """
    provider = get_model_provider(model)
    if provider:
        register_harness_profile(provider, HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            tool_description_overrides=TOOL_DESCRIPTIONS))


def build_agent(settings: Settings, workspace: Path, checkpointer: BaseCheckpointSaver | None, *,
                model: BaseChatModel | None = None, reviewer_model: BaseChatModel | None = None,
                backend: CompositeBackend | None = None, env_text: str = "",
                prompt_store: PromptStore | None = None, memory_trust: MemoryTrust | None = None):
    """코딩 에이전트 그래프를 조립한다.

    승인 대상 도구(INTERRUPT_TOOLS)와 계획 승인 요청(request_plan_approval)은 항상 interrupt 한다.
    계획 모드(planning.PlanModeMiddleware)와 그 리뷰어 서브에이전트(plan-reviewer)를 붙인다.
    메모리(사용자 공통·프로젝트 AGENTS.md)와 스킬 폴더를 잇는다(memory.py). thread 의 첫 요청 때
    한 번 읽어 system 끝에 붙는다.

    Args:
        settings: 모델·셸·데이터 폴더·컨텍스트 상한 설정(settings.load_settings).
        workspace: 작업 폴더. 파일 도구의 루트이자 셸의 현재 폴더다.
        checkpointer: 대화 상태 저장소. None 이면 상태를 남기지 않는다.
        model: 쓸 모델. None 이면 settings 로 만든다(models.make_model). 시험은 대본 모델을 넣는다.
        reviewer_model: plan-reviewer 모델. None 이면 settings.reviewer_model 로 만들고,
            그것이 비었거나 메인과 같은 이름이면 메인 모델을 같이 쓴다.
        backend: 파일·셸 백엔드. None 이면 make_backend 로 만든다(사용자 공통 메모리 라우트 포함).
        env_text: 시스템 프롬프트 끝의 '이 PC 의 도구' 절이다(prompts.coding_prompt).
        prompt_store: thread 별 프롬프트 지문 저장소다(resume_compat).
            없으면 이 agent 안에서만 기억한다.
        memory_trust: 프로젝트 /AGENTS.md 를 메모리로 읽어도 되는 내용(memory.MemoryTrust).
            App 이 thread 를 시작할 때 고친다. None 이면 프로젝트 AGENTS.md 를 읽지 않는다.

    Returns:
        컴파일된 deepagents 그래프. run_turn 으로 돌리고 get_state 로 상태를 읽는다.

    Raises:
        ValueError: backend 를 줬는데 사용자 공통 메모리 라우트(USER_MEMORY_ROUTE)가 없을 때.
            없으면 그 경로가 작업 폴더로 가서 저장소 파일이 사용자 메모리로 읽힌다.
    """
    model = model if model is not None else make_model(settings)
    if reviewer_model is None:
        name = settings.reviewer_model or settings.model
        reviewer_model = (model if name == settings.model
                          else make_model(replace(settings, model=name)))
    _register_harness_profile(model)
    if reviewer_model is not model:
        # 공급자가 다르면 그 프로필에도 도구 설명을 넣는다
        _register_harness_profile(reviewer_model)
    if backend is None:
        backend = make_backend(workspace, artifacts_dir=settings.data_dir / "artifacts",
                               memory_dir=memory_dir(settings.data_dir), shell=settings.shell)
    elif USER_MEMORY_ROUTE not in backend.routes:
        raise ValueError(f"backend 에 사용자 공통 메모리 라우트({USER_MEMORY_ROUTE})가 없다. "
                         "make_backend(memory_dir=...) 로 만든다.")
    # 메모리·스킬은 deepagents 의 기본 것과 이름이 같아 그 자리(system 의 맨 끝)를 대신한다.
    # 지문은 그 두 절도 본다(system_parts).
    trust = memory_trust if memory_trust is not None else MemoryTrust()
    memory = AgentMemoryMiddleware(backend=backend, sources=MEMORY_SOURCES, trust=trust)
    # 프로젝트 스킬도 AGENTS.md 와 같은 신뢰 결정을 따른다
    skills = QuietSkillsMiddleware(backend=backend, trust=trust, workspace=workspace,
                                   sources=[(path, SKILL_LABELS[path]) for path in SKILL_SOURCES])
    # 잘림 복구는 맨 앞이어야 after_model 이 예산 미들웨어의 호출 집계 뒤에 돈다
    # (after_model 은 역순).
    # 요약은 이름이 같아 core 자리를 대신하므로 목록 위치와 상관없다.
    # .env 가드는 작업 폴더로 실제 파일 이름도 본다.
    # 지문은 TodoList 와 도구 인자 고치기 뒤라야(wrap_model_call 은 앞이 바깥)
    # 모델이 받는 system·도구를 본다.
    # 계획 모드는 도구 실행만 감싸므로(wrap_tool_call) 위치가 system·도구에 영향을 주지 않는다.
    store = prompt_store if prompt_store is not None else MemoryPromptStore()
    middleware: list[Middleware] = [
        TruncationRecoveryMiddleware(), TodoListMiddleware(system_prompt=TODO_PROMPT),
        PlanModeMiddleware(), ToolArgsMiddleware(),
        PromptFingerprintMiddleware(store, system_parts=(skills.modify_request,
                                                         memory.modify_request)),
        DotenvGuardMiddleware(workspace), OpenRouterCacheMiddleware(),
        *budget_middleware(), summarization_middleware(model, backend, settings.context_tokens),
        skills, memory]
    reviewer = plan_reviewer(reviewer_model, backend, workspace,
                             summarization=summarization_middleware(
                                 reviewer_model, backend, settings.context_tokens))
    # default 는 make_backend 의 셸 백엔드다
    shell = cast(WindowsShellBackend, backend.default).shell
    return create_deep_agent(
        name="code-agent", model=model, backend=backend,
        system_prompt=coding_prompt(workspace=workspace, shell=shell, env_text=env_text),
        middleware=middleware, subagents=[reviewer],
        # 기본 자리를 만들게 넘기고, 같은 이름의 우리 middleware 가 그것을 대신한다
        memory=list(MEMORY_SOURCES), skills=list(SKILL_SOURCES),
        # 결정은 approve/reject 만 받는다. edit 은 고친 인자가 재검증 없이 실행된다.
        # 계획 승인은 사람이 하는 일이라 모드와 관계없이 묻는다(approval.auto_decision)
        interrupt_on={tool: {"allowed_decisions": ["approve", "reject"]}
                      for tool in (*INTERRUPT_TOOLS, PLAN_APPROVAL_TOOL)},
        checkpointer=checkpointer,
    )


def run_turn(agent, inp: dict | Command, config: dict, *,
             decide: Callable[[list[dict]], list[dict]],
             on_part: Callable[[StreamPart], None]) -> None:
    """interrupt 가 없어질 때까지 스트림 → 결정 → 재개를 반복한다
    (pjt Day08 guardlab/runner.py:107).

    서브에이전트의 interrupt 는 ns 가 다른 두 이벤트로 같은 id 를 달고 오므로
    id 로 중복을 없앤다(0단계 d).
    custom 모드는 미들웨어가 get_stream_writer 로 보내는 알림이다
    (resume_compat 의 prompt_migrated 등).

    Args:
        agent: build_agent 가 만든 그래프.
        inp: 새 입력({"messages": [...]}) 또는 재개 Command.
        config: langgraph 실행 설정(configurable.thread_id 등).
            recursion_limit 이 없으면 RECURSION_LIMIT 을 쓴다.
        decide: 승인 요청(action_requests) 목록을 받아 같은 순서의 결정
            ({"type": "approve"} 또는 reject) 목록을 돌려준다(approval.decide_all).
        on_part: interrupt 가 아닌 스트림 조각 (ns, mode, data) 마다 불린다.

    Returns:
        None. 출력은 on_part 로 흘려보내고, 대화 상태는 agent 의 checkpointer 에 남는다.
    """
    config = {"recursion_limit": RECURSION_LIMIT, **config}
    while True:
        pending: dict[str, Any] = {}
        for part in agent.stream(inp, config, stream_mode=["messages", "updates", "custom"],
                                 subgraphs=True):
            _ns, mode, data = part
            if mode == "updates" and "__interrupt__" in data:
                for interrupt in data["__interrupt__"]:
                    pending.setdefault(interrupt.id, interrupt)
            else:
                on_part(part)
        if not pending:
            return
        answers = {iid: {"decisions": decide(i.value["action_requests"])}
                   for iid, i in pending.items()}
        inp = Command(resume=next(iter(answers.values())) if len(answers) == 1 else answers)
