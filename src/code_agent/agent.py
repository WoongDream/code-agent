"""에이전트 조립과 한 턴 실행. deepagents·langgraph API 와 결합되는 곳은 여기와 guards.py 뿐이다."""

from __future__ import annotations

from collections.abc import Callable
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
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from code_agent.backend import WindowsShellBackend, make_backend
from code_agent.caching import OpenRouterCacheMiddleware
from code_agent.guards import DotenvGuardMiddleware
from code_agent.models import make_model
from code_agent.prompts import coding_prompt
from code_agent.settings import Settings
from code_agent.stops import TruncationRecoveryMiddleware

INTERRUPT_TOOLS = ("execute", "write_file", "edit_file", "delete")
MODEL_CALLS = 100  # 사용자 메시지 하나(재개 포함)의 모델 호출 상한
TOOL_CALLS = 300
RECURSION_LIMIT = 1000  # 그래프 단계 수. 예산 미들웨어가 먼저 끊도록 넉넉히 둔다

StreamPart = tuple[tuple[str, ...], str, Any]
# 미들웨어마다 state 타입이 다르다(PlanningState 등). AgentMiddleware 는 state 에 불변이라 한 목록에 담으려면 Any 로 묶는다.
Middleware = AgentMiddleware[Any, Any, Any]


def budget_middleware(model_calls: int = MODEL_CALLS, tool_calls: int = TOOL_CALLS) -> list[Middleware]:
    """exit_behavior="end" 라 초과하면 오류 대신 안내 메시지로 턴을 끝낸다(pjt Day08 guardlab/budget.py)."""
    return [
        ModelCallLimitMiddleware(run_limit=model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=tool_calls, exit_behavior="end"),
    ]


def _disable_general_purpose_subagent(model: BaseChatModel) -> None:
    """MVP 는 서브에이전트를 쓰지 않는다. 프로필은 공급자 단위로 등록되고 내장 프로필과 병합된다."""
    provider = get_model_provider(model)
    if provider:
        register_harness_profile(provider, HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))


def build_agent(settings: Settings, workspace: Path, checkpointer: BaseCheckpointSaver | None, *,
                model: BaseChatModel | None = None, backend: CompositeBackend | None = None):
    model = model if model is not None else make_model(settings)
    _disable_general_purpose_subagent(model)
    if backend is None:
        backend = make_backend(workspace, artifacts_dir=settings.data_dir / "artifacts", shell=settings.shell)
    # 잘림 복구는 맨 앞이어야 after_model 이 예산 미들웨어의 호출 집계 뒤에 돈다(after_model 은 역순)
    middleware: list[Middleware] = [TruncationRecoveryMiddleware(), TodoListMiddleware(), DotenvGuardMiddleware(),
                                    OpenRouterCacheMiddleware(), *budget_middleware()]
    shell = cast(WindowsShellBackend, backend.default).shell  # default 는 make_backend 의 셸 백엔드다
    return create_deep_agent(
        name="code-agent", model=model, backend=backend,
        system_prompt=coding_prompt(workspace=workspace, shell=shell),
        middleware=middleware,
        # 결정은 approve/reject 만 받는다. edit 은 고친 인자가 재검증 없이 실행된다.
        interrupt_on={tool: {"allowed_decisions": ["approve", "reject"]} for tool in INTERRUPT_TOOLS},
        checkpointer=checkpointer,
    )


def run_turn(agent, inp: dict | Command, config: dict, *,
             decide: Callable[[list[dict]], list[dict]], on_part: Callable[[StreamPart], None]) -> None:
    """interrupt 가 없어질 때까지 스트림 → 결정 → 재개를 반복한다(pjt Day08 guardlab/runner.py:107).

    서브에이전트의 interrupt 는 ns 가 다른 두 이벤트로 같은 id 를 달고 오므로 id 로 중복을 없앤다(0단계 d).
    """
    config = {"recursion_limit": RECURSION_LIMIT, **config}
    while True:
        pending: dict[str, Any] = {}
        for part in agent.stream(inp, config, stream_mode=["messages", "updates"], subgraphs=True):
            _ns, mode, data = part
            if mode == "updates" and "__interrupt__" in data:
                for interrupt in data["__interrupt__"]:
                    pending.setdefault(interrupt.id, interrupt)
            else:
                on_part(part)
        if not pending:
            return
        answers = {iid: {"decisions": decide(i.value["action_requests"])} for iid, i in pending.items()}
        inp = Command(resume=next(iter(answers.values())) if len(answers) == 1 else answers)
