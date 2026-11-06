"""턴이 말없이 멈추지 않게 한다. 출력 한도로 잘린 응답의 자동 복구와, 잘림·예산 상한 도달의 턴 끝 경고.

- 출력 한도에 걸린 응답에 도구 호출이 없으면 에이전트 루프는 그것을 최종 응답으로 보고 턴을 끝낸다. 생각만 하다
  한도를 다 쓰면 화면에는 사용량 한 줄만 남는다(az-quiz 세션: 64,000토큰 전부 reasoning, $1.32).
  TruncationRecoveryMiddleware 가 안내 메시지를 덧붙여 턴당 한 번 다시 부른다. Opus 5.5 는 이전 thinking 을
  보존하므로 기록은 고치지 않고 덧붙이기만 한다.
- 예산 미들웨어(exit_behavior="end")의 종료 메시지는 model 이 아닌 미들웨어 노드에서 나온다.
- TurnStops 는 스트림 조각에서 둘을 모아 턴 끝의 경고 문구를 만든다. CLI 는 노란 줄로, 트레이스는 turn 의
  WARNING 수준과 작업 기록의 경고 절로 남긴다.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from code_agent.models import MAX_TOKENS

TRUNCATED = frozenset({"length", "max_tokens"})  # OpenAI 계열 finish_reason, Anthropic stop_reason
RECOVERY_PROMPT = "직전 응답이 출력 한도로 잘렸다. 생각을 줄이고 결과물은 파일에 나눠서 바로 써라."
RECOVERY_NAME = "truncation-recovery"  # 덧붙인 안내 메시지 표시. 두 공급자 모두 name 을 API 로 보내지 않는다
# 노드 이름은 "<미들웨어 name>.before_model" 형식이고, ToolCallLimitMiddleware 는 도구 이름이 붙을 수 있다
BUDGET_NODES = (ModelCallLimitMiddleware.__name__, ToolCallLimitMiddleware.__name__)


def is_truncated(message: BaseMessage) -> bool:
    meta = message.response_metadata
    return isinstance(message, AIMessage) and (meta.get("finish_reason") in TRUNCATED
                                               or meta.get("stop_reason") in TRUNCATED)


def is_recovery(message: BaseMessage) -> bool:
    """자동 복구가 덧붙인 안내인지. 사용자 메시지가 아니므로 턴을 나눌 때 건너뛴다."""
    return isinstance(message, HumanMessage) and message.name == RECOVERY_NAME


def _recovered_this_turn(messages: list[BaseMessage]) -> bool:
    for message in reversed(messages):
        if is_recovery(message):
            return True
        if isinstance(message, HumanMessage):
            return False  # 이번 턴의 사용자 메시지까지 왔다
    return False


class TruncationRecoveryMiddleware(AgentMiddleware):
    """잘렸고 도구 호출도 없는 응답이면 턴을 끝내지 않고 안내를 덧붙여 모델을 한 번 더 부른다(턴당 1번).

    after_model 은 미들웨어 목록의 역순으로 돈다. 목록 맨 앞에 두어 모델 호출 상한이 이번 호출을 센 뒤에 돌게 한다.
    jump_to "model" 은 루프 입구(before_model)로 가므로 다시 부를 때도 상한 검사를 거친다.
    """

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if (not isinstance(last, AIMessage) or not is_truncated(last) or last.tool_calls
                or _recovered_this_turn(messages)):
            return None
        return {"messages": [HumanMessage(RECOVERY_PROMPT, name=RECOVERY_NAME)], "jump_to": "model"}


def _limit(message: AIMessage) -> int:
    """걸린 출력 한도. 공급자가 알려 준 출력 토큰 수가 없으면 설정값."""
    return (message.usage_metadata or {}).get("output_tokens") or MAX_TOKENS


class TurnStops:
    """메인 에이전트의 스트림 조각에서 잘린 응답, 자동 복구, 예산 상한 도달을 모은다."""

    def __init__(self) -> None:
        self.truncated: list[AIMessage] = []
        self.recoveries = 0
        self.last: AIMessage | None = None  # 마지막 모델 응답
        self.last_is_retry = False  # last 가 자동 복구 뒤에 나온 응답인지
        self.budget = ""  # 예산 미들웨어의 종료 메시지

    def on_part(self, part: tuple[tuple[str, ...], str, Any]) -> None:
        ns, mode, data = part
        if ns or mode != "updates":
            return
        for node, update in data.items():
            messages = update.get("messages") if isinstance(update, dict) else None
            for message in messages if isinstance(messages, list) else []:
                if node == "model" and isinstance(message, AIMessage):
                    self.last, self.last_is_retry = message, self.recoveries > 0
                    if is_truncated(message):
                        self.truncated.append(message)
                elif is_recovery(message):
                    self.recoveries += 1
                elif isinstance(message, AIMessage) and node.startswith(BUDGET_NODES):
                    self.budget = message.text

    def warnings(self) -> list[str]:
        """턴 끝에 보일 경고 문장들. 없으면 빈 목록."""
        warnings = []
        last = self.last
        if last is not None and is_truncated(last) and not last.tool_calls:
            retried = " 자동으로 한 번 다시 요청했지만 또 잘렸다." if self.last_is_retry else ""
            what = ("생각만 하다 한도를 다 써서 답변도 도구 호출도 없다." if not last.text.strip()
                    else "답변이 중간에 끊겼다.")
            warnings.append(f"응답이 출력 한도({_limit(last):,}토큰)에 걸려 잘렸다.{retried} {what} "
                            "'계속해줘'로 이어 갈 수 있다.")
        elif self.truncated:
            retried = " 자동으로 한 번 다시 요청해 이어 갔다." if self.recoveries else ""
            warnings.append(f"응답 {len(self.truncated)}개가 출력 한도({_limit(self.truncated[0]):,}토큰)에 "
                            f"걸려 잘렸다.{retried}")
        if self.budget:
            warnings.append(f"예산 상한에 걸려 턴을 멈췄다({self.budget.rstrip('.')}). '계속해줘'로 이어 갈 수 있다.")
        return warnings
