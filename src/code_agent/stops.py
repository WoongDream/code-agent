"""턴이 말없이 멈추지 않게 한다. 멈춘 응답(출력 한도·공급자 오류·컨텍스트 창)의 정리와 자동 복구, 턴 끝 경고.

- 멈춘 응답에 도구 호출이 없으면 에이전트 루프는 그것을 최종 응답으로 보고 턴을 끝낸다. 생각만 하다 한도를 다 쓰면
  화면에는 사용량 한 줄만 남는다(az-quiz 세션: 64,000토큰 전부 reasoning, $1.32).
- TruncationRecoveryMiddleware 가 그런 응답을 정리하고(생각·잘린 도구 호출을 빼고 자리표시를 넣어 같은 id 로 교체)
  안내를 덧붙여 턴당 한 번 다시 부른다.
  - 정리하는 것은 방금 나온 마지막 메시지뿐이다. 원본은 다음 요청에 실린 적이 없어 프롬프트 캐시에도 없고, 그 뒤에
    서명된 thinking 도 없다. 그래서 Opus 5.5 의 보존된 thinking(서명이 앞쪽 기록 전체에 묶인다)을 깨지 않는다.
    그대로 두면 잘린 64K reasoning 을 이후 모든 호출이 다시 보낸다(az-quiz 기준 약 $1.15).
  - 정리할 때 서명 없는 thinking 블록(K1)과, OpenRouter 가 그대로 다시 보내는 반쯤 쓴 도구 인자 JSON(K2)도 빠진다.
  - 멈춘 응답의 마지막 도구 호출은 실행하지 않는다. langchain 이 잘린 인자를 고쳐 유효한 호출로 만들기 때문이다(실측).
    다른 유효한 호출은 그대로 실행하고, 잘린 호출(invalid_tool_calls)의 인자만 "{}" 로 바꾼다.
- 예산 미들웨어(exit_behavior="end")의 종료 메시지는 model 이 아닌 미들웨어 노드에서 나온다.
- TurnStops 는 스트림 조각에서 멈춘 응답·복구·예산 상한 도달을 모아 턴 끝의 경고 문구를 만든다. CLI 는 노란 줄로,
  트레이스는 turn 의 WARNING 수준과 작업 기록의 경고 절로 남긴다.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallLimitMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from code_agent.models import MAX_TOKENS

# 정지 사유(OpenAI 계열 finish_reason, Anthropic stop_reason). refusal·content_filter 는 다루지 않는다
STOP_LENGTH = frozenset({"length", "max_tokens"})  # 정리, 1회 복구, 경고
STOP_ERROR = frozenset({"error"})  # OpenRouter: choices 가 있는 오류 청크는 예외 없이 이 값으로 끝난다. 정리, 1회 복구
STOP_CONTEXT = frozenset({"model_context_window_exceeded"})  # Anthropic. 다시 불러도 같으므로 정리만 한다
RECOVERY_PROMPT = "직전 응답이 출력 한도로 잘렸다. 생각을 줄이고 결과물은 파일에 나눠서 바로 써라."  # 생각만 하다 잘렸다
RECOVERY_PROMPT_CUT = "직전 응답이 출력 한도로 잘려 답변이 끊겼다. 끊긴 곳부터 이어서 짧게 마무리하라."  # 답변이 끊겼다
RECOVERY_PROMPT_ERROR = "직전 응답이 공급자 오류로 중단됐다. 하던 작업을 이어서 진행하라."
RECOVERY_NAME = "truncation-recovery"  # 덧붙인 안내 메시지 표시. 두 공급자 모두 name 을 API 로 보내지 않는다
# 노드 이름은 "<미들웨어 name>.before_model" 형식이고, ToolCallLimitMiddleware 는 도구 이름이 붙을 수 있다
BUDGET_NODES = (ModelCallLimitMiddleware.__name__, ToolCallLimitMiddleware.__name__)
_REASONING_BLOCKS = frozenset({"thinking", "redacted_thinking", "reasoning"})


def stop_reason(message: BaseMessage) -> str | None:
    """멈춘 응답이면 그 사유(finish_reason 또는 stop_reason), 아니면 None."""
    if not isinstance(message, AIMessage):
        return None
    meta = message.response_metadata
    for reason in (meta.get("finish_reason"), meta.get("stop_reason")):
        if reason in STOP_LENGTH | STOP_ERROR | STOP_CONTEXT:
            return reason
    return None


def is_truncated(message: BaseMessage) -> bool:
    """출력 한도로 잘렸는지."""
    return stop_reason(message) in STOP_LENGTH


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


def _placeholder(reason: str, dropped: list[str]) -> str:
    """텍스트가 하나도 없는 멈춘 응답에 넣는 글. Anthropic 은 빈 assistant content 를 거부한다."""
    lost = f"{', '.join(dropped)} 호출 인자가 잘려 실행하지 않았다" if dropped else ""
    if reason in STOP_LENGTH:
        return f"[출력 한도({MAX_TOKENS:,}토큰)에 걸려 잘린 응답: {lost or '생각만 하다 끝나 결과물이 없다'}. 생각 내용은 기록에서 뺐다.]"
    if reason in STOP_ERROR:
        return f"[공급자 오류로 중단된 응답: {lost or '결과물이 없다'}.]"
    return f"[컨텍스트 창이 가득 차 중단된 응답{': ' + lost if lost else ''}.]"


PLACEHOLDER_LENGTH = _placeholder("length", [])
PLACEHOLDER_ERROR = _placeholder("error", [])
PLACEHOLDER_CONTEXT = _placeholder("model_context_window_exceeded", [])
_CUT_MARKERS = {"length": "[출력 한도로 여기서 잘림]", "error": "[공급자 오류로 여기서 끊김]",
                "context": "[컨텍스트 창이 가득 차 여기서 끊김]"}


def _cut_marker(reason: str, dropped: list[str]) -> str:
    kind = "length" if reason in STOP_LENGTH else "error" if reason in STOP_ERROR else "context"
    marker = _CUT_MARKERS[kind]
    return marker if not dropped else f"{marker[:-1]}. {', '.join(dropped)} 호출 인자가 잘려 실행하지 않았다]"


def _reasoning_chars(message: AIMessage) -> int:
    """뺄 생각의 글자 수. OpenRouter 는 reasoning_content 와 reasoning_details 에 같은 글이 함께 있어 한쪽만 센다."""
    kwargs = message.additional_kwargs
    text = kwargs.get("reasoning_content")
    if not isinstance(text, str):
        details = kwargs.get("reasoning_details")
        items = details if isinstance(details, list) else []
        text = "".join(t for d in items if isinstance(d, dict) and isinstance(t := d.get("text") or d.get("summary"), str))
    blocks = message.content if isinstance(message.content, list) else []
    in_blocks = sum(len(str(b.get("thinking") or b.get("reasoning") or "")) for b in blocks
                    if isinstance(b, dict) and b.get("type") in _REASONING_BLOCKS)
    return len(text) + in_blocks


def strip_stopped(message: BaseMessage) -> AIMessage | None:
    """도구 호출 없이 멈춘 응답에서 생각과 잘린 도구 호출을 뺀 같은 id 의 메시지. 대상이 아니면 None.

    텍스트만 남기고(없으면 자리표시), 끝에 끊긴 표시를 붙인다. 사용량과 응답 메타데이터(정지 사유)는 유지하고
    무엇을 뺐는지 response_metadata["code_agent_stripped"] 에 남긴다.
    """
    reason = stop_reason(message)
    if reason is None or not isinstance(message, AIMessage) or message.tool_calls:
        return None
    dropped = [call.get("name") or "도구" for call in message.invalid_tool_calls]
    text = message.text.rstrip()
    content = f"{text}\n\n{_cut_marker(reason, dropped)}" if text.strip() else _placeholder(reason, dropped)
    kwargs = {k: v for k, v in message.additional_kwargs.items()
              if k not in ("reasoning_content", "reasoning_details", "tool_calls")}
    meta = {**message.response_metadata,
            "code_agent_stripped": {"reasoning_chars": _reasoning_chars(message), "dropped_tool_calls": dropped}}
    return message.model_copy(update={"content": content, "additional_kwargs": kwargs, "tool_calls": [],
                                      "invalid_tool_calls": [], "response_metadata": meta})


HELD_ERROR = "출력이 멈춰 인자가 잘렸을 수 있어 실행하지 않았다"


def hold_cut_call(message: BaseMessage) -> AIMessage | None:
    """멈춘 응답의 마지막 도구 호출(쓰다가 멈춘 것)을 실행하지 않게 invalid_tool_calls(args "{}")로 옮긴 같은 id 의 메시지.

    langchain-core 는 스트림 청크를 합칠 때 parse_partial_json 으로 잘린 인자를 고쳐 유효한 호출로 만든다
    (AIMessageChunk.init_tool_calls, 실측: write_file content 가 중간에서 끊긴 채 유효한 호출이 됐다). 그대로 두면
    절반만 쓴 파일이나 잘린 셸 명령이 실행된다. 옮긴 호출에는 langchain 이 '인자가 잘렸다' 결과를 붙여 다시 쓰게 한다.
    대상이 아니면 None.
    """
    if stop_reason(message) is None or not isinstance(message, AIMessage) or not message.tool_calls:
        return None
    *kept, cut = message.tool_calls
    held = {"type": "invalid_tool_call", "name": cut["name"], "args": "{}", "id": cut["id"], "error": HELD_ERROR}
    meta = {**message.response_metadata, "code_agent_held_call": cut["name"]}
    return message.model_copy(update={"tool_calls": kept, "invalid_tool_calls": [*message.invalid_tool_calls, held],
                                      "response_metadata": meta})


def sanitize_invalid_calls(message: AIMessage) -> AIMessage | None:
    """유효한 호출과 잘린 호출이 섞인 멈춘 응답에서 잘린 호출의 인자만 "{}" 로 바꾼 같은 id 의 메시지(K2). 없으면 None.

    그러면 langchain 이 그 호출에 '인자가 잘렸다'는 결과를 붙여 다시 쓰게 하고, OpenRouter 로 반쯤 쓴 JSON 이 나가지 않는다.
    """
    if not message.invalid_tool_calls or all(call.get("args") == "{}" for call in message.invalid_tool_calls):
        return None
    calls = [{**call, "args": "{}"} for call in message.invalid_tool_calls]
    return message.model_copy(update={"invalid_tool_calls": calls})


class TruncationRecoveryMiddleware(AgentMiddleware):
    """멈춘 응답을 정리하고, 도구 호출이 없으면 턴을 끝내지 않고 안내를 덧붙여 모델을 한 번 더 부른다(턴당 1번).

    - wrap_model_call: 멈춘 응답의 마지막 도구 호출은 실행하지 않는다(hold_cut_call). 응답이 state 에 들어가기 전에
      바꿔야 HITL after_model 보다 앞서 잘린 호출의 승인을 묻지 않는다.
    - 출력 한도·공급자 오류: 정리하고 다시 부른다. 두 사유가 복구 1회를 함께 쓴다. 이미 복구했으면 정리만 한다.
    - 컨텍스트 창: 다시 불러도 같으므로 정리만 한다.
    - 유효한 도구 호출이 남은 응답은 잘린 호출의 인자만 고친다.

    after_model 은 미들웨어 목록의 역순으로 돈다. 목록 맨 앞에 두어 모델 호출 상한이 이번 호출을 센 뒤에 돌게 한다.
    jump_to "model" 은 루프 입구(before_model)로 가므로 다시 부를 때도 상한 검사를 거친다.
    """

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        response = handler(request)
        return replace(response, result=[hold_cut_call(m) or m for m in response.result])

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or (reason := stop_reason(last)) is None:
            return None
        if last.tool_calls:
            sanitized = sanitize_invalid_calls(last)
            return {"messages": [sanitized]} if sanitized is not None else None
        stripped = strip_stopped(last)
        if reason in STOP_CONTEXT or _recovered_this_turn(messages):
            return {"messages": [stripped]}
        prompt = (RECOVERY_PROMPT_ERROR if reason in STOP_ERROR
                  else RECOVERY_PROMPT_CUT if last.text.strip() else RECOVERY_PROMPT)
        return {"messages": [stripped, HumanMessage(prompt, name=RECOVERY_NAME)], "jump_to": "model"}


RECOVERY_NODE = TruncationRecoveryMiddleware.__name__  # 안내를 덧붙이는 노드는 "<이름>.after_model" 이다


def recovery_notice(node: str, message: BaseMessage) -> bool:
    """복구 미들웨어가 방금 덧붙인 안내인지.

    노드를 보는 이유: PatchToolCallsMiddleware 는 승인 중 끊긴 턴 다음에 기록 전체를 다시 내보내므로 옛 안내가 섞인다.
    """
    return node.startswith(RECOVERY_NODE) and is_recovery(message)


def budget_stop(node: str, message: BaseMessage) -> bool:
    """예산 미들웨어(exit_behavior="end")가 턴을 끝내며 남긴 AI 메시지인지.

    model 이 아닌 노드의 AI 메시지가 모두 이것은 아니다. HITL after_model 은 승인 뒤 같은 응답을 다시 내보낸다.
    """
    return node.startswith(BUDGET_NODES) and isinstance(message, AIMessage) and bool(message.text)


def _limit(message: AIMessage) -> int:
    """걸린 출력 한도. 공급자가 알려 준 출력 토큰 수가 없으면 설정값."""
    return (message.usage_metadata or {}).get("output_tokens") or MAX_TOKENS


def _why(reason: str, message: AIMessage) -> str:
    """'응답(N개)이 ~' 뒤에 오는 멈춘 사유."""
    if reason in STOP_LENGTH:
        return f"출력 한도({_limit(message):,}토큰)에 걸려 잘렸다."
    if reason in STOP_ERROR:
        return "공급자 오류로 중단됐다."
    return "컨텍스트 창이 가득 차 중단됐다."


class TurnStops:
    """메인 에이전트의 스트림 조각에서 멈춘 응답, 자동 복구, 예산 상한 도달을 모은다. model 노드의 원본만 본다."""

    def __init__(self) -> None:
        self.stopped: list[AIMessage] = []
        self.recoveries = 0
        self.last: AIMessage | None = None  # 마지막 모델 응답
        self.last_is_retry = False  # last 가 자동 복구 뒤에 나온 응답인지
        self.budget = ""  # 예산 미들웨어의 종료 메시지
        self.held: list[str] = []  # 실행하지 않은 잘린 도구 호출 이름

    def on_part(self, part: tuple[tuple[str, ...], str, Any]) -> None:
        ns, mode, data = part
        if ns or mode != "updates":
            return
        for node, update in data.items():
            messages = update.get("messages") if isinstance(update, dict) else None
            for message in messages if isinstance(messages, list) else []:
                if node == "model" and isinstance(message, AIMessage):
                    self.last, self.last_is_retry = message, self.recoveries > 0
                    if stop_reason(message) is not None:
                        self.stopped.append(message)
                    if held := message.response_metadata.get("code_agent_held_call"):
                        self.held.append(held)
                elif recovery_notice(node, message):
                    self.recoveries += 1
                elif budget_stop(node, message):
                    self.budget = message.text

    def warnings(self) -> list[str]:
        """턴 끝에 보일 경고 문장들. 없으면 빈 목록."""
        warnings = []
        last = self.last
        reason = stop_reason(last) if last is not None else None
        if last is not None and reason is not None and not last.tool_calls:
            if reason in STOP_CONTEXT:
                warnings.append("응답이 컨텍스트 창이 가득 차 중단됐다. /new 로 새로 시작하거나 요약한 뒤 이어 가라.")
            else:
                again = "또 잘렸다" if reason in STOP_LENGTH else "또 멈췄다"
                retried = f" 자동으로 한 번 다시 요청했지만 {again}." if self.last_is_retry else ""
                if last.text.strip():
                    what = "답변이 중간에 끊겼다."
                elif reason in STOP_LENGTH:
                    what = "생각만 하다 한도를 다 써서 답변도 도구 호출도 없다."
                else:
                    what = "답변도 도구 호출도 없다."
                warnings.append(f"응답이 {_why(reason, last)}{retried} {what} '계속해줘'로 이어 갈 수 있다.")
        elif self.stopped:
            retried = " 자동으로 한 번 다시 요청해 이어 갔다." if self.recoveries else ""
            for why, count in Counter(_why(stop_reason(m) or "", m) for m in self.stopped).items():
                warnings.append(f"응답 {count}개가 {why}{retried}")
        if self.held:
            warnings.append(f"{', '.join(self.held)} 호출은 {HELD_ERROR}.")
        if self.budget:
            warnings.append(f"예산 상한에 걸려 턴을 멈췄다({self.budget.rstrip('.')}). '계속해줘'로 이어 갈 수 있다.")
        return warnings
