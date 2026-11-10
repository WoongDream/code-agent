"""단위 시험 전용 가짜 모델.

비용 없이 에이전트 배선을 시험한다. pjt Day02_new/tests/fakes.py 에서 가져왔다.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, SkipValidation


class ScriptExhaustedError(RuntimeError):
    """대본을 다 썼는데 모델이 또 불렸다."""


class ScriptedChatModel(BaseChatModel):
    """steps를 차례로 돌려준다. callable 단계는 그때까지의 메시지를 받아 응답을 만든다."""
    steps: SkipValidation[list[AIMessage | Callable[[list[BaseMessage]], AIMessage]]]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)
    # 모델이 받는 도구 스키마(이름·설명·인자)
    bound_tools: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[dict | type[BaseModel] | Callable | BaseTool],
                   **kwargs: object) -> ScriptedChatModel:
        self.bound_tools = [convert_to_openai_tool(t) for t in tools]
        self.bound_tool_names = [t["function"]["name"] for t in self.bound_tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: object = None, **kwargs: object) -> ChatResult:
        index = len(self.calls)
        self.calls.append(list(messages))
        if index >= len(self.steps):
            raise ScriptExhaustedError(
                f"대본은 {len(self.steps)}단계인데 {index + 1}번째 호출이 왔습니다.")
        step = self.steps[index]
        message = step if isinstance(step, AIMessage) else step(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])


class StreamingChatModel(ScriptedChatModel):
    """단계마다 AIMessageChunk 목록을 차례로 스트리밍한다.
    callable 단계는 그때까지의 메시지를 받아 청크 목록을 만든다.

    messages 스트림 모드에서는 langchain-core 가 _stream 을 부르고 청크마다 on_llm_new_token 을
    낸다(실제 공급자 모델과 같다).
    """
    steps: SkipValidation[list[list[AIMessageChunk]
                               | Callable[[list[BaseMessage]], list[AIMessageChunk]]]]

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None,
                run_manager: object = None, **kwargs: object) -> Iterator[ChatGenerationChunk]:
        index = len(self.calls)
        self.calls.append(list(messages))
        if index >= len(self.steps):
            raise ScriptExhaustedError(
                f"대본은 {len(self.steps)}단계인데 {index + 1}번째 호출이 왔습니다.")
        step = self.steps[index]
        for chunk in step if isinstance(step, list) else step(messages):
            yield ChatGenerationChunk(message=chunk)


def tool_call(name: str, args: dict, call_id: str | None = None) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id or f"call-{uuid4().hex[:8]}",
         "type": "tool_call"}])


def tool_calls(*calls: tuple[str, dict]) -> AIMessage:
    """병렬 도구 호출 하나의 응답."""
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call-{uuid4().hex[:8]}", "type": "tool_call"}
        for name, args in calls])
