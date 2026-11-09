"""deepagents·langchain 기본 문구 중 우리 환경과 맞지 않는 것을 고친 글(task11 D1, D2).

도구 설명은 HarnessProfile.tool_description_overrides 로 넣는다(agent._register_harness_profile).
- execute: 기본 설명은 '격리된 sandbox', '절대 경로를 쓰고 cd 를 피하라',
  '; 와 && 로 연결하라'고 한다. 우리 셸은 격리되지 않고,
  작업 폴더가 현재 디렉터리이며(system 의 '실행 환경' 절과 반대였다),
  연결은 승인 대상이고 PowerShell 5.1 에는 && 가 없다.
- ls: 'almost ALWAYS' 때문에 빌드 세션 셋 모두 첫 호출이 ls / 였다.
- grep: 오프로드 경로가 우리 artifacts 경로와 다르다.
- glob: 예시로 .env 를 든다(비밀 파일 정책과 맞지 않는다).
override 는 도구 설명만 바꾼다. execute 의 command 인자 설명('in the sandbox environment')은
ToolArgsMiddleware 가 모델에 가는 요청의 도구 목록에서 고친다.
도구 설명과 할 일 절은 deepagents 의 다른 글과 같이 영어로 둔다.
내용은 system 의 '실행 환경' 절과 맞춘다.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from deepagents.middleware.filesystem import (
    GLOB_TOOL_DESCRIPTION,
    GREP_TOOL_DESCRIPTION,
    ExecuteSchema,
)
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import BaseTool
from pydantic import Field

from code_agent.backend import ARTIFACTS_ROUTE

EXECUTE_DESCRIPTION = """\
Runs a shell command on the user's machine (not sandboxed) with the workspace as the current \
directory, and returns stdout/stderr and the exit code. \
Large output is truncated or moved to a file whose path is returned.
- The shell is the one described under 실행 환경 (Git Bash or Windows PowerShell 5.1). \
Use paths relative to the workspace; quote paths that contain spaces.
- Commands may need the user's approval. \
Read-only commands (e.g. ls, cat, git status/diff/log/show; in Git Bash also head, tail, wc) \
without chaining, redirection or paths outside the workspace run without asking, \
so prefer one simple command per call over chains.
- Default timeout is 120 s; pass `timeout` (seconds) for longer runs such as test suites.
- To search or read files, the grep, glob and read_file tools are cheaper and need no approval."""

LS_DESCRIPTION = ("Lists files and directories at a path (the virtual root / is the workspace). "
                  "Use it to orient in an unfamiliar directory; "
                  "when the exact path is already known, read or edit it directly.")

# deepagents 0.7.32 글에서 우리와 다른 부분만 바꾼다. 원문이 바뀌면 시험(test_prompts)이 잡는다
GREP_DESCRIPTION = GREP_TOOL_DESCRIPTION.replace("(`/large_tool_results/` by default)",
                                                 f"(`{ARTIFACTS_ROUTE}large_tool_results/`)")
GLOB_DESCRIPTION = GLOB_TOOL_DESCRIPTION.replace("(use `.env`, or `.github/**/*.yml`)",
                                                 "(e.g. `.github/**/*.yml`)")

TOOL_DESCRIPTIONS = {"execute": EXECUTE_DESCRIPTION, "ls": LS_DESCRIPTION,
                     "grep": GREP_DESCRIPTION, "glob": GLOB_DESCRIPTION}


class ExecuteArgs(ExecuteSchema):
    """execute 인자.

    deepagents 의 command 설명은 'Shell command to execute in the sandbox environment.' 다.
    """

    command: str = Field(description="Shell command to run in the workspace directory.")


class ToolArgsMiddleware(AgentMiddleware):
    """모델에 가는 요청의 도구 목록에서 execute 를 인자 설명만 고친 사본으로 바꾼다.

    langchain 은 요청의 도구를 이름으로만 확인하고 실행은 원래 도구가 하므로(인자 필드는 같다)
    실행에는 영향이 없다.
    PromptFingerprintMiddleware 보다 앞(바깥)에 두어 지문이 모델이 실제로 받는 도구를 보게 한다.
    """

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """요청의 execute 도구를 인자 설명만 고친 사본으로 바꿔 넘긴다."""
        return handler(_own_args(request))

    async def awrap_model_call(self, request: ModelRequest,
                               handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
                               ) -> ModelResponse:
        """wrap_model_call 의 비동기판."""
        return await handler(_own_args(request))


def _own_args(request: ModelRequest) -> ModelRequest:
    tools = [t.model_copy(update={"args_schema": ExecuteArgs})
             if isinstance(t, BaseTool) and t.name == "execute" and t.args_schema is ExecuteSchema
             else t
             for t in request.tools]
    return request.override(tools=tools)


# langchain TodoListMiddleware 기본 절의 'Finishing a task'
# (마지막 write_todos 뒤 별도 메시지로 답하라) 때문에 할 일을 끝냄으로 바꾸기만 하는 호출이
# 한 번 더 생겼다(az-quiz [47], 문맥 120K 에 $0.03~0.59)
TODO_PROMPT = """\
## write_todos
Use write_todos for work with several distinct steps the user would want to track; \
skip it for small tasks. Update it when the plan changes or a step finishes, \
together with the next tool call in the same response rather than as a call of its own. \
Do not make a separate call just to mark the last items complete: \
finish with the final answer, which should start with the result the user asked for."""
