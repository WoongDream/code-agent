"""에이전트 배선: 도구 목록, interrupt, reject/approve 재개, .env 차단, 오프로드 위치.

모델은 대본이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fakes import ScriptedChatModel, tool_call, tool_calls
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from code_agent.agent import build_agent, run_turn
from code_agent.approval import APPROVE, reject
from code_agent.settings import Settings

DONE = AIMessage(content="끝")


class Harness:
    """대본 모델로 에이전트를 만들고 한 턴을 돌린다. 받은 승인 요청과 스트림 조각을 남긴다."""

    def __init__(self, tmp_path: Path, steps: list, decisions: list[dict] | None = None) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir()
        self.artifacts = tmp_path / "data" / "artifacts"
        self.model = ScriptedChatModel(steps=steps)
        self.agent = build_agent(Settings(data_dir=tmp_path / "data"), self.workspace,
                                 InMemorySaver(), model=self.model)
        self.decisions = list(decisions or [])
        self.requests: list[list[dict]] = []
        self.parts: list = []

    def decide(self, action_requests: list[dict]) -> list[dict]:
        self.requests.append(action_requests)
        return [self.decisions.pop(0) for _ in action_requests]

    def run(self, prompt: str = "go") -> None:
        run_turn(self.agent, {"messages": [{"role": "user", "content": prompt}]},
                 {"configurable": {"thread_id": "t"}}, decide=self.decide,
                 on_part=self.parts.append)

    def tool_messages(self) -> list[ToolMessage]:
        return [m for m in self.model.calls[-1] if isinstance(m, ToolMessage)]


def test_exposed_tools(tmp_path: Path) -> None:
    h = Harness(tmp_path, [DONE])
    h.run()
    assert set(h.model.bound_tool_names) == {
        "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute",
        "write_todos"}


def test_system_prompt_has_runtime_info(tmp_path: Path) -> None:
    h = Harness(tmp_path, [DONE])
    h.run()
    system = h.model.calls[0][0].text
    assert str(h.workspace) in system
    assert "/.code-agent-artifacts/" in system
    assert "생각(thinking)을 포함해 64,000토큰까지" in system  # 출력 한도와 파일에 나눠 쓰기 규칙
    # 한 사건의 예시 대신 일반 원칙(D7)
    assert "파일에 하나뿐인 표식" in system and "END-OF-DATA" not in system
    # env_text 를 주지 않으면 절이 없다(옛 thread 와 같은 system)
    assert "이 PC 의 도구" not in system


def test_system_prompt_has_env_section(tmp_path: Path) -> None:
    env_text = "\n# 이 PC 의 도구 (세션 시작 때 확인했다)\n- 있음: uv 0.13.7\n"
    model = ScriptedChatModel(steps=[DONE])
    (tmp_path / "ws").mkdir()
    agent = build_agent(Settings(data_dir=tmp_path / "data"), tmp_path / "ws", InMemorySaver(),
                        model=model, env_text=env_text)
    run_turn(agent, {"messages": [{"role": "user", "content": "go"}]},
             {"configurable": {"thread_id": "t"}},
             decide=lambda requests: [], on_part=lambda part: None)
    assert env_text in model.calls[0][0].text


def test_execute_interrupts_with_approve_or_reject_only(tmp_path: Path) -> None:
    h = Harness(tmp_path, [tool_call("execute", {"command": "echo hi"}), DONE], [APPROVE])
    h.run()
    [[request]] = h.requests
    assert (request["name"], request["args"]["command"]) == ("execute", "echo hi")


def test_reject_does_not_run_command(tmp_path: Path) -> None:
    h = Harness(tmp_path, [tool_call("execute", {"command": "echo x > made.txt"}), DONE],
                [reject("안 돼")])
    h.run()
    assert not (h.workspace / "made.txt").exists()
    [message] = h.tool_messages()
    assert "거부" in message.text and "안 돼" in message.text


def test_approve_runs_command_and_finishes(tmp_path: Path) -> None:
    h = Harness(tmp_path, [tool_call("execute", {"command": "echo x > made.txt"}), DONE],
                [APPROVE])
    h.run()
    assert (h.workspace / "made.txt").exists()
    assert len(h.model.calls) == 2  # 재개 후 마지막 응답까지 갔다


def test_parallel_requests_are_decided_in_order(tmp_path: Path) -> None:
    h = Harness(tmp_path, [tool_calls(("write_file", {"file_path": "/a.txt", "content": "A"}),
                                      ("write_file", {"file_path": "/b.txt", "content": "B"})),
                           DONE],
                [APPROVE, reject()])
    h.run()
    assert [r["args"]["file_path"] for r in h.requests[0]] == ["/a.txt", "/b.txt"]
    assert (h.workspace / "a.txt").read_text(encoding="utf-8") == "A"
    assert not (h.workspace / "b.txt").exists()


def test_edit_file_after_approval(tmp_path: Path) -> None:
    h = Harness(tmp_path, [tool_call("read_file", {"file_path": "/app.py"}),
                           tool_call("edit_file", {"file_path": "/app.py", "old_string": "1 + 1",
                                                   "new_string": "1 + 2"}), DONE], [APPROVE])
    (h.workspace / "app.py").write_text("x = 1 + 1\n", encoding="utf-8")
    h.run()
    assert (h.workspace / "app.py").read_text(encoding="utf-8") == "x = 1 + 2\n"
    assert [r["name"] for [r] in h.requests] == ["edit_file"]  # read_file 은 묻지 않는다


@pytest.mark.parametrize("path", ["/.env", "/sub/.env.local"])
def test_env_file_read_is_denied(tmp_path: Path, path: str) -> None:
    h = Harness(tmp_path, [tool_call("read_file", {"file_path": path}), DONE])
    target = h.workspace / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("API_KEY=sk-secret-value\n", encoding="utf-8")
    h.run()
    [message] = h.tool_messages()
    assert "권한 거부" in message.text
    assert all("sk-secret-value" not in str(m.content) for m in h.model.calls[-1])


def test_env_example_is_readable(tmp_path: Path) -> None:
    h = Harness(tmp_path, [tool_call("read_file", {"file_path": "/.env.example"}), DONE])
    (h.workspace / ".env.example").write_text("API_KEY=\n", encoding="utf-8")
    h.run()
    assert "API_KEY=" in h.tool_messages()[0].text


@pytest.mark.parametrize("mode", ["content", "files_with_matches", "count"])
def test_grep_hides_env_matches(tmp_path: Path, mode: str) -> None:
    h = Harness(tmp_path, [tool_call("grep", {"pattern": "TOKEN", "output_mode": mode}), DONE])
    (h.workspace / ".env").write_text("TOKEN=sk-secret-value\n", encoding="utf-8")
    (h.workspace / "app.py").write_text("TOKEN = load()\n", encoding="utf-8")
    h.run()
    text = h.tool_messages()[0].text
    assert "/app.py" in text
    assert ".env" not in text and "sk-secret-value" not in text


def test_grep_filter_runs_before_offload(tmp_path: Path) -> None:
    """가드가 오프로드보다 안쪽에 있어야
    큰 grep 결과의 오프로드 파일에도 .env 내용이 남지 않는다.
    """
    h = Harness(tmp_path, [tool_call("grep", {"pattern": "TOKEN", "output_mode": "content"}),
                           DONE])
    (h.workspace / ".env").write_text(
        "".join(f"TOKEN_{i}=sk-secret-value\n" for i in range(5000)), encoding="utf-8")
    (h.workspace / "app.py").write_text("TOKEN = load()\n", encoding="utf-8")
    h.run()
    assert "sk-secret-value" not in h.tool_messages()[0].text
    offloaded = h.artifacts / "large_tool_results"
    assert not offloaded.exists() or all("sk-secret-value" not in p.read_text(encoding="utf-8")
                                         for p in offloaded.iterdir())


def test_large_output_offloads_to_artifacts(tmp_path: Path) -> None:
    """큰 결과 파일은 ~/.code-agent/artifacts 에 생기고 작업 폴더에는 아무것도 생기지 않는다."""
    exe = Path(sys.executable).as_posix()
    h = Harness(tmp_path,
                [tool_call("execute", {"command": f"'{exe}' -c \"print('x' * 200000)\""}), DONE],
                [APPROVE])
    h.run()
    assert list(h.workspace.iterdir()) == []
    offloaded = list((h.artifacts / "large_tool_results").iterdir())
    assert len(offloaded) == 1
    assert "/.code-agent-artifacts/large_tool_results/" in h.tool_messages()[0].text


def test_stream_parts_are_namespaced_tuples(tmp_path: Path) -> None:
    h = Harness(tmp_path, [DONE])
    h.run()
    assert h.parts and all(isinstance(ns, tuple) and mode in ("messages", "updates", "custom")
                           for ns, mode, _ in h.parts)
