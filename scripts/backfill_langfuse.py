"""Langfuse 연동 전에 돌린 세션을 sessions.db 에서 꺼내 turn 작업 기록으로 다시 올린다. 올리면 LLM-as-a-Judge rule 이 채점한다.

- 턴은 사용자 메시지 단위다. 실행한 도구와 결과 요약, 거부된 승인, 최종 답변, 사용량은 대화 기록에서 그대로 복원한다.
- 자동/사람 승인 구분은 기록에 없어 '승인(소급)' 으로 남긴다.
- 바뀐 파일과 diff 는 기준 폴더(기본: 빈 폴더)와 지금 작업 폴더를 비교해, 파일을 바꾼 마지막 턴에 몰아 넣는다.
  세션 뒤에 사람이 폴더를 고쳤다면 그 변경도 섞인다.
- 같은 세션을 이미 올렸으면 건너뛴다(tag backfill). 기존 트레이스는 지우지 않는다.

사용: uv run --extra langfuse python scripts/backfill_langfuse.py [--dry-run] [--baseline THREAD=DIR ...] [THREAD ...]
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from code_agent.agent import INTERRUPT_TOOLS, build_agent
from code_agent.approval import REJECT_MESSAGE
from code_agent.render import describe_call
from code_agent.settings import load_settings
from code_agent.tracing import make_tracer
from code_agent.usage import Usage
from code_agent.worklog import WorkLog, diff_snapshots, one_line, snapshot

FILE_TOOLS = {"write_file", "edit_file", "delete"}
TAG = "backfill"


def load_turns(conn: sqlite3.Connection, thread_id: str, workspace: Path) -> list[list]:
    """대화 기록을 사용자 메시지 단위로 나눈다."""
    agent = build_agent(load_settings(), workspace, SqliteSaver(conn), model=GenericFakeChatModel(messages=iter([])))
    messages = agent.get_state({"configurable": {"thread_id": thread_id}}).values.get("messages", [])
    turns: list[list] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    return turns


def build_record(turn: list, *, diff: tuple[list[str], str] | None) -> dict:
    log, usage, models, answer = WorkLog(), Usage(), set(), ""
    tool_calls: dict[str, dict] = {}
    for message in turn[1:]:
        if isinstance(message, AIMessage):
            usage.add_message(message)
            models.add(message.response_metadata.get("model_name") or "")
            log.on_part(((), "updates", {"model": {"messages": [message]}}))
            tool_calls.update({call["id"]: call for call in message.tool_calls})
            if message.text.strip():
                answer = message.text
        elif isinstance(message, ToolMessage):
            log.on_part(((), "updates", {"tools": {"messages": [message]}}))
            call = tool_calls.get(message.tool_call_id)
            if call and call["name"] in INTERRUPT_TOOLS:
                rejected = REJECT_MESSAGE in message.text
                reason = message.text.split("사용자 의견: ", 1)[1] if "사용자 의견: " in message.text else ""
                verdict = f"거부(사용자){': ' + reason if reason else ''}" if rejected else "승인(소급)"
                log.approvals.append(one_line(f"{describe_call(call)} → {verdict}"))
    changes, diff_text = diff if diff is not None else ([], "")
    return {"input": turn[0].text, "output": log.render(answer=answer, changes=changes, diff=diff_text),
            "usage": usage, "models": sorted(m for m in models if m)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("threads", nargs="*", help="올릴 세션 ID (기본: 전부)")
    parser.add_argument("--baseline", action="append", default=[], metavar="THREAD=DIR",
                        help="세션 시작 때 작업 폴더 내용과 같은 폴더 (기본: 빈 폴더)")
    parser.add_argument("--dry-run", action="store_true", help="보내지 않고 작업 기록만 출력한다")
    args = parser.parse_args()
    baselines = dict(item.split("=", 1) for item in args.baseline)

    settings = load_settings()
    db = (settings.data_dir / "sessions.db").as_posix()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
    rows = conn.execute("SELECT thread_id, workspace, created_at FROM code_agent_threads ORDER BY created_at").fetchall()
    tracer = None if args.dry_run else make_tracer(settings)
    if not args.dry_run and tracer is None:
        raise SystemExit("Langfuse 키나 langfuse extra 가 없다.")
    api = tracer.client.api if tracer else None
    sent: list[tuple[str, int, str]] = []

    for thread_id, ws, created in rows:
        if args.threads and thread_id not in args.threads:
            continue
        workspace = Path(ws)
        if api is not None and any(TAG in (o.tags or []) for o in api.observations.get_many(
                session_id=thread_id, name="turn", fields="core,basic", limit=50).data):
            print(f"{thread_id}: 이미 올렸다. 건너뛴다")
            continue
        turns = load_turns(conn, thread_id, workspace)
        file_turns = [i for i, turn in enumerate(turns) if any(
            isinstance(m, AIMessage) and any(c["name"] in FILE_TOOLS for c in m.tool_calls) for m in turn)]
        after = snapshot(workspace) if workspace.exists() else None
        before = snapshot(Path(baselines[thread_id])) if thread_id in baselines else {}
        diff = diff_snapshots(before, after) if after is not None and before is not None else None
        print(f"== {thread_id} ({workspace.name}, {created[:16]}) 턴 {len(turns)}개, 파일 변경 턴 {file_turns}")
        for index, turn in enumerate(turns):
            record = build_record(turn, diff=diff if file_turns and index == file_turns[-1] else ([], ""))
            if args.dry_run:
                print(f"\n--- 턴 {index + 1}: {record['input'][:60]!r}\n{record['output'][:2500]}")
                continue
            from langfuse import propagate_attributes

            tags = [workspace.name[:200], TAG, *(f"model:{m}" for m in record["models"])]
            with (tracer.client.start_as_current_observation(name="turn", as_type="agent", input=record["input"]) as root,
                  propagate_attributes(session_id=thread_id, trace_name="code-agent turn", tags=tags)):
                root.update(output=record["output"], metadata={
                    "backfill": True, "original_session_created_at": created, "turn_index": index + 1,
                    "usage": {k: getattr(record["usage"], k) for k in ("calls", "input_tokens", "output_tokens", "cost")}})
                sent.append((thread_id, index + 1, root.trace_id))
    if tracer is None:
        return
    tracer.client.flush()
    print(f"\n{len(sent)}개 turn 을 올렸다. 평가를 기다린다...")
    pending = {(t, i): trace_id for t, i, trace_id in sent}
    for _ in range(60):
        for key, trace_id in list(pending.items()):
            scores = api.scores.get_many(trace_id=trace_id, limit=5, fields="details").data
            if scores:
                score = scores[0]
                print(f"\n[{key[0]} 턴 {key[1]}] 점수 {score.value}\n{score.comment}")
                del pending[key]
        if not pending:
            break
        time.sleep(5)
    for (thread_id, index), trace_id in pending.items():
        print(f"[{thread_id} 턴 {index}] 아직 점수가 없다 (trace {trace_id})")


if __name__ == "__main__":
    main()
