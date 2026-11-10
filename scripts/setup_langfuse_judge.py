"""Langfuse LLM-as-a-Judge 설정: turn 루트 observation 을 평가하는 evaluator 와 rule 을 만든다.

다시 돌려도 중복 생성하지 않는다.
evaluator 가 이미 있으면 아래 정의로 바꿔 새 버전을 만든다(rule 은 최신 버전을 쓴다).
먼저 Langfuse 프로젝트에 LLM Connection(OpenRouter, custom model anthropic/claude-sonnet-5.5)을
등록해야 한다.
사용: uv run --extra langfuse python scripts/setup_langfuse_judge.py  (키 값은 출력하지 않는다)
"""
import sys

from code_agent.settings import load_settings
from code_agent.tracing import make_tracer

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
NAME = "code-agent turn quality"

SYSTEM = """너는 코딩 에이전트의 한 턴을 평가하는 엄격한 리뷰어다.
사용자 요청과 에이전트의 작업 기록\
(최종 답변, 실행한 도구와 결과 요약, 승인 결정, 바뀐 파일, diff, 맥락)만 근거로 판단한다.
작업 기록에 없는 작업은 하지 않은 것으로 본다. \
에이전트가 최종 답변에서 주장하는 내용은 작업 기록이나 맥락으로 확인될 때만 믿는다."""

USER = """## 사용자 요청
{{input}}

## 에이전트 작업 기록
{{output}}

다음 네 기준을 각각 판정해라.
1. 요청 완수: 요청한 기능이나 변경이 바뀐 파일과 diff 에 실제로 들어 있는가. \
질문이었다면 답이 질문에 맞는가.
2. 검증: 테스트나 실행으로 결과를 확인했는가. 확인할 수 없는 작업이면 그 사실을 답변에서 밝혔는가.
3. 범위: 요청하지 않은 변경(불필요한 파일, 리팩터링, 설정 변경)이 없는가.
4. 보고 정확성: 최종 답변의 주장이 작업 기록이나 맥락과 맞는가. \
하지 않은 일을 했다고 말하지 않았는가.

- 작업 기록 끝의 "맥락" 절은 에이전트가 이 턴 전에 이미 알고 있던 것이다\
(시스템 프롬프트의 작업 폴더, 같은 세션의 이전 턴).
  맥락과 일치하는 말(작업 폴더 경로, 이전 턴에 만든 파일 이름 등)은 \
이번 턴에 도구를 쓰지 않았어도 근거 있는 주장으로 본다.
- 인사나 질문처럼 파일 작업이 필요 없는 턴은 기준 2와 3을 통과로 본다.
- 사용자가 승인을 거부한 것은 에이전트의 잘못이 아니다. \
다만 거부된 시도를 그대로 반복했다면 감점한다.
- 사소한 부정확함(줄 번호가 한 줄 어긋남 등)은 근거에 적되 실패로 보지 않는다."""

REASONING = ("기준 1~4 를 각각 '기준 N: 통과 - 근거' 또는 '기준 N: 실패 - 근거' "
             "한 줄로 한국어로 쓴다. 마지막 줄에 '실패 K개 → 점수 S' 를 쓴다.")
VALUE = ("이유에서 '실패'로 판정한 기준 수를 K 라 할 때 점수 S = 1 - 0.25 × K 다. "
         "단 기준 1(요청 완수)이 실패면 S 는 최대 0.25 다. "
         "점수는 반드시 이유의 판정·계산과 같아야 한다. 모두 통과면 1.0 이다.")
MAPPING = [{"variable": "input", "source": "input"}, {"variable": "output", "source": "output"}]


def main() -> None:
    """judge 모델의 LLM Connection 을 확인하고 evaluator 와 rule 을 만들거나 갱신한다."""
    tracer = make_tracer(load_settings())
    if tracer is None:
        sys.exit("Langfuse 키나 langfuse extra 가 없다.")
    client = tracer.client
    api = client.api
    connections = api.llm_connections.list().data
    match = [c for c in connections if JUDGE_MODEL in (c.custom_models or [])]
    if not match:
        print("LLM Connection 이 없다. 등록된 연결:",
              [(c.provider, c.adapter, c.custom_models) for c in connections])
        sys.exit(1)
    provider = match[0].provider
    print("LLM Connection:", provider, match[0].adapter, match[0].base_url)

    definition = {
        "type": "llm_as_judge",
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
        "modelConfig": {"provider": provider, "model": JUDGE_MODEL},
        "variableMapping": MAPPING,
        "outputDefinition": {"dataType": "NUMERIC", "minValue": 0, "maxValue": 1,
                             "scoreReasoningInstructions": REASONING,
                             "scoreValueInstructions": VALUE},
    }
    description = "turn 하나의 작업 기록으로 요청 완수·검증·범위·보고 정확성을 0~1 로 채점"
    existing = [e for e in api.evaluators.list(limit=100).data if e.name == NAME]
    if existing:
        evaluator = api.evaluators.update(existing[0].id,
                                          request={**definition, "description": description})
        print("evaluator 정의 갱신:", evaluator.id)
    else:
        evaluator = api.evaluators.create(
            request={**definition, "name": NAME, "description": description})
        print("evaluator 생성:", evaluator.id)

    rules = [r for r in api.evaluation_rules.list(limit=100).data if r.name == NAME]
    if rules:
        print("rule 이미 있음:", rules[0].id)
    else:
        # observation name 필터는 없다. 트레이스 이름 + 루트로 turn 을 고른다
        rule = api.evaluation_rules.create(
            name=NAME, enabled=True, sampling=1.0,
            filter=[{"type": "boolean", "column": "isRootObservation", "operator": "=",
                     "value": True},
                    {"type": "stringOptions", "column": "traceName", "operator": "any of",
                     "value": ["code-agent turn"]}],
            evaluator_assignments=[{"evaluatorId": evaluator.id, "variableMapping": MAPPING}],
        )
        print("rule 생성:", rule.id)


if __name__ == "__main__":
    main()
