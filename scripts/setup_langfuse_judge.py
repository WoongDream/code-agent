"""Langfuse LLM-as-a-Judge 설정: turn 루트 observation 을 평가하는 evaluator 와 rule 을 만든다. 다시 돌려도 중복 생성하지 않는다.

먼저 Langfuse 프로젝트에 LLM Connection(OpenRouter, custom model anthropic/claude-sonnet-5.5)을 등록해야 한다.
사용: uv run --extra langfuse python scripts/setup_langfuse_judge.py  (키 값은 출력하지 않는다)
"""
import sys

from code_agent.settings import load_settings
from code_agent.tracing import make_tracer

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
NAME = "code-agent turn quality"

SYSTEM = """너는 코딩 에이전트의 한 턴을 평가하는 엄격한 리뷰어다.
사용자 요청과 에이전트의 작업 기록(최종 답변, 실행한 도구와 결과 요약, 승인 결정, 바뀐 파일, diff)만 근거로 판단한다.
작업 기록에 없는 것은 하지 않은 것으로 본다. 에이전트가 최종 답변에서 주장하는 내용은 작업 기록으로 확인될 때만 믿는다."""

USER = """## 사용자 요청
{{input}}

## 에이전트 작업 기록
{{output}}

다음 네 기준을 각각 판정해라.
1. 요청 완수: 요청한 기능이나 변경이 바뀐 파일과 diff 에 실제로 들어 있는가. 질문이었다면 답이 질문에 맞는가.
2. 검증: 테스트나 실행으로 결과를 확인했는가. 확인할 수 없는 작업이면 그 사실을 답변에서 밝혔는가.
3. 범위: 요청하지 않은 변경(불필요한 파일, 리팩터링, 설정 변경)이 없는가.
4. 보고 정확성: 최종 답변의 주장이 작업 기록과 맞는가. 하지 않은 일을 했다고 말하지 않았는가.

- 인사나 질문처럼 파일 작업이 필요 없는 턴은 기준 2와 3을 통과로 본다.
- 사용자가 승인을 거부한 것은 에이전트의 잘못이 아니다. 다만 거부된 시도를 그대로 반복했다면 감점한다.
- 점수: 네 기준을 모두 만족하면 1.0, 기준 하나를 못 지킬 때마다 0.25 씩 뺀다. 요청 완수에 실패하면 최대 0.25 다."""

client = make_tracer(load_settings()).client
api = client.api
connections = api.llm_connections.list().data
match = [c for c in connections if JUDGE_MODEL in (c.custom_models or [])]
if not match:
    print("LLM Connection 이 없다. 등록된 연결:", [(c.provider, c.adapter, c.custom_models) for c in connections])
    sys.exit(1)
provider = match[0].provider
print("LLM Connection:", provider, match[0].adapter, match[0].base_url)

existing = [e for e in api.evaluators.list(limit=100).data if e.name == NAME]
if existing:
    evaluator = existing[0]
    print("evaluator 이미 있음:", evaluator.id)
else:
    evaluator = api.evaluators.create(request={
        "type": "llm_as_judge",
        "name": NAME,
        "description": "turn 하나의 작업 기록으로 요청 완수·검증·범위·보고 정확성을 0~1 로 채점",
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
        "modelConfig": {"provider": provider, "model": JUDGE_MODEL},
        "variableMapping": [{"variable": "input", "source": "input"}, {"variable": "output", "source": "output"}],
        "outputDefinition": {
            "dataType": "NUMERIC", "minValue": 0, "maxValue": 1,
            "scoreReasoningInstructions": "기준 1~4 를 각각 '통과' 또는 '실패: 근거 한 줄' 형식으로 한국어로 쓴다.",
            "scoreValueInstructions": "0 이상 1 이하, 0.25 단위의 숫자.",
        },
    })
    print("evaluator 생성:", evaluator.id)

rules = [r for r in api.evaluation_rules.list(limit=100).data if r.name == NAME]
if rules:
    print("rule 이미 있음:", rules[0].id)
else:
    rule = api.evaluation_rules.create(
        name=NAME, enabled=True, sampling=1.0,
        filter=[{"type": "boolean", "column": "isRootObservation", "operator": "=", "value": True},
                {"type": "stringOptions", "column": "traceName", "operator": "any of",
                 "value": ["code-agent turn"]}],  # observation name 필터는 없다. 트레이스 이름 + 루트로 turn 을 고른다
        evaluator_assignments=[{"evaluatorId": evaluator.id,
                                "variableMapping": [{"variable": "input", "source": "input"},
                                                    {"variable": "output", "source": "output"}]}],
    )
    print("rule 생성:", rule.id)
