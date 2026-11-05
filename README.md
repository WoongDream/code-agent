# code-agent

LangChain [Deep Agents](https://github.com/langchain-ai/deepagents) SDK(`deepagents==0.7.32`)로 만드는 터미널 코딩 에이전트. 계획과 진행 상황은 [PLAN.md](PLAN.md)에 있다.

## 설치

```
uv sync                      # OpenRouter 모델까지 쓰려면 uv sync --extra openrouter
cp .env.example .env         # ANTHROPIC_API_KEY 채우기 (또는 ~/.code-agent/.env)
```

## 사용

```
uv run code-agent --workspace <작업 폴더> [--mode manual|auto-edit|yolo] [--model anthropic:<id>]
uv run code-agent --resume <세션 ID>
```

- 파일 도구는 작업 폴더 안에 갇히지만 **셸은 샌드박스가 아니다**. 그래서 기본 `manual` 모드는 파일 쓰기와 셸 명령을 모두 묻는다.
- `auto-edit`는 파일 쓰기를 자동 승인하고 셸은 묻는다(`.git/` 아래 쓰기는 묻는다). `yolo`는 전부 자동 승인한다.
- 읽기 전용 명령(`git status`, `git diff`, `ls`, `cat` 등)은 모드와 관계없이 자동 승인한다. 단 작업 폴더 밖, `.env`, 명령 연결이 들어가면 묻는다.
- 슬래시 커맨드: `/help`, `/new`, `/mode <모드>`, `/diff`, `/todos`, `/cost`, `/exit`. Ctrl+C 는 실행 중인 턴만 취소한다.
- 턴이 끝날 때마다 모델 호출 수·토큰(캐시 읽기/쓰기)·비용을 한 줄로 보이고, `/cost` 는 세션 합계를 보인다. 비용은 OpenRouter 가 알려 준 값이다.
- OpenRouter 의 Anthropic 모델은 프롬프트 캐시(5분 TTL)를 자동으로 켠다.
- 세션과 큰 도구 결과는 `~/.code-agent/`(`sessions.db`, `artifacts/`)에 저장해 작업 폴더를 더럽히지 않는다.

## Langfuse 트레이스 (선택)

```
uv sync --extra openrouter --extra langfuse
```

`.env` 에 `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`(리전 주소)을 넣으면 켜진다. 시작 화면에 `Langfuse 트레이스 켜짐` 이 보인다.

- 사용자 메시지 하나가 트레이스 하나(`turn`)다. 세션 ID 는 code-agent 세션 ID 이고, 태그는 작업 폴더 이름·승인 모드·모델이다.
- 그 아래에 그래프 노드, 모델 호출(입력 전체·출력·도구 선택·reasoning 텍스트·토큰), 도구 실행(인자·결과)이 남는다. 승인 대기 후 재개도 같은 트레이스로 이어진다.
- 승인 결정은 `approval:<도구>` event 로 남는다(호출 요약, approve/reject, 자동인지 사람이 정했는지, 거부 사유). 거부는 WARNING 이다.
- `turn` 출력에는 최종 답변과 함께 작업 기록(실행한 도구와 결과 요약, 승인 결정, 바뀐 파일, diff)이 남는다. Langfuse LLM-as-a-Judge 평가기는 같은 트레이스의 다른 observation 을 읽지 못하므로, `turn` 하나만 보고 채점할 수 있게 하려는 것이다. `.env` 는 diff 에 넣지 않는다.
- 중단(Ctrl+C)과 오류는 `turn` 을 ERROR 로 표시한다.
- 에이전트가 읽은 코드와 명령 출력이 Langfuse 로 전송된다. `.env` 내용은 가드가 막는다.
- **비용**: Langfuse 는 모델 정의로 비용을 계산한다. `anthropic/claude-opus-5.5` 같은 OpenRouter 이름은 기본 정의에 없어 Project Settings > Models 에 가격을 등록해야 한다(사용량 키: `input`, `output`, `output_reasoning`, `input_cache_read`, `input_cache_creation`). OpenRouter 가 알려 준 실제 비용은 `turn` 의 metadata `usage` 와 `/cost` 에 있다.

## 시험

```
uv run ruff check . && uv run pytest -q    # 모델 호출 없음
```
