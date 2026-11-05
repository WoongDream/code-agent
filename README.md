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

## 시험

```
uv run ruff check . && uv run pytest -q    # 모델 호출 없음
```
