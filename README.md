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
- 슬래시 커맨드: `/help`, `/new`, `/mode <모드>`, `/plan [<요청>|off]`, `/remember [메모]`, `/memory`, `/improve [--no-eval] [--tasks a,b]`, `/diff`, `/todos`, `/cost`, `/thinking [on|off]`, `/exit`. Ctrl+C 는 실행 중인 턴만 취소한다.
- **계획 모드**: 여러 파일을 바꾸거나 요구사항이 모호한 작업이면 에이전트가 스스로 계획을 세운다. `/plan <요청>` 은 항상 계획부터 세운다.
  - 순서: 읽기 전용 조사 → `write_plan`(목표, 요구사항, 범위, 단계별 파일·확인 방법, 완료 조건, 위험) → 리뷰어 서브에이전트 `plan-reviewer` 검토(읽기 전용 도구만, 계획 하나에 2회까지) → 반영 → 사용자 승인(`[y] 승인 / [n] 거부(의견)`, 모든 모드에서 묻는다) → 단계가 할 일이 되고 단계대로 구현 → 완료 조건마다 확인 결과 보고.
  - 계획이 승인되기 전에는 파일 쓰기·수정·삭제와 읽기 전용이 아닌 셸 명령이 막힌다. 검토를 받지 않은 승인 요청은 묻지 않고 돌려보낸다.
  - 승인 뒤 계획에 없는 파일을 쓰려면 auto-edit·yolo 에서도 묻고(이유 "계획에 없는 파일"), 승인하면 계획 범위에 더해 기록한다. 셸 명령으로 바꾸는 파일(`sed -i` 등)은 추적하지 않는다.
  - `/plan` 은 현재 계획·상태·검토·범위 변경을 보이고, `/plan off` 는 계획 모드와 계획을 지운다. 계획은 세션에 저장돼 `--resume` 해도 이어진다.
  - 리뷰어 모델은 `CODE_AGENT_REVIEWER_MODEL`(비우면 `CODE_AGENT_MODEL`)이다.
- **메모리**: 세션을 시작할 때 `AGENTS.md` 두 개를 한 번 읽어 시스템 프롬프트 끝에 붙인다.
  - 모든 프로젝트 공통은 `~/.code-agent/memory/AGENTS.md`(에이전트에게는 `/.code-agent-user/AGENTS.md`), 프로젝트 것은 작업 폴더의 `AGENTS.md` 다. 없으면 만들지 않는다.
  - 프로젝트 `AGENTS.md` 는 처음 볼 때(또는 내용이 바뀌었을 때) 앞부분을 보여 주고 메모리로 읽을지 묻는다(`[y/N]`). 신뢰한 내용은 `sessions.db` 에 남아 다시 묻지 않는다. 파이프 입력처럼 물을 수 없으면 읽지 않는다.
  - `/remember [메모]` 는 이번 대화에서 확인된 규칙·사실을 짧은 항목으로 덧붙이게 한다. 에이전트가 스스로 적기도 한다. `/.code-agent-user/` 아래 쓰기는 yolo 를 포함한 모든 모드에서 묻고, 프로젝트 `AGENTS.md` 는 일반 파일처럼 모드를 따른다. 메모리 파일은 계획 승인 전 차단과 '계획에 없는 파일' 재승인에서 빠진다. 셸 명령으로 바꾸는 것은 막지 못한다.
  - 고친 메모리는 **다음 `/new` 부터** 반영된다. 대화 중에 다시 읽으면 시스템 프롬프트가 바뀌어 프롬프트 캐시와 이전 생각이 깨지기 때문이다. `--resume` 은 그 세션이 읽은 것을 그대로 쓴다.
  - `/memory` 는 파일 위치, 이 세션이 읽은 글자 수, 그 뒤 디스크에서 바뀌었는지, 스킬 목록을 보인다. 합계가 8,000자를 넘으면 시작과 `/new` 때 노란 줄로 경고한다(매 요청 비용이 는다).
  - 스킬: `~/.code-agent/memory/skills/<이름>/SKILL.md` 와 작업 폴더의 `.agents/skills/<이름>/SKILL.md`(같은 이름이면 프로젝트가 이긴다). 이름과 설명만 시스템 프롬프트에 들어가고 본문은 필요할 때 읽는다.
  - 프로젝트 스킬(`.agents/skills/`)도 프로젝트 `AGENTS.md` 와 같은 신뢰 결정을 따른다. 설명이 시스템 프롬프트에 들어가므로 한 번에 묻고("스킬 N개 포함"), 스킬이 바뀌어도 다시 묻는다. 신뢰하지 않으면 읽지 않는다.
  - 셸 명령으로 메모리 폴더를 바꾸는 것은 막지 못한다(셸은 샌드박스가 아니다). 승인·면제 규칙은 파일 도구(write_file·edit_file·delete)만 다룬다.
- 모델과 도구를 기다리는 동안 상태 줄을 보인다. 예: `⠋ 생각 중 1,234자… 42초`, `write_file /quizzes.js 작성 중 12.3KB… 8초`, `$ uv run pytest 실행 중… 35초`. 답변이 나오기 시작하거나 승인을 물을 때는 사라진다. 터미널이 아니면(파이프 등) 그리지 않는다.
- 모델의 생각(reasoning)은 흐리게 보인다. 기본은 문단마다 앞부분만 보이는 짧은 메모이고, `/thinking` 으로 원문 표시를 켜고 끈다.
- 턴이 끝날 때마다 모델 호출 수·토큰(캐시 읽기/쓰기)·비용을 한 줄로 보이고, `/cost` 는 세션 합계를 보인다. 비용은 OpenRouter 가 알려 준 값이다.
- OpenRouter 의 Anthropic 모델은 프롬프트 캐시(5분 TTL)를 자동으로 켠다.
- 응답이 출력 한도(64,000토큰, 생각 포함)에 걸려 잘렸거나 공급자 오류로 끊겼는데 도구 호출이 없으면, 턴을 끝내지 않고 한 번 다시 요청한다(턴당 1번). 이때 잘린 응답의 생각은 기록에서 빼서 이후 호출이 다시 보내지 않는다. 그래도 멈췄거나, 컨텍스트 창이 찼거나, 예산 상한(모델 100회·도구 300회)에 걸리면 턴 끝에 노란 경고를 보인다. '계속해줘'로 이어 갈 수 있다.
- 대화가 입력 컨텍스트 상한(기본 200,000토큰, `CODE_AGENT_CONTEXT_TOKENS` 로 20,000~1,000,000)의 85% 에 이르면 요약한다. 그 전에는 기록을 고치지 않는다(옛 도구 인자도 자르지 않는다). 프롬프트 캐시와 Opus 5.5 의 보존된 생각을 깨지 않기 위해서다.
- 세션과 큰 도구 결과는 `~/.code-agent/`(`sessions.db`, `artifacts/`)에 저장해 작업 폴더를 더럽히지 않는다.
- 셸은 code-agent 자신의 venv 를 물려받지 않는다(`VIRTUAL_ENV`, `UV` 등과 venv 의 `Scripts` 를 뺀다). 활성화한 venv 안에 code-agent 를 설치해 실행하면 그 venv 도 에이전트 셸에서 빠진다. `python` 이 없거나 Windows 스토어 스텁이면 `uv python find` 의 인터프리터를 쓴다. 이 인터프리터는 PEP 668 표식이 있어 `pip install` 이 막힌다.
- 시작할 때 셸이 쓸 PATH 로 도구·버전·브라우저(python, uv, git, node, npm, deno, py, pwsh, PowerShell, Chrome, Edge)를 확인해 시스템 프롬프트에 적는다(약 1초). 이 내용은 세션마다 저장하고 `--resume` 은 다시 확인하지 않고 그것을 쓴다. 바뀌면 프롬프트 캐시와 이전 생각이 깨지기 때문이다.
- code-agent 를 업데이트해 시스템 프롬프트나 도구 설명이 달라진 뒤 `--resume` 하면(이 기능 전에 만든 세션도), 이전 대화의 생각 기록을 요청에서 빼고 이어 간다. 이전 생각의 서명은 옛 프롬프트에 묶여 있어 그대로 보내면 공급자가 거부하거나 무효인 생각이 가기 때문이다. 세션 기록은 그대로 두고, 처음 한 번 노란 줄로 알린다. 프롬프트 캐시는 한 번 새로 쓴다.

## 자기 개선

세션을 쌓다 보면 같은 실수가 되풀이된다. `/improve` 는 그 흔적에서 메모리·스킬 개선안을 만들고, 미니 평가 세트로 검증한 뒤 **사람이 승인하면** 반영한다.

```
/improve [--no-eval] [--tasks a,b]          # REPL
uv run code-agent --improve [--yes] [--no-eval] [--tasks a,b] [-w <작업 폴더>]
uv run code-agent --eval [--tasks a,b] [--yes]    # 평가 세트만 돌려 표를 본다
```

1. **신호 수집**: 최근 20턴의 로컬 트레이스(`~/.code-agent/runs/`)에서 경고·재시도, 사용자 거부와 이유, 같은 도구 오류 반복(2회 이상), 통과하지 못한 pytest, 높음(high) 지적이 있는 계획 리뷰를 모은다. Langfuse 키가 있으면 낮은 Judge 점수도 본다(실패하면 조용히 건너뜀).
2. **개선안**: 모델이 신호와 현재 메모리를 읽고 `{target, skill_name?, patch, evidence, expected_effect}` 를 최대 3개 만든다. 대상은 사용자·프로젝트 `AGENTS.md` 와 `SKILL.md` 뿐이다.
3. **정적 검사**: 비밀값 모양, `AGENTS.md` 합계 8,000자, 중복 항목, `SKILL.md` frontmatter·BOM, 쓸 경로 화이트리스트.
4. **평가**: 평가 과제 3개(`evals/`)를 현재 메모리(기준)와 개선안을 적용한 사본(후보)으로 각각 돌린다. 후보의 통과 수가 기준 이상이고, 통과에서 실패로 바뀐 과제가 없고, 비용이 +30% 이하여야 채택한다. 평가는 임시 data dir 에서 돌아 `~/.code-agent` 를 건드리지 않는다. 프로젝트 대상 개선안은 "평가 대상 아님"이고 사람이 승인해야 한다.
5. **반영**: diff 를 보이고 y/N 을 묻는다. `--yes` 는 평가에서 채택된 사용자 대상 개선안만 묻지 않고 반영한다. 반영한 메모리는 다음 `/new` 부터 적용된다.

모든 단계는 `~/.code-agent/runs/improve-<id>/trace.jsonl`(`improve_signals`, `improve_proposal`, `improve_eval`, `improve_decision` 사건)과 `~/.code-agent/improvements.jsonl`(개선안, 근거, 평가 결과, 결정)에 남는다. 평가 실행 전에 예상 비용을 보이고 묻는다(`--no-eval` 은 무과금). 평가 세트는 `evals/README.md`, 비용은 `/improve` 1회 약 $2~3, `--eval` 1회 약 $1.1 이다.

`CODE_AGENT_DATA_DIR` 환경변수로 `~/.code-agent` 대신 쓸 폴더를 정할 수 있다(평가·시험의 격리용).

## Langfuse 트레이스 (선택)

```
uv sync --extra openrouter --extra langfuse
```

`.env` 에 `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`(리전 주소)을 넣으면 켜진다. 시작 화면에 `Langfuse 트레이스 켜짐` 이 보인다.

- 사용자 메시지 하나가 트레이스 하나(`turn`)다. 세션 ID 는 code-agent 세션 ID 이고, 태그는 작업 폴더 이름·승인 모드·모델이다.
- 그 아래에 그래프 노드, 모델 호출(입력 전체·출력·도구 선택·reasoning 텍스트·토큰), 도구 실행(인자·결과)이 남는다. 승인 대기 후 재개도 같은 트레이스로 이어진다.
- 승인 결정은 `approval:<도구>` event 로 남는다(호출 요약, approve/reject, 자동인지 사람이 정했는지, 거부 사유). 거부는 WARNING 이다.
- `turn` 출력에는 최종 답변과 함께 작업 기록(실행한 도구와 결과 요약, 승인 결정, 바뀐 파일, diff)이 남는다. Langfuse LLM-as-a-Judge 평가기는 같은 트레이스의 다른 observation 을 읽지 못하므로, `turn` 하나만 보고 채점할 수 있게 하려는 것이다. `.env` 는 diff 에 넣지 않는다.
- 중단(Ctrl+C)과 오류는 `turn` 을 ERROR 로 표시한다. 출력 한도로 잘린 응답과 예산 상한 도달은 WARNING 으로 표시하고 작업 기록에 `## 경고` 절을 남긴다.
- 에이전트가 읽은 코드와 명령 출력이 Langfuse 로 전송된다. `.env` 내용은 가드가 막는다.
- **비용**: Langfuse 는 모델 정의로 비용을 계산한다. `anthropic/claude-opus-5.5` 같은 OpenRouter 이름은 기본 정의에 없어 Project Settings > Models 에 가격을 등록해야 한다(사용량 키: `input`, `output`, `output_reasoning`, `input_cache_read`, `input_cache_creation`). OpenRouter 가 알려 준 실제 비용은 `turn` 의 metadata `usage` 와 `/cost` 에 있다.

## 시험

```
uv run ruff check . && uv run pyright && uv run pytest -q    # 모델 호출 없음. CI(windows-latest)도 같다
```

## 보안 메모

- `.env` 는 이름 변형(대소문자, 끝 공백·점, `::$DATA`, 8.3 짧은 이름 `ENV~1`)까지 파일 도구가 막는다. 셸의 자동 승인도 이런 변형, 작업 폴더 밖(PowerShell 공급자 경로 `FileSystem::`, `env:`, `HKCU:` 포함), `--opt=경로`, 둥근 따옴표, 끝나지 않는 읽기(`tail -f`, PowerShell `cat -Wait`), `wc --files0-from` 이 들어가면 묻는다. GNU 도구는 긴 옵션의 앞부분만 써도 받으므로 `tail --fo`, `wc --files0` 같은 약어도 묻는다.
- 셸의 PATH 에서 상대 경로 항목(`.`, `bin` 등)을 뺀다. 셸의 현재 폴더가 작업 폴더라 저장소에 심은 실행 파일이 `git status` 같은 자동 승인 명령으로 불릴 수 있기 때문이다. 절대 경로 항목은 작업 폴더 안이라도 둔다(켜 둔 대상 프로젝트 venv). 다만 시작할 때 버전을 확인하려고 실행하지는 않는다.
- 개인 키·자격 증명 파일(`id_rsa`, `*.pem`, `credentials*.json`, `.netrc` 등) 읽기는 manual·auto-edit 에서 묻는다.
- 현재 폴더에 있는 같은 이름의 실행 파일(`git.cmd` 등)은 고르지 않는다. 낯선 저장소 폴더에서 띄워도 그 안의 실행 파일이 에이전트 셸이 되지 않는다.
- 작업 폴더가 code-agent 저장소 자신(실제 키가 든 `.env`)이거나 그 상위 폴더면 시작할 때 경고한다. `-w` 를 빼먹었으면 시작할지 한 번 묻는다.
