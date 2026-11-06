# Deep Agents 기반 터미널 코드 에이전트 구축 계획

> 상태: 1단계 MVP 구현과 무과금 시험 완료(2026-11-05). 남은 것은 실제 API 수동 E2E(6장 2)다. 작성일은 2026-11-04.
> 근거 표기: **[V]**는 `deepagents==0.7.32` 태그 소스로 확인한 사실, **[D]**는 공식 문서(버전 고정 아님)에 근거한 사실, **[I]**는 추론이라 0단계에서 확인해야 하는 사실이다.

## 1. Context
- **목적**: LangChain `deepagents` SDK로 Claude Code·dcode와 비슷한 **터미널 코딩 에이전트**를 직접 만든다. 학습과 실사용이 목적이고, Day08 과제 제출용은 아니다.
- **확정된 결정**

  | 항목 | 결정 |
  |---|---|
  | 인터페이스 | 터미널 CLI(REPL). 스트리밍 출력과 승인 프롬프트를 둔다 |
  | 실행 환경 | 지정한 로컬 작업 폴더. 파일 쓰기와 셸 실행 전에 사람 승인(HITL) |
  | 모델 | 기본은 `anthropic:claude-opus-5-5`. 환경변수로 OpenRouter 모델로 교체할 수 있다 |
  | 구현 방식 | dcode(약 20만 줄)를 포크하지 않고 SDK `create_deep_agent` 위에 직접 조립한다. dcode는 설계 참고만 한다 |
  | 범위 | MVP를 먼저 만들고 단계적으로 확장한다 |
- **환경 현황**
  - `projects\`가 비어 있었다.
  - 실제 Python과 uv가 없다(`python.exe`는 WindowsApps 스텁뿐). git 2.56(Git Bash 포함)은 있다.
  - 참고 저장소 `..\pjt`는 deepagents 0.7.32 기준이지만 `.venv`가 없다.
- **버전 기준**
  - `deepagents==0.7.32`(langchain 1.4.7, langgraph 1.2.19, langchain-anthropic 1.7.7)를 쓴다. pjt `uv.lock`과 같은 버전이다.
  - 같은 SDK 버전을 고정한 공식 코딩 에이전트는 `deepagents-code` 0.2.4다. 다만 이 패키지는 Python 3.14가 필요하다.

## 2. 최종 구조 (`projects\code-agent\`)
```
code-agent/
├── PLAN.md                 # 이 문서
├── pyproject.toml          # 독립 uv 프로젝트 (pjt 워크스페이스에 넣지 않음)
├── .env.example            # 변수 이름만 둔다
├── README.md
├── src/code_agent/
│   ├── __main__.py         # 진입점: python -m code_agent / 스크립트 `code-agent`
│   ├── settings.py         # .env 로드(dotenv_values), Settings dataclass
│   ├── models.py           # make_model(): "anthropic:" / "openrouter:" 분기
│   ├── backend.py          # WindowsShellBackend + CompositeBackend 조립
│   ├── prompts.py          # 코딩 시스템 프롬프트, 서브에이전트 프롬프트
│   ├── agent.py            # build_agent(): create_deep_agent 조립, run_turn(): 스트림·interrupt·재개 루프
│   ├── guards.py           # DotenvGuardMiddleware (.env 차단, grep 결과 필터)
│   ├── usage.py            # Usage, UsageCallback (모델 호출별 토큰·비용 집계)
│   ├── caching.py          # OpenRouterCacheMiddleware (session_id, cache_control)
│   ├── approval.py         # 승인 모드, 셸 allow-list, HITL 결정 생성 (순수 함수)
│   ├── render.py           # 스트림 → rich 터미널 출력
│   ├── cli.py              # argparse, REPL, 슬래시 커맨드, interrupt/resume 루프
│   ├── subagents.py        # (2단계)
│   ├── tracing.py          # Langfuse 트레이스 (turn 루트, CallbackHandler, 승인 event)
│   ├── stops.py            # 잘린 응답 자동 복구(TruncationRecoveryMiddleware), 잘림·예산 상한 경고(TurnStops)
│   └── skills/             # (2단계) 내장 SKILL.md (UTF-8, BOM 없음)
└── tests/
    ├── fakes.py            # ScriptedChatModel (pjt에서 복사)
    ├── fixtures/buggy_repo/
    └── test_*.py
```
사용자 데이터는 `~/.code-agent/`에 둔다: `sessions.db`, `AGENTS.md`(사용자 메모리), `artifacts/`, `runs/`.

## 3. 단계별 작업

### 0단계 — 환경 준비와 API 실측
1. uv를 설치하고(`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`) `uv python install 3.13`을 실행한다.
2. `projects\code-agent`에서 `uv init --package`을 실행한다.
   - 의존성: `deepagents==0.7.32`, `langgraph-checkpoint-sqlite`, `python-dotenv`, `rich`, `prompt_toolkit`
   - 선택 extra: `langchain-openrouter==0.2.8`
   - dev: `pytest`, `ruff`
   - pjt처럼 `exclude-newer`로 재현성을 고정한다.
3. **설치한 소스로 확인할 것** (`.venv\Lib\site-packages\deepagents`)
   - (a) 큰 도구 결과와 대화 기록을 오프로드하는 가상 경로 이름. 예상은 `/large_tool_results/`, `/conversation_history/`다.
   - (b) `permissions=`와 `LocalShellBackend`를 함께 쓸 때 `NotImplementedError`가 나는지.
   - (c) `agent.stream(..., subgraphs=True)`의 반환 형태. `(ns, mode, data)` 튜플인지 v2 dict인지.
   - (d) 서브에이전트가 부모의 `interrupt_on`을 상속하는지.
   - 결과는 이 문서의 "확인 결과" 절에 적는다.
4. `.env.example`에 둘 변수:
   - `ANTHROPIC_API_KEY`
   - `CODE_AGENT_MODEL`(기본 `anthropic:claude-opus-5-5`)
   - `CODE_AGENT_EFFORT`(기본 `high`)
   - `CODE_AGENT_SHELL`(`bash`|`powershell`)
   - `OPENROUTER_API_KEY`(선택)

#### 0단계 확인 결과 (2026-11-05)
- **설치**: uv 0.13.7, CPython 3.13.18. `exclude-newer = 2026-10-31`로 해석한 결과 deepagents 0.7.32, langchain 1.4.7, langgraph 1.2.19, langchain-anthropic 1.7.7이 설치되어 계획과 같다. `.env.example`, `.gitattributes`(LF 고정)를 추가했다.
- **(a) 오프로드 경로** [V]: 고정 경로가 아니라 `{artifacts_root}/large_tool_results/`, `{artifacts_root}/conversation_history/`, `{artifacts_root}/blobs/`다. `artifacts_root`는 `CompositeBackend(..., artifacts_root="/")` 인자에서 온다(`backends/composite.py:260`, `middleware/filesystem.py:1993`, `middleware/summarization.py:738`).
  - 그래서 `artifacts_root="/.code-agent-artifacts"`로 두고 `/.code-agent-artifacts/` 라우트 하나만 `FilesystemBackend(~/.code-agent/artifacts)`로 보내면 세 경로가 모두 작업 폴더 밖으로 나간다.
- **(b) permissions + 셸 백엔드** [V]: 충돌한다. 실행 가능한 백엔드에 `permissions`를 주면 `NotImplementedError`가 난다(`middleware/filesystem.py:2055`). 예외는 모든 규칙 경로가 라우트 접두사 아래에 있을 때뿐인데, `/.env`는 default 백엔드 경로라 해당하지 않는다.
  - 그래서 `.env` 차단은 계획대로 `wrap_tool_call` 미들웨어로 구현한다.
- **(c) stream 반환 형태** [V]: `version` 기본값이 `"v1"`이라(`langgraph/pregel/main.py:2759`) `stream_mode=[...]`와 `subgraphs=True`를 함께 주면 `(ns, mode, data)` 튜플이 온다. `mode == "messages"`일 때 `data`는 `(chunk, metadata)` 튜플이다. `version="v2"`를 주면 `StreamPart` dict로 바뀌지만 쓰지 않는다.
- **(d) 서브에이전트의 `interrupt_on`** [V]: 상속한다. `spec.get("interrupt_on", interrupt_on)`이다(`graph.py:797`). 대본 모델로 실측했을 때 서브에이전트의 `execute`도 승인을 요구했다.
  - **주의**: 서브에이전트에서 난 interrupt는 스트림에 **두 번** 온다. 한 번은 `ns=("tools:<id>",)`, 한 번은 `ns=()`이고, `Interrupt.id`가 같다. 클라이언트는 **id로 중복을 제거**해야 한다. 그렇지 않으면 승인을 두 번 묻는다. resume은 결정 하나(`Command(resume={"decisions": [...]})`)로 끝까지 진행됐다.
- **그 외 확인**
  - `TodoListMiddleware`는 `langchain.agents.middleware`에서 import한다. 아래 agent.py 스니펫의 [I]는 해소됐다.
  - 기본 도구 목록을 실측했다: `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, `execute`, `task`. `write_todos`는 없다. `task`는 GP 서브에이전트가 기본으로 켜져 있어서 생긴다.
  - interrupt의 `review_configs` 기본 `allowed_decisions`는 `approve`, `edit`, `reject`, `respond`다. 계획대로 approve와 reject만 쓰려면 `interrupt_on={"execute": {"allowed_decisions": ["approve", "reject"]}, ...}`로 제한한다.
  - 파이프로 출력을 받으면 콘솔 코드페이지(cp949) 때문에 한글이 깨졌다. CLI 진입점에서 `sys.stdout.reconfigure(encoding="utf-8")`을 호출하거나 `PYTHONUTF8=1`을 준다.
  - `git config core.autocrlf=true`가 시스템 전역 설정이라 `.gitattributes`의 `* text=auto eol=lf`로 LF를 고정했다.

### 1단계 — MVP: 파일 편집 + 셸 + 승인 + 세션

**settings.py / models.py**
- `pjt\Day04_new\cap_lab\settings.py:90` 패턴을 따른다: `ChatAnthropic(model, reasoning_effort=effort, max_tokens=..., timeout, max_retries)`. 모델은 **항상 명시**한다. `model=None`이면 deprecated 기본값 `claude-sonnet-4-6`이 쓰인다 [V].
- Claude Opus 5.5 제약:
  - `temperature`와 `thinking`을 보내지 않는다(400).
  - 강제 `tool_choice`도 400이므로 `response_format`에 `ToolStrategy`를 쓰지 않는다.
  - effort 기본값이 `medium`이라서 명시적으로 지정한다.
- `openrouter:` 접두사면 `ChatOpenRouter`를 쓴다(`pjt\Day08\guardlab\config.py:45` 참고).

**backend.py**
- `WindowsShellBackend(LocalShellBackend)`: `root_dir=workspace, virtual_mode=True, timeout=120`
  - 파일 도구는 `/` = 작업 폴더로 갇힌다.
  - **셸은 전혀 격리되지 않는다** [V]. 그래서 셸 쪽 안전장치는 승인이다.
  - `execute`를 오버라이드한다. 0.7.32 기본 구현은 `subprocess.run(shell=True)`라서 Windows에서는 cmd.exe로 실행된다.
    - 셸 선택: Git Bash(`bash.exe -lc`)가 있으면 그것을, 없으면 `powershell -NoProfile -Command`를 쓴다.
    - 타임아웃: `Popen` 후 `taskkill /T /F /PID`로 프로세스 트리까지 종료한다. Windows에서는 `start_new_session`이 빠져 고아 프로세스가 남는다(PR #5907).
    - 출력: `encoding="utf-8", errors="replace"`로 디코드하고, env에 `PYTHONUTF8=1`을 준다. 한국어 Windows 기본 코드페이지는 cp949다.
    - 오버라이드 패턴은 `pjt\Day04\coding_skills\run_files.py`의 `RecordedLocalShellBackend`를 참고한다.
  - 셸 환경변수:
    - `inherit_env=False, env=None`이면 **빈 환경**이라 PATH도 없다 [V]. 그래서 `os.environ` 복사본을 넘긴다.
    - 복사본에서 `*_API_KEY`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`는 제거한다.
    - `SYSTEMROOT`, `COMSPEC`, `PATHEXT`, `TEMP`, `PATH`는 유지한다. PATH 구분자는 `os.pathsep`을 쓴다.
- `CompositeBackend(default=shell, routes={"/.code-agent-artifacts/": FilesystemBackend(~/.code-agent/artifacts)}, artifacts_root="/.code-agent-artifacts")`
  - 큰 도구 결과, 대화 기록, blob 오프로드가 모두 `artifacts_root` 아래로 가므로(0단계 a) 라우트 하나로 `~/.code-agent/artifacts/`에 모인다. 이렇게 해야 **작업 폴더가 오염되지 않는다**(dcode와 같은 방식).
  - `execute`는 경로와 상관없이 항상 default 백엔드로 간다 [V].

**prompts.py** — dcode `system_prompt.md`에서 가져올 주제
- 런타임 정보: 작업 폴더, OS와 셸 종류, 경로 규칙(파일 도구는 `/`부터의 가상 경로, 셸은 작업 폴더가 cwd), 승인 모드 안내.
- 작업 루프: 이해 → `write_todos`로 계획 → 기존 관례를 따르는 최소 수정(새 파일보다 기존 파일 수정 우선) → 테스트·린트로 검증 → 짧게 요약.
- 대용량 파일은 `read_file`의 offset/limit으로 나눠 읽는다. 독립적인 도구 호출은 병렬로 한다. 같은 수정 시도는 최대 3회까지만 한다.
- git 안전 규칙:
  - 요청이 없으면 commit하지 않는다.
  - force-push와 amend는 금지한다.
  - stage는 파일을 지정해서 한다.
  - 비밀값은 commit하지 않는다.
- 코드를 언급할 때는 `file_path:line` 형식으로 쓰고, 답변은 한국어로 한다.
- `system_prompt`를 빠뜨리면 빈 프롬프트가 되므로 반드시 지정한다.

**agent.py — `build_agent(settings, workspace, checkpointer)`**
```python
from langchain.agents.middleware import TodoListMiddleware  # [V] 0단계에서 확인

register_harness_profile("anthropic", HarnessProfile(          # MVP: GP 서브에이전트 끔
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))

APPROVE_OR_REJECT = {"allowed_decisions": ["approve", "reject"]}
create_deep_agent(
    name="code-agent", model=model, backend=backend, system_prompt=CODING_PROMPT,
    middleware=[TodoListMiddleware(),          # 0.7.0부터 write_todos가 기본 아님 [V]
                DotenvGuardMiddleware(),       # .env 차단. permissions=는 셸 백엔드와 충돌한다 [V] 0단계 b
                *budget_middleware()],         # Model/ToolCallLimitMiddleware (pjt Day08 guardlab/budget.py)
    interrupt_on={t: APPROVE_OR_REJECT for t in ("execute", "write_file", "edit_file", "delete")},
    checkpointer=checkpointer,
)
```
- 최종 도구는 `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, `execute`, `write_todos`다. GP 서브에이전트를 끄면 `task`는 빠진다.
- `DotenvGuardMiddleware`는 `wrap_tool_call`에서 파일 도구의 경로 인자가 `.env*`와 맞으면 실행하지 않고 거부 ToolMessage를 돌려준다. `glob`과 `grep` 결과에서도 `.env*`를 걸러 낸다.
- 커스텀 도구는 예외를 문자열로 바꿔 반환한다. 도구 예외는 루프를 끊는다.

**approval.py** (순수 로직이라 테스트하기 쉽다)
- 모드:
  - `manual`(기본): 전부 묻는다.
  - `auto-edit`: 파일 쓰기는 자동 승인하고 셸은 묻는다.
  - `yolo`: 전부 자동 승인하고, 켤 때 경고를 띄운다.
- 그래프는 항상 interrupt하고, **클라이언트가 모드에 따라 자동으로 결정한다**. 그래서 그래프는 하나로 고정되고 `/mode`로 즉시 바꿀 수 있다.
- 결정은 dcode처럼 **approve와 reject 두 가지만** 쓴다.
  - reject에는 `message`를 담는다(예: "사용자가 거부함. 재시도 말고 대안을 물어볼 것").
  - edit은 쓰지 않는다. 수정된 인자는 재검증 없이 바로 실행되기 때문이다(pjt Day08 05 교훈).
- 읽기 전용 셸 allow-list(`git status`, `git diff`, `git log`, `ls`, `cat`, `pwd` 등)는 모드와 관계없이 자동 승인한다. 단, 다음 경우는 예외다.
  - `&&`, `||`, `;`, `|`, `>`, `<`, 백틱, `$(`가 들어 있으면 무조건 묻는다.
  - `python`, `bash`, `uv`, `node`, `make` 같은 인터프리터와 빌드 도구는 allow-list에 넣지 않는다(dcode THREAT_MODEL T13).

**render.py / cli.py**
```python
for ns, mode, data in agent.stream(inp, cfg, stream_mode=["messages", "updates"], subgraphs=True):
    # ns == () → 메인 에이전트, ("tools:<call_id>",) → 서브에이전트
    # mode == "messages": (AIMessageChunk|ToolMessage, meta) → 토큰 스트리밍 / 도구 결과
    # mode == "updates" and "__interrupt__" in data → 승인 처리
```
- AI 텍스트는 토큰 단위로 스트리밍한다.
- 도구 호출은 이름과 핵심 인자만, 도구 결과는 앞 N줄만 보여 준다. 서브에이전트 출력에는 접두사를 붙인다.
- 승인 처리:
  - 서브에이전트 interrupt는 ns가 다른 두 이벤트로 중복해서 오므로 `Interrupt.id`로 중복을 제거한다(0단계 d).
  - 한 interrupt에 `action_requests`가 **여러 개** 올 수 있다(병렬 도구 호출).
  - 각 요청에 대해 같은 순서로 결정을 만든다. `execute`는 명령을, `edit_file`은 old→new diff를, `write_file`은 경로와 크기를 보여 준다.
  - 그다음 `Command(resume={"decisions": [...]})`로 같은 스트림 루프를 다시 돈다.
  - interrupt가 없어질 때까지 반복한다(`pjt\Day08\guardlab\runner.py:107` 패턴).
- 세션:
  - `SqliteSaver(sqlite3.connect("~/.code-agent/sessions.db", check_same_thread=False))`
  - `thread_id`는 uuid로 만들고, `--resume <id>`로 이어 간다. 작업 폴더는 thread와 함께 저장한다.
- 슬래시 커맨드: `/help`, `/new`, `/mode <m>`, `/diff`(git diff), `/todos`, `/exit`.
- `Ctrl+C`를 누르면 현재 실행만 취소하고 REPL로 돌아간다.
- CLI: `code-agent [--workspace PATH] [--model ...] [--mode manual] [--resume ID]`

**테스트 (모델 호출 없음)**
- `pjt\Day02_new\tests\fakes.py`의 `ScriptedChatModel`을 복사해 쓴다.
- `test_backend.py`
  - `echo` 실행과 종료 코드를 확인한다.
  - 한글 출력이 깨지지 않는지 확인한다.
  - 타임아웃 시 자식 프로세스가 종료되는지 확인한다.
  - 셸 env에 `ANTHROPIC_API_KEY`가 없는지 확인한다.
  - `../`와 `C:/Windows` 경로를 파일 도구가 거부하는지 확인한다.
  - 오프로드 파일이 작업 폴더가 아닌 artifacts에 생기는지 확인한다.
- `test_approval.py`
  - 모드별 결정을 확인한다.
  - allow-list 우회 사례(`git status && rm -rf x`, `git log | sh`)가 차단되는지 확인한다.
  - 다중 action_request의 순서가 유지되는지 확인한다.
- `test_agent.py`
  - 노출되는 도구 목록을 확인한다.
  - 대본 모델이 `execute`를 호출하면 interrupt가 걸리는지 확인한다.
  - reject 후 명령이 실행되지 않는지 확인한다.
  - approve 후 resume이 끝까지 완료되는지 확인한다.
  - `.env` 읽기가 거부되는지 확인한다.
- **MVP 완료 조건**: `tests\fixtures\buggy_repo`(테스트 1개 실패)의 복사본에서 "실패하는 테스트를 고쳐줘"를 실행하면, 승인을 거쳐 수정하고 pytest가 통과해야 한다.

#### 1단계 진행 결과 (2026-11-05)
- **시험**: `ruff check` 통과, `pytest` 117개 통과(모델 호출 없음). 파일은 `test_backend.py`(19), `test_approval.py`(75), `test_agent.py`(16), `test_cli.py`(7)다.
  - `test_cli.py::test_fixes_buggy_repo_with_approvals`는 대본 모델로 완료 조건의 배선을 확인한다: 실패 확인 → 읽기 → 수정 승인 → pytest 승인 → 4개 통과.
  - 가드 시험은 가드를 꺼서 실제로 실패하는지 확인했다.
- **계획과 달라진 점**
  - 셸은 `bash.exe -c`로 실행한다. `-lc`(로그인 셸)는 필요 없었다. Windows PATH만 있어도 Git Bash가 `/usr/bin`을 잡는다.
    - Git 루트는 `git.exe`의 상위 폴더에서 `bin/bash.exe`를 찾는다. git이 `Git\ucrt64\bin`에 있는 경우도 있다.
  - 출력이 UTF-8로 디코드되지 않으면 OEM 코드페이지(cp949)로 다시 디코드한다. `tasklist` 같은 Windows 기본 도구는 파이프에 cp949로 쓴다.
  - 기본 `LocalShellBackend`의 타임아웃 문제를 실측했다. 고아 프로세스가 남을 뿐 아니라, 손자 프로세스가 쥔 파이프가 닫힐 때까지 **돌아오지 않는다**(타임아웃 3초에 31초 걸림). 새 백엔드는 3.4초에 끝난다.
  - allow-list를 강화했다. 계획대로라면 아래가 승인 없이 실행될 수 있었다.
    - `cat .env`, `cat ~/.ssh/id_rsa`
    - `git -c core.fsmonitor=... status`
    - auto-edit 모드에서 `.git/config`를 고친 뒤 자동 승인되는 `git status`로 그 설정을 실행하는 경로
    - 대응:
      - 작업 폴더 밖, `.env`, 와일드카드를 가리키는 인자는 묻는다.
      - `git` 바로 뒤에는 하위 명령만 허용하고 `--output`은 막는다.
      - `$`, `(`, `)`, `{`, `}`, `&`는 무조건 묻는다.
      - auto-edit 모드에서도 `.git/` 아래 쓰기는 묻는다.
  - `.env` 차단은 `guards.py`의 `DotenvGuardMiddleware`로 했다.
    - `.env.example`, `.env.sample`, `.env.template`은 허용한다.
    - grep은 세 출력 형식 모두에서 `.env` 줄을 지운다. 가드가 오프로드보다 안쪽에서 돌아 오프로드 파일에도 비밀값이 남지 않는다(시험으로 확인).
  - thinking은 langchain-anthropic 1.7.7이 모델 프로필에 따라 `{"type": "adaptive", "display": "summarized"}`로 채운다. Opus 5.5에서 허용되는 값이다. 그래서 thinking 요약을 흐리게 스트리밍한다.
  - `max_tokens=64000`, `streaming=True`로 둔다. 예산은 사용자 메시지 하나당 모델 100회, 도구 300회다.
  - 세션 표 `code_agent_threads`(thread → 작업 폴더)는 `sessions.db`에 둔다. 시작에 성공한 뒤에만 기록한다.
  - mintty(Git Bash 창)에서는 prompt_toolkit이 `NoConsoleScreenBufferError`를 내서 `input()`으로 대체한다.
- **1단계 이후 추가 (2026-11-05)**: 3단계의 사용량 표시를 앞당겼고, OpenRouter 프롬프트 캐시를 붙였다.
  - **사용량** (`usage.py`)
    - `UsageCallback.on_llm_end`로 모든 모델 호출을 센다. 서브에이전트와 요약 미들웨어의 호출도 포함한다.
    - 턴이 끝나면 한 줄(`모델 N회 · 입력 · 캐시 · 출력 · $비용`)을 보이고, `sessions.db`의 `code_agent_usage` 표에 기록한다. 중단된 턴도 기록한다.
    - `/cost`는 세션 합계를 보인다. 비용은 OpenRouter가 알려 준 값이다. Anthropic 직접 호출은 토큰만 보인다.
    - 세션 상태로 세지 않는 이유가 있다. 이 버전의 체크포인트는 `messages` 채널 값을 통째로 저장하지 않고 `writes`를 재생해 복원한다. 그래서 읽으려면 `agent.get_state()`가 필요하다. 게다가 요약이 기록을 바꾸면 합계가 줄어든다.
  - **캐시** (`caching.py`)
    - `OpenRouterCacheMiddleware`가 OpenRouter 호출에 `session_id=code-agent-<thread>`를 더한다. 공급자 sticky routing을 위해서다.
    - Anthropic 모델이면 최상위 `cache_control: {"type": "ephemeral"}`도 더한다. 자동 캐시이고 TTL은 5분이다.
    - deepagents의 캐시 미들웨어는 `ChatAnthropic`에만 붙는다.
    - **실측**: Opus 5.5, effort low로 모델 호출 2번을 돌렸다.
      - 2번째 호출이 7,693토큰을 캐시에서 읽어 $0.0020이었다. 캐시가 없으면 약 $0.031이다.
      - 1번째 호출은 캐시 쓰기라 1.25배 가격이다.
      - 비교로, 캐시 전 tetris 세션은 17회 호출에 입력 33.9만 토큰, 캐시 읽기 0, 비용 $1.77이었다.
- **남은 것**
  - 실제 API 수동 E2E(6장 2).
  - 확인 필요: Opus 5.5의 refusal `fallbacks`. langchain-anthropic 1.7.7이 응답의 `fallback` 블록을 다루지 못할 수 있어 켜지 않았다. E2E에서 확인한 뒤 켤지 정한다.

### 2단계 — 컨텍스트: 메모리·스킬·서브에이전트·헤드리스
- **메모리** [V]
  - `memory=["/.code-agent-user/AGENTS.md", "/AGENTS.md"]`로 둔다. 앞은 `~/.code-agent/` 라우트(사용자 메모리), 뒤는 작업 폴더 루트(프로젝트 메모리)다.
  - 메모리는 스레드당 한 번 로드된다. 그래서 대화 중에 수정한 내용은 `/new` 이후에 반영된다는 점을 안내한다.
  - `/remember` 커맨드는 "이번 대화에서 배운 규칙을 AGENTS.md에 반영하라"는 프롬프트를 주입한다.
- **스킬** [D]
  - `skills=["/.code-agent-skills/", "/.agents/skills/"]`로 둔다. 앞은 내장 스킬 라우트(읽기 전용), 뒤는 프로젝트 스킬이다. 같은 이름이면 뒤쪽 소스가 이긴다.
  - 내장 스킬은 `pjt\Day04\coding_skills\skills\{debug-python,add-python-feature,review-python}`을 참고해 작성한다.
  - **BOM 없는 UTF-8**로 저장한다. BOM이 있는 SKILL.md는 아무 오류 없이 무시된다(#6811). 테스트로 검사한다.
- **서브에이전트** (`subagents.py`)
  - `explorer`: `tools`를 ls, read_file, glob, grep으로 한정한 읽기 전용 코드베이스 탐색 에이전트다. 저비용 모델을 선택할 수 있게 한다.
  - `reviewer`: 현재 diff를 리뷰하고 위험을 지적한다. 읽기 전용이다.
  - 커스텀 서브에이전트는 스킬을 상속하지 않으므로(`skills` 명시 필요) 필요하면 명시한다. 서브에이전트에서 발생한 interrupt도 부모 스트림의 `__interrupt__`로 올라오므로 같은 승인 루프로 처리한다.
  - GP 서브에이전트를 다시 켤지는 여기서 결정한다.
- **헤드리스 모드**: `code-agent -n "프롬프트" --mode auto-edit`
  - 셸은 allow-list만 허용하고 나머지는 reject한다.
  - 종료 코드는 성공 0, 실패 1, 타임아웃 124로 둔다. CI와 평가에 쓴다.

### 3단계 — 관측·가드레일
- `tracing.py`: `TraceMiddleware`(`pjt\Day08\guardlab\trace.py` 참고)
  - 요청마다 `~/.code-agent/runs/<run_id>/trace.jsonl`을 남긴다(모델 호출, 도구 호출과 결과, 승인 결정, 오류).
  - `metrics.json`에는 소요 시간, 호출 수, 토큰(`usage_metadata`)을 기록한다.
- 셸 명령 기록은 `commands/*.json`으로 남긴다(`RecordedLocalShellBackend` 방식).
- `/cost`와 `/trace` 커맨드를 둔다.
- 선택: 환경변수 `LANGSMITH_TRACING=true`만 켜면 LangSmith 추적이 된다.
- CRLF 보존 미들웨어: `edit_file`이 파일 전체의 줄바꿈을 LF로 바꾸는 문제(#7488)를 막는다. 원본이 CRLF였으면 수정 후 복원한다.

#### 3단계 진행 결과 (2026-11-05)
- **`/cost`**: 1단계 이후 추가에서 구현했다(`usage.py`).
- **`tracing.py`**: 로컬 `trace.jsonl` 대신 Langfuse Cloud로 구현했다. 선택 extra로 `langfuse>=5.1,<6`을 쓴다.
  - **켜지는 조건**: `LANGFUSE_PUBLIC_KEY`와 `LANGFUSE_SECRET_KEY`가 둘 다 있을 때만 켜진다. 키는 `Langfuse(...)` 인자로만 넘겨 `os.environ`을 바꾸지 않는다.
  - **호스트 환경변수**: v5는 `LANGFUSE_BASE_URL`을 쓴다. pjt(v4)의 `LANGFUSE_HOST`도 받는다.
  - **트레이스 구조**
    - 사용자 메시지 하나가 루트 observation `turn`(agent 타입) 하나다.
    - `propagate_attributes`로 `session_id`(thread)와 태그를 붙인다.
    - 그 아래에 `langfuse.langchain.CallbackHandler`가 그래프·모델·도구를 남긴다.
    - 한 턴 안의 stream 호출(재개 포함)은 같은 핸들러 인스턴스를 쓴다.
  - **승인 결정**: `decide_all(on_decision=...)`으로 받아 `approval:<도구>` event로 남긴다. `approval.py`는 여전히 Langfuse를 모른다.
  - **턴 끝**: `UsageCallback` 합계를 `turn`의 metadata에 남긴다. 핸들러는 비용을 넘기지 않아서 OpenRouter 실제 비용은 여기 있다.
  - **시험**: 가짜 tracer로 기록 내용을 확인했다. 실제 SDK는 `span_exporter=InMemorySpanExporter()`로 네트워크 없이 확인했다(트레이스 하나, 모델 호출이 generation 3개). 닿지 않는 서버로도 턴이 끝나고 종료가 오래 걸리지 않는 것을 확인했다.
- 셸 env에서 `_KEY$`로 끝나는 변수도 뺀다. 전에는 `LANGFUSE_PUBLIC_KEY`가 남았다.
- **실측 (jp.cloud.langfuse.com)**: 승인이 한 번 끼는 턴(`echo hello`)을 돌렸다.
  - 트레이스 1개에 observation 23개가 남았다.
    - 루트 `turn`에 입력·출력과 usage metadata가 있다.
    - `approval:execute` event가 있다.
    - 재개 후 단계도 같은 트레이스에 있다.
    - generation 2개에 모델 `anthropic/claude-opus-5.5`와 캐시 쓰기·읽기 토큰이 있다.
  - 조회 API에 주의한다. SDK v5에서 `api.trace.*`와 `GET /api/public/traces`로는 보이지 않았다. `api.observations.get_many(session_id=..., fields="core,basic,model,usage")`로 조회한다.
  - **가격 정의 등록**: `anthropic/claude-opus-5.5`는 Langfuse 기본 정의에 없어 비용이 비었다. 그래서 프로젝트에 직접 등록했다.
    - 패턴: `(?i)^anthropic/claude-opus-5\.5$`
    - 100만 토큰당 가격: input $4, output·output_reasoning $20, input_cache_read $0.20, input_cache_creation $5
      - 정정(2026-11-06): 처음에 캐시 읽기를 $0.40(입력의 0.1배)으로 적었다. Opus 5.5 는 0.05배인 $0.20 이다(bomber·az-quiz 실측 역산 $0.19~0.20, Anthropic 문서). Langfuse 정의도 고친다.
    - 시험 generation으로 계산값이 맞는 것을 확인했다. 등록 전에 수집된 generation에는 소급되지 않는다.
- **LLM-as-a-Judge 준비 (`worklog.py`)**
  - **배경**: Langfuse의 observation 평가기는 같은 트레이스의 다른 observation(도구 실행 등)을 읽지 못한다. 그래서 `turn` 출력을 최종 답변만이 아니라 판단 근거가 되는 작업 기록 전체로 바꿨다.
  - **작업 기록 내용**
    - 최종 답변
    - 실행한 도구와 결과 요약: execute는 종료 코드와 마지막 줄, read_file은 읽은 범위
    - 승인 결정: 자동/사용자, 거부 사유
    - 바뀐 파일(A/M/D와 +/- 줄 수)과 diff
  - **바뀐 파일과 diff 구하는 법**: 턴 시작과 끝에 작업 폴더 스냅샷을 비교한다.
    - `.env`는 넣지 않는다.
    - `.git`, `node_modules`, `.venv` 등은 건너뛴다.
    - 파일이 5,000개를 넘으면 생략한다.
  - **길이 제한**: 평가 비용 때문에 도구 호출 80개, 줄당 240자, diff 3만 자까지만 남긴다.
  - **평가 설정**: evaluator 하나(`code-agent turn quality`, 0~1 NUMERIC)가 기준 네 가지를 한 번에 채점하고 이유를 남긴다. 기준은 요청 완수, 검증, 범위, 보고 정확성이다.
    - 평가 모델: OpenRouter `anthropic/claude-sonnet-5.5`. Langfuse LLM Connection으로 연결한다(adapter openai, provider `openrouter`, base URL `https://openrouter.ai/api/v1`).
    - rule: `isRootObservation = true`이고 `traceName = code-agent turn`인 것, 샘플링 100%.
      - observation `name`으로는 필터할 수 없다. 허용 컬럼은 traceName, tags, type, metadata 등이다.
    - 설정은 `scripts/setup_langfuse_judge.py`에 코드로 둔다. 다시 돌려도 중복 생성하지 않는다.
  - **실측**
    - 실제 턴(`hello.py` 만들고 실행해 확인): 1.0점. 기준별 "통과"와 근거가 comment에 남았다.
    - 가짜 turn(실행 기록 없이 "테스트 4개 통과, README 정리"라고 주장): 0.5점.
      - 검증: 실패(pytest 실행 기록 없음)
      - 보고 정확성: 실패(작업 기록에 없는 일을 했다고 주장)
    - 점수 조회는 `api.scores.get_many(trace_id=..., fields="details")`로 한다. comment는 `details`에 있다.
  - **소급 채점과 평가기 v2**
    - **소급 채점**: `scripts/backfill_langfuse.py`가 Langfuse 연동 전 세션(buggy_copy, tetris, bomber-game, 8턴)을 sessions.db에서 복원해 `backfill` 태그로 올린다.
    - **1차 결과에서 문제 두 가지가 드러났다**
      - 점수와 이유가 어긋났다. tetris 턴 2는 이유가 네 기준 모두 통과인데 점수가 0.0이었다.
      - 평가기가 시스템 프롬프트(작업 폴더)와 이전 턴을 보지 못해, 맞는 말을 "근거 없는 주장"으로 감점했다(0.75가 세 건).
    - **고친 것**
      - 이유 끝에 "실패 K개 → 점수 S"를 쓰게 했다. 점수는 S = 1 − 0.25×K로 이유와 일치하게 했다.
      - 작업 기록 끝에 `## 맥락` 절을 둔다. 작업 폴더와 같은 세션의 이전 턴 요약(요청 80자, 도구 수, 바꾼 파일 최대 10개)이 들어간다. 실시간 CLI는 턴 시작 때 `agent.get_state()`로 만든다.
      - 평가 프롬프트에 "맥락과 일치하는 말은 근거 있는 주장"이라는 규칙을 넣었다.
      - setup 스크립트는 기존 evaluator를 새 버전으로 갱신한다.
    - **2차 결과**
      - 8턴 모두 1.0이고, 기준별 근거와 계산이 남았다.
      - 실행 기록 없이 "테스트 통과, README 정리"를 주장한 가짜 turn은 여전히 0.5다(검증·보고 정확성 실패). 평가기가 느슨해지지 않았다.
    - **남긴 것**: bomber-game의 실시간 옛 트레이스(답변만 있는 형식)는 모델·도구 단계 기록 때문에 지우지 않았다. 같은 세션에 backfill 트레이스가 함께 있다.

#### 개선 1 — 말없이 멈추는 문제 (2026-11-06, `stops.py`)
- **근거**: az-quiz 세션(`0cbbc12e9ef6`)의 첫 턴에서 모델 호출 한 번이 출력 64,000토큰을 **전부 reasoning**으로 썼다.
  - `finish_reason="length"`이고 텍스트도 도구 호출도 없었다. 에이전트 루프는 도구 호출이 없으면 최종 응답으로 보고 턴을 끝낸다.
  - 화면에는 사용량 한 줄만 나왔고 $1.32가 들었다.
  - 같은 세션의 다음 사용자 메시지는 잘린 응답 뒤에 그대로 덧붙었고, OpenRouter가 받아들였다(다음 호출이 정상 응답). 그래서 덧붙이기만 하는 복구가 이 경로에서 API 오류를 내지 않는다는 실측 근거가 있다.
- **자동 복구** (`TruncationRecoveryMiddleware`)
  - 대상은 `finish_reason`(OpenRouter) 또는 `stop_reason`(Anthropic)이 `length`·`max_tokens`이고 도구 호출이 없는 응답이다.
  - `after_model`에서 안내 메시지("직전 응답이 출력 한도로 잘렸다. 생각을 줄이고 결과물은 파일에 나눠서 바로 써라.")를 덧붙이고 `jump_to: "model"`로 다시 부른다.
  - 기록은 고치지 않고 덧붙이기만 한다. Opus 5.5는 이전 thinking을 보존하기 때문이다.
  - **턴당 1번**이다. 마지막 사용자 메시지 뒤에 안내가 이미 있으면 다시 부르지 않는다. 기록으로 판단하므로 승인 대기 후 재개해도 횟수가 유지된다.
  - 안내는 `HumanMessage(name="truncation-recovery")`다. `name`은 두 공급자 모두 API로 보내지 않는다(langchain-openrouter는 `additional_kwargs["name"]`만 보낸다). 턴을 나누는 곳(`session_context`, backfill 스크립트)은 이 메시지를 건너뛴다.
  - **[V] 그래프 배선** (langchain 1.4.7 `agents/factory.py`)
    - `after_model`은 미들웨어 목록의 **역순**으로 돈다.
    - `jump_to: "model"`은 루프 입구(첫 `before_model`)로 간다.
    - 그래서 복구 미들웨어를 목록 **맨 앞**에 둔다. 그래야 `ModelCallLimitMiddleware.after_model`이 호출 수를 센 뒤에 돌고, 다시 부를 때도 모델 호출 상한 검사를 거친다.
    - 순서를 바꾸면 상한 1회 시험에서 세지 않은 호출이 한 번 더 나가는 것을 확인했다(시험이 잡는다).
  - **비용 주의**: 다시 부를 때 잘린 응답의 reasoning도 입력으로 다시 보낸다. az-quiz 기준으로 입력이 약 7.5만 토큰이고, 캐시 쓰기로 약 $0.38이다. 사용자가 '계속해줘'를 쳐도 같은 비용이 든다.
- **턴 끝 경고** (`TurnStops`)
  - 메인 에이전트의 `updates` 조각에서 잘린 응답, 자동 복구, 예산 상한 도달을 모은다.
  - 마지막 응답이 잘렸으면 이렇게 쓴다: "응답이 출력 한도(64,000토큰)에 걸려 잘렸다. [자동으로 한 번 다시 요청했지만 또 잘렸다.] 생각만 하다 한도를 다 써서 답변도 도구 호출도 없다. '계속해줘'로 이어 갈 수 있다." 텍스트가 있으면 "답변이 중간에 끊겼다"라고 쓴다.
  - 복구해서 이어 갔으면 "응답 N개가 출력 한도에 걸려 잘렸다. 자동으로 한 번 다시 요청해 이어 갔다."라고 쓴다.
  - 한도 숫자는 응답의 `output_tokens`이고, 없으면 `models.MAX_TOKENS`다.
  - CLI는 사용량 줄 앞에 노란 줄로 보인다. 다시 부르는 순간에도 노란 줄로 알린다.
  - Langfuse는 `turn`을 WARNING 수준과 status_message로 표시한다. 오류가 있으면 ERROR가 우선이다. 작업 기록에는 최종 답변 뒤에 `## 경고` 절이 생긴다.
- **시스템 프롬프트** (`# 출력 한도` 절)
  - 응답 하나의 출력 한도가 생각을 포함해 64,000토큰이라는 것을 알린다.
  - 큰 데이터나 긴 파일은 생각 속에서 완성하지 말고 바로 파일에 나눠 쓰게 한다. 예: `write_file`로 50개를 쓴 뒤 `edit_file`로 50개씩 이어 붙인다.
  - 숫자는 `models.MAX_TOKENS`에서 가져온다(`_MAX_TOKENS`를 공개 이름으로 바꿨다).
- **예산 상한 도달 표시**
  - `ModelCallLimitMiddleware`(`before_model`)와 `ToolCallLimitMiddleware`(`after_model`)가 `exit_behavior="end"`로 만든 종료 AI 메시지는 `model`이 아닌 미들웨어 노드에서 나온다.
  - 상한을 1로 낮춘 대본 시험으로 확인했다. 수정 전에는 두 메시지 모두 state 마지막에는 있었지만 **화면에는 없었다**.
  - `render.py`가 `model`이 아닌 노드의 AI 텍스트를 노란 줄로 그리고 `last_text`(트레이스의 최종 답변)로 삼는다. 턴 끝에는 "예산 상한에 걸려 턴을 멈췄다(…). '계속해줘'로 이어 갈 수 있다."를 보인다.
- **시험**: `test_stops.py` 19개를 추가해 전체 190개가 통과한다(모델 호출 없음).
  - 잘린 응답을 다시 부를 때 기록에 덧붙이기만 하는지
  - 턴당 1번인지, 다음 턴에는 다시 1번을 받는지
  - 도구 호출이 있는 잘린 응답은 다시 부르지 않는지
  - 다시 부를 때 모델 호출 상한을 지키는지
  - CLI 경고 문구 세 가지와, 정상 턴에는 경고가 없는지
  - 모델·도구 상한 도달 표시
  - 작업 기록의 경고 절, 실제 Langfuse SDK에서 WARNING이 남는지
- **확인하지 못한 것**
  - Anthropic 직접 호출(`ChatAnthropic`) 경로는 실측하지 않았다. thinking 도중 `max_tokens`로 잘린 thinking 블록을 다시 보낼 때 API가 받아들이는지 모른다. OpenRouter 경로는 az-quiz 세션으로 확인됐다.
  - 서브에이전트(ns가 있는 조각)의 잘림은 경고에 넣지 않는다. 지금은 서브에이전트를 쓰지 않는다.
- **리뷰 반영 (2026-11-06)**
  - **B1**: `render.py`가 `model`이 아닌 노드의 AI 메시지를 모두 노란 답변으로 그렸다. 두 경로에서 같은 텍스트가 다시 찍혔다.
    - HITL `after_model`은 승인 뒤 같은 응답을 다시 내보낸다. 그래서 "텍스트 + 도구 호출" 응답을 승인할 때마다 텍스트가 한 번 더 찍혔다.
    - `PatchToolCallsMiddleware.before_agent`는 승인 중 끊긴 턴 다음에 `[RemoveMessage(REMOVE_ALL), *기록 전체]`를 내보낸다. 그래서 옛 답변과 **옛 도구 결과**(이것은 main에도 있던 문제)가 다시 찍히고, `last_text`도 옛 텍스트가 될 수 있었다.
    - 수정: 예산 노드(`budget_stop`)의 메시지만 그린다. 기록을 통째로 다시 쓰는 update는 그리지 않는다(`rewrites_history`).
  - **B2**: 복구 안내도 노드를 보지 않아, 다시 내보낸 옛 안내가 다시 찍히고 `TurnStops.recoveries`가 늘었다. `recovery_notice`가 `TruncationRecoveryMiddleware` 노드의 것만 센다.
  - **K4**: 실제 그래프 간선으로 순서를 고정하는 시험을 넣었다. 복구 `after_model`이 루프 출구이고, 그보다 앞에서 모델 호출 상한이 집계하며, `jump_to "model"`은 `ModelCallLimitMiddleware.before_model`로 간다.
  - **M1**: 안내를 둘로 나눴다. 생각만 하다 잘린 경우와 답변이 끊긴 경우("끊긴 곳부터 이어서 짧게 마무리하라")다.
  - **M2**: 프롬프트 예시를 `];` 바꾸기에서 고유한 끝 표식(`// END-OF-DATA`) 줄을 바꾸는 방식으로 바꿨다. 파일을 나눠도 된다고 적었다. `];`는 여러 번 나올 수 있어 `edit_file`이 실패한다.
  - 시험 4개를 추가하고 M1 확인을 기존 시험에 넣었다. `test_stops.py`가 23개, 전체가 194개다. 수정 전 코드로 돌리면 새 시험이 실패하는 것을 확인했다.

#### 개선 3 — 컨텍스트 한도와 멈춘 응답 정리 (2026-11-06, 사양 `code-agent-specs/task3-context-and-stops.md`)
- **사실 정정**
  - Opus 5.5 캐시 읽기는 $0.20/M(입력의 0.05배)이다. 위 가격 정의의 $0.40과 `caching.py` 주석의 0.1배를 고쳤다.
  - Opus 5.5에는 preserved thinking이 있다. thinking 블록의 서명은 그 앞의 prefix 전체와 직전 thinking 체인에 묶인다. 그래서 앞쪽 기록을 고치면 뒤쪽 thinking이 무효가 되고, 프롬프트 캐시도 깨진다.
- **컨텍스트 한도** (`settings.context_tokens`, `models.with_context_limit`)
  - 기본은 200,000이고 `CODE_AGENT_CONTEXT_TOKENS`로 20,000~1,000,000을 줄 수 있다. 범위 밖이거나 정수가 아니면 시작 때 오류를 낸다.
  - **[V] 프로필**: langchain-openrouter 0.2.8 데이터에는 5.5 계열(opus·sonnet·haiku-5.5, fable-5.1)이 없어 `profile`이 비어 있다. Anthropic 직결 Opus 5.5와 OpenRouter opus-5는 1M이다.
  - `make_model`이 두 공급자 모두에 `max_input_tokens=min(설정, 알려진 창)`을 넣는다. 다른 키는 유지하고, `max_output_tokens`가 없으면 64,000을 넣는다. `model_copy`로 만든다.
  - 비용 근거(session2 시뮬레이션): 5분 넘게 쉬어 캐시가 식으면 최악 입력 비용이 200K는 $0.85, 1M은 $4.25다. 100회를 넘는 긴 세션에서도 200K가 싸다.
- **요약 자리 대체와 인자 잘림 끄기** (`agent.summarization_middleware`)
  - **[V]** 프로필이 없으면 deepagents 기본값은 메시지 20개부터 옛 `write_file`·`edit_file` 인자를 2,000자로 자른다(`compute_summarization_defaults`).
    - 30메시지 시험에서 우리 대체를 빼면 실제로 앞쪽 기록이 바뀌어 실패하는 것을 확인했다.
    - 이것은 캐시만 깨는 것이 아니라 Opus 5.5의 뒤쪽 thinking도 무효로 만든다.
  - deepagents `SummarizationMiddleware`를 같은 이름으로 넘겨 core 자리를 제자리에서 대신한다(graph.py `_apply_custom_middleware`).
    - 설정: `trigger=("tokens", 0.85×한도)`, `keep=("tokens", 0.10×한도)`, `truncate_args_settings=None`
    - fraction은 프로필이 없는 모델에서 생성자가 ValueError를 내서 tokens로 준다.
- **멈춘 응답 정리** (`stops.strip_stopped`, K3·K1)
  - 대상은 도구 호출 없이 멈춘 응답이다. 텍스트만 남기고, 없으면 자리표시를 넣는다(예: `[출력 한도(64,000토큰)에 걸려 잘린 응답: 생각만 하다 끝나 결과물이 없다. 생각 내용은 기록에서 뺐다.]`). Anthropic은 빈 content를 거부한다.
  - 텍스트가 남으면 끝에 `[출력 한도로 여기서 잘림]`을 붙인다.
  - `reasoning_content`, `reasoning_details`, thinking·redacted_thinking 블록(서명 없는 것 포함), 잘린 도구 호출을 뺀다. 사용량과 정지 사유는 남기고, 뺀 것은 `response_metadata["code_agent_stripped"]`에 적는다.
  - **같은 id**로 교체한다. 바뀌는 것은 방금 나온 마지막 메시지뿐이다. 원본은 다음 요청에 실린 적이 없어 캐시에도 없고, 그 뒤에 서명된 thinking도 없다. 그래서 덧붙이기 원칙과 preserved thinking을 깨지 않는다.
  - 효과: 잘린 64K reasoning을 이후 모든 호출이 다시 보내지 않는다(az-quiz 턴 2에서 약 $1.15).
    - 실측: az-quiz의 잘린 응답에는 reasoning 31,160자 외에, 청크 병합으로 이어 붙은 서명(171,425자)과 format 필드("anthropic-claude-v1"이 8,024번)도 있었다.
  - 유효한 호출이 남은 응답은 메시지를 두고, 잘린 호출(`invalid_tool_calls`)의 인자만 `"{}"`로 바꾼다(K2). 그러면 langchain이 '인자가 잘렸다' 결과를 붙인다.
  - 유효한 호출이 없는 잘린 호출은 정리하면서 빼고, 자리표시에 이름을 남긴다.
- **멈춘 응답의 마지막 도구 호출은 실행하지 않는다** (`stops.hold_cut_call`, 오케스트레이터 결정 변경)
  - 처음 결정은 "도구 호출이 있는 응답은 건드리지 않는다"였다. 아래 실측 1에서 langchain-core가 잘린 인자를 `parse_partial_json`으로 고쳐 **유효한 호출**을 만든다는 것이 드러나 바꿨다. 그대로 두면 절반만 쓴 파일이나 잘린 셸 명령(예: 잘린 `rm -rf`)이 실행된다.
  - 출력 한도·오류·컨텍스트 창으로 멈춘 응답의 마지막 호출(쓰다가 멈춘 것)을 `invalid_tool_calls`(args `"{}"`)로 옮긴다. langchain이 '인자가 잘렸다' 결과를 붙이고 모델이 다시 쓴다. 앞의 완결된 호출은 그대로 실행한다. 남는 유효 호출이 없으면 정리·복구 경로로 간다.
    - 컨텍스트 창도 넣었다. 출력 도중 창이 차서 끊긴 것이라 같은 위험이 있다.
  - **`wrap_model_call`에서** 한다. 응답이 state에 들어가기 전이라, HITL `after_model`(우리 `after_model`보다 먼저 돈다)이 잘린 호출의 승인을 묻지 않는다.
  - 마지막 메시지만 바꾸므로 preserved thinking 원칙과 같다.
  - 화면에는 "write_file 호출은 출력이 멈춰 인자가 잘렸을 수 있어 실행하지 않았다." 노란 줄과 턴 끝 경고가 나온다.
- **정지 사유 3분류** (`stop_reason`)
  - `STOP_LENGTH`(length, max_tokens): 정리, 1회 복구, 경고
  - `STOP_ERROR`(OpenRouter error): 정리, 1회 복구, 경고. 복구 1회를 출력 한도와 함께 쓴다.
    - **[V]** choices가 있는 오류 청크는 예외 없이 이 값으로 끝난다. 가짜 SDK로 시험했다.
  - `STOP_CONTEXT`(Anthropic model_context_window_exceeded): 다시 불러도 같으므로 정리만 하고 경고한다("/new 로 새로 시작하거나 요약한 뒤 이어 가라").
  - 경고 문장과 화면의 즉시 안내는 사유별로 다르다.
- **실측 1** (OpenRouter opus-5.5, effort low, max_tokens 1,200, write_file 하나로 숫자 1~1000; 비용 $0.0255)
  - 결과: `finish_reason=length`, `tool_calls=1`, `invalid_tool_calls=0`, reasoning 0토큰
  - **[V] K2의 전제와 달랐다.** langchain-core가 스트림 청크를 합칠 때 `parse_partial_json`으로 잘린 인자를 고쳐 **유효한 호출**을 만든다(`AIMessageChunk.init_tool_calls`).
    - 그래서 출력 한도로 잘린 write_file이 잘린 내용 그대로 실행된다.
    - K2의 `invalid_tool_calls` 정리는 고치기가 실패하는 드문 경우의 안전망이다.
    - 400 여부를 보려던 2·3단계는 전제가 성립하지 않아 돌리지 않았다. 대신 위의 결정 변경으로 이어졌다.
- **확인 실측** (같은 조건, max_tokens 600; 비용 $0.0237, 작업 3 실측 합계 $0.0492)
  - 잘린 응답: `finish_reason=length`. write_file `content`가 983자("273"까지)에서 끊긴 채 유효한 호출이었다.
  - `hold_cut_call` 뒤 `tool_calls=0`, `invalid=[write_file "{}"]`
  - (a) 막은 호출 + '인자가 잘렸다' 결과 + "1~5만" → **200**. 모델이 완결된 write_file을 다시 냈다.
  - (b) 정리한 자리표시 + 복구 안내 → **200**
  - 실제 응답을 대본 모델로 에이전트에 다시 넣어 모두 승인하게 해도 numbers.txt가 생기지 않았고, 승인 요청은 0번이었다.
- **시험**: 전체 234개가 통과한다(+40).
  - `test_context.py` 14개: 프로필 3개, 설정 9개, 요약 자리 대체, 30메시지 append-only
  - `test_stops.py` 26개
    - 정지 사유 7개, 부분 텍스트 표시, Anthropic thinking 제거, 복구 뒤 재정리
    - 정상 응답의 도구 호출 보존 2개, 렌더 안 됨, 사용량 보존, 두 공급자 요청 본문
    - K2 두 가지, 오류 청크, 오류 복구 1회, 컨텍스트 창
    - 잘린 호출 막기 6개: manual에서 승인 안 물음, auto-edit에서 파일 안 바뀜(write·edit), 병렬 중 잘린 것만 막음, 잘린 execute 실행 0회, 단위
  - 기존 시험 1개(잘렸지만 도구 호출이 있으면 재시도하지 않는다)는 새 결정에 맞게 "유일한 잘린 호출은 막고 재시도한다"로 바꿨다.
  - `wrap_model_call`의 막기를 빼면 막기 시험 7개가 실패하는 것을 확인했다.
- **후속**
  - **reasoning_details 병합 부풀림**: langchain-core가 스트림 청크를 `index`로 합칠 때 문자열 필드를 이어 붙여, `format`이 "anthropic-claude-v1"을 청크 수만큼 반복한다. 정상 응답도 4~267번, az-quiz 잘린 응답은 8,024번이었다.
    - langchain-openrouter의 `_merge_reasoning_details`는 항목이 하나뿐이면 그대로 두므로 이 값이 매 요청에 실려 나간다. 지금까지 OpenRouter는 받아들였다.
    - 정리 대상이 아닌 응답에도 생기므로 별도 과제다.
  - OpenRouter 경로에서 요약이 남긴 메시지의 `reasoning_details`(사양 1절)
  - Anthropic 직결 실측(키 없음)

### 4단계 — 평가·확장 (선택)
- `evals/`: 수정 과제 5~10개(fixture 저장소와 숨김 검증 테스트)를 헤드리스로 돌려 통과율, 비용, 시간을 집계한다. `pjt\Day04\coding_skills\verify_agent.py` 패턴을 참고한다.
- 확장 후보:
  - MCP 도구(`langchain-mcp-adapters`)
  - `fetch_url`: dcode처럼 SSRF 방어(사설 IP 차단, 리다이렉트 재검사)를 넣는다.
  - `/model`로 런타임에 모델을 교체한다.
  - `langgraph.json` + `langgraph dev`로 Studio에서 디버깅한다.

## 4. 재사용할 기존 코드 (pjt 기준)
| 용도 | 위치 |
|---|---|
| 코드 에이전트 조립 예시 | `Day04\coding_skills\agent.py:38-136` |
| 명령 기록 셸 백엔드, diff | `Day04\coding_skills\run_files.py` |
| ChatAnthropic 설정 | `Day04_new\cap_lab\settings.py:90-99` |
| HITL + SqliteSaver + resume | `Day08\05_hitl_approval.py`, `Day08\guardlab\runner.py:107-124` |
| 예산 미들웨어 | `Day08\guardlab\budget.py` |
| 트레이스 미들웨어 | `Day08\guardlab\trace.py` |
| 대본 모델(무과금 테스트) | `Day02_new\tests\fakes.py` |
| 내장 스킬 예시 | `Day04\coding_skills\skills\` |
| dcode 설계 분석 | `Day08\dcode-analysis\analysis\02,04,06-*.md` |

## 5. 위험과 대응
| 위험 | 대응 |
|---|---|
| 셸이 샌드박스가 아니다(`virtual_mode`는 파일 도구만 가둔다) | 기본 `manual` 모드, 좁은 allow-list, 셸 env에서 비밀값 제거, `yolo` 경고 |
| Windows: 기본 셸이 cmd.exe이고, 빈 env이고, 타임아웃 때 고아 프로세스가 남고, cp949 디코딩 | `WindowsShellBackend` 오버라이드(1단계)와 그 테스트 |
| Windows: `edit_file`이 CRLF를 LF로 바꾼다(#7488), grep 결과가 cp1252로 디코드된다(#6964), BOM 있는 스킬이 무시된다(#6811) | fixture는 LF로 만들고, 3단계에서 CRLF 보존, 한글 grep 테스트, 스킬은 BOM 없이 저장 |
| `permissions`가 셸 백엔드와 충돌한다(0단계에서 확인) | `.env` 차단을 `wrap_tool_call` 미들웨어(`DotenvGuardMiddleware`)로 구현 |
| Opus 5.5 preserved thinking: 요약 미들웨어가 이력을 바꾸면 thinking 블록 검증에서 400이 날 수 있다 | 긴 세션 테스트로 확인하고, 문제가 생기면 요약 트리거와 모델 설정을 조정 |
| 0.7.14 분석 문서와 0.7.32의 차이 | 0단계 3에서 설치된 소스 기준으로 확인 |

## 6. 검증 방법
1. `uv run ruff check . && uv run pytest -q`: 모델 호출 없이 백엔드, 승인, 에이전트 배선을 검증한다.
2. 수동 E2E(실제 API 사용, 모델 호출 약 10~30회):
   ```
   uv run code-agent --workspace <buggy_repo 복사본>
   > 실패하는 테스트를 찾아 고쳐줘
   ```
   - 탐색 → todo 표시 → `edit_file` 승인 → `pytest` 실행 승인 → 통과 보고 순서로 진행되는지 확인한다.
   - reject하면 파일이 그대로인지 확인한다.
   - 작업 폴더에 오프로드 파일이 생기지 않았는지 확인한다.
   - `/exit` 후 `--resume <id>`로 이어지는지 확인한다.
3. 헤드리스로 `uv run code-agent -n "..." --mode auto-edit`를 실행해 종료 코드와 `trace.jsonl`을 확인한다(2·3단계).

## 7. 참고 자료
- SDK 소스(0.7.32 태그): [graph.py](https://github.com/langchain-ai/deepagents/blob/deepagents%3D%3D0.7.32/libs/deepagents/deepagents/graph.py), [local_shell.py](https://github.com/langchain-ai/deepagents/blob/deepagents%3D%3D0.7.32/libs/deepagents/deepagents/backends/local_shell.py), [CHANGELOG](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/CHANGELOG.md)
- 문서: [customization](https://docs.langchain.com/oss/python/deepagents/customization), [backends](https://docs.langchain.com/oss/python/deepagents/backends), [human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop), [subagents](https://docs.langchain.com/oss/python/deepagents/subagents), [skills](https://docs.langchain.com/oss/python/deepagents/skills), [memory](https://docs.langchain.com/oss/python/deepagents/memory), [streaming](https://docs.langchain.com/oss/python/deepagents/streaming)
- dcode: [agent.py](https://github.com/langchain-ai/deepagents/blob/main/libs/code/deepagents_code/agent.py), [system_prompt.md](https://github.com/langchain-ai/deepagents/blob/main/libs/code/deepagents_code/system_prompt.md), 로컬 분석 `..\pjt\Day08\dcode-analysis\`
