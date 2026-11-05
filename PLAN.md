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
│   ├── approval.py         # 승인 모드, 셸 allow-list, HITL 결정 생성 (순수 함수)
│   ├── render.py           # 스트림 → rich 터미널 출력
│   ├── cli.py              # argparse, REPL, 슬래시 커맨드, interrupt/resume 루프
│   ├── subagents.py        # (2단계)
│   ├── tracing.py          # (3단계)
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
