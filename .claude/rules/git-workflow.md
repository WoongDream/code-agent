# Git 작업 규칙

## 브랜치
- `main`: 안정 버전. 직접 커밋하지 않는다. dev 가 안정되면 dev → main PR 로 올린다.
- `dev`: 기본 작업 브랜치. 모든 작업 브랜치는 dev 에서 따고 dev 로 돌아온다.
- 작업 브랜치 이름은 `<type>/<하는-일>-woong` 이다. 하는 일은 영어 소문자, 단어 사이는 `-`.
  - 예: `feat/approval-ux-woong`, `fix/resume-crash-woong`, `docs/git-convention-woong`
- type 은 브랜치와 커밋에서 같은 단어를 쓴다.

| type | 쓰는 때 |
|---|---|
| feat | 새 기능 |
| fix | 버그 수정 |
| refactor | 동작을 바꾸지 않는 코드 정리 |
| docs | 문서 |
| test | 시험 추가·수정 |
| chore | 그 밖의 잡일(의존성, 설정, outputs 결과물 추가 등) |
| ci | GitHub Actions 등 CI |

## 작업 폴더
- 저장소 폴더 자체는 항상 `dev` 에 둔다. 사용자가 그 폴더에서 code-agent 를 실행한다.
- 작업 브랜치는 git worktree 로 연다: `..\code-agent-wt\<브랜치의 / 를 - 로 바꾼 이름>`
  - 예: `git worktree add ../code-agent-wt/feat-approval-ux-woong -b feat/approval-ux-woong dev`

## 커밋
- 제목은 `[type] 한 일` 이다. 한국어로 쓴다.
  - 예: `[fix] resume 때 옛 생각이 요청에 섞이는 문제 수정`
- 제목 아래 한 줄을 비우고, 본문에 무엇을 왜 바꿨는지 쓴다.

## PR
- base 는 `dev` 로 연다. 제목은 커밋 제목과 같은 `[type] 한 일` 형식이다.
- 병합은 사용자가 한다.

## outputs
- code-agent 로 만든 결과물(`outputs/<이름>/`)도 커밋한다. 에이전트가 만든 그대로 올린다.
- 결과물을 추가할 때마다 `outputs/README.md` 표에 한 줄을 넣는다.
