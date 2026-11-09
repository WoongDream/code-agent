# outputs

code-agent 로 만든 결과물을 모아 둔다. 폴더 하나가 결과물 하나다. 사람이 손대지 않은, 에이전트가 만든 그대로의 상태로 커밋한다.

| 결과물 | 만든 날 | 모델 | 사용량 | 요청 |
|---|---|---|---|---|
| [tetris](tetris/) | 2026-11-05 | `openrouter:anthropic/claude-opus-5.5` (effort high) | 모델 19회 · 입력 400,860 · 출력 23,096 · $2.0447 (캐시 기능 전) | HTML/CSS/JS 만으로 브라우저에서 동작하는 테트리스. `index.html` 을 열면 바로 플레이, 빌드·설치·서버 없이. 방향키 이동/회전, 스페이스 하드 드롭, 줄 삭제, 점수, 다음 블록 미리보기, 게임 오버/재시작 |
| [bomber-game](bomber-game/) | 2026-11-05 | `openrouter:anthropic/claude-opus-5.5` | 모델 6회 · 입력 79,016 (캐시 읽기 60,449) · 출력 11,587 · $0.3333 | HTML/CSS/JS 만으로 된 지뢰찾기. `index.html` 을 열면 바로 플레이. 난이도 3단계(9x9·16x16·30x16), 왼쪽 클릭 열기·오른쪽 클릭 깃발, 첫 클릭은 지뢰가 아님, 빈 칸 자동 펼치기, 남은 지뢰 수·경과 시간, 게임 로직과 화면 코드 분리 |
| [az-quiz](az-quiz/) | 2026-11-05 | `openrouter:anthropic/claude-opus-5.5` | 모델 19회 · 입력 1,966,802 (캐시 읽기 1,584,121) · 출력 111,495 · $4.4155 | HTML/CSS/JS 만으로 된 아재 퀴즈. 150문제 이상(question·answers·explain·category), 끝없이 이어지는 모드, 제한 시간 선택(10/20/30/60초), 띄어쓰기·문장부호를 무시한 채점, 정답률 표시, 휴대폰 화면 지원 |
| [pokemon](pokemon/) | 2026-11-06 | `openrouter:anthropic/claude-opus-5.5` | 모델 33회 · 입력 1,955,246 (캐시 읽기 1,851,659) · 출력 67,841 · $2.2226 | 포켓몬 몸무게 맞히기 퀴즈. PokeAPI 에서 전국도감 1~1025번을 받는 Python 표준 라이브러리 수집 스크립트(`scripts/fetch_pokemon.py`)와 빌드 없는 HTML/CSS/JS 웹앱. 문제 수·문제당 시간 선택, 무작위 출제, 채점, 끝나면 통계 |

## 새 결과물 만들기

저장소 루트에서 결과물 폴더를 작업 폴더로 지정해 실행한다.

```powershell
mkdir outputs\<이름>
uv run code-agent --workspace outputs\<이름> --mode auto-edit
```

- 끝나면 `/cost` 로 본 사용량과 요청을 위 표에 한 줄 추가한다.
- 작업 폴더가 이 저장소 안이라, 에이전트가 실행하는 `git` 명령은 code-agent 저장소를 대상으로 한다. 에이전트는 요청하지 않으면 commit 하지 않는다.
- 결과물 안에 `.git` 을 만들지 않는다. 상위 저장소에서 중첩 저장소(gitlink)로 잡혀 파일이 커밋되지 않는다.
