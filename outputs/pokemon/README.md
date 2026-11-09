# 포켓몬 몸무게 맞히기

포켓몬 이미지와 도감 정보를 보고 몸무게(kg)를 맞히는 퀴즈 웹앱입니다. 빌드 단계 없는 순수 HTML/CSS/JS 이며,
데이터는 [PokeAPI](https://pokeapi.co/) 에서 받았습니다(전국도감 1~1025번 기본 폼). 이미지는 내려받지 않고 PokeAPI 스프라이트 URL 을 핫링크합니다.

## 실행 방법

- `index.html` 을 더블클릭해 브라우저로 엽니다(file:// 로 동작). 이미지 표시에는 인터넷 연결이 필요합니다.
- 정적 호스팅(GitHub Pages 등)에는 폴더를 그대로 올리면 됩니다. 로컬 서버로 보고 싶다면:
  `python -m http.server 8000` 후 http://localhost:8000/

대문에서 문제 수(5/10/20/30 또는 직접 입력 1~100)와 문제당 시간(10/20/30/60초 또는 직접 입력 5~180초)을 고르고 시작합니다.
마지막 설정은 브라우저 localStorage 에 기억됩니다.

## 정답 기준

- **|추측 − 실제| ≤ max(실제 × 20%, 0.5kg)** 이면 정답. (`rules.js` 의 `TOLERANCE_RATIO`, `TOLERANCE_MIN_KG`)
- 오차율 = (추측 − 실제) / 실제 × 100
- 시간 초과는 입력값과 상관없이 오답이며, 오차율·추측 경향·평균 응답 시간 통계에서 뺍니다.
- 답은 0 초과 10,000 미만, 소수 첫째 자리까지 입력할 수 있습니다.
- 추측 경향은 (추측 / 실제) 비율의 기하평균입니다. "가장 가까웠던/빗나간 문제"도 비율 기준(|log(추측/실제)|)으로 고릅니다.
  그래서 10배 무겁게 본 것과 10배 가볍게 본 것을 같은 크기의 실수로 봅니다.
- 답을 내기 전에는 정답 몸무게가 화면과 DOM 에 나오지 않습니다("??? kg"). 도감 설명 중 몸무게를 드러내는 문장(숫자+kg, 킬로, pounds, lbs 등)은 수집할 때 다른 버전 설명으로 바꿉니다.

## 데이터 다시 받기

Python 3 표준 라이브러리만 씁니다(별도 설치 불필요).

```bash
python scripts/fetch_pokemon.py           # 이어 받기: scripts/.cache/ 에 있는 건 건너뜀
python scripts/fetch_pokemon.py --fresh   # 캐시를 지우고 처음부터
```

- `/pokemon/{id}`, `/pokemon-species/{id}` 를 포켓몬마다, `/ability/{name}` 을 고유 특성마다 한 번 부릅니다(약 2,300건, 동시 6개, 1~2분).
- 타임아웃 20초, 실패하면 지수 백오프(1·2·4초)로 3번까지 재시도. 필요한 필드만 뽑아 `scripts/.cache/*.jsonl` 에 한 줄씩 저장하므로 중간에 끊겨도 다시 실행하면 이어 받습니다.
- 인증서·네트워크 오류가 나면 인증서 검증을 끄지 않고 오류를 보여 준 뒤 멈춥니다.
- 마지막에 #1 이상해씨 6.9kg·0.7m, #25 피카츄 6.0kg·0.4m, #143 잠만보 460.0kg 을 확인하고, 성공 수·실패 id·몸무게 최소/최대·필드별 null 개수를 출력합니다. 검증에 실패하거나 500마리 미만이면 데이터 파일을 쓰지 않습니다.
- 출력: `data/pokemon-data.js`(앱이 읽음, `window.POKEMON_DATA = [...]`), `data/pokemon.json`(같은 내용).

PokeAPI 의 `weight` 는 hg(0.1kg), `height` 는 dm(0.1m) 단위라서 10 으로 나눠 kg·m 로 저장합니다.

## 파일 구조

```
index.html              한 페이지(대문 → 퀴즈 → 결과)
style.css               스타일(데스크톱·모바일 375px)
rules.js                규칙 상수·순수 함수(출제, 채점, 오차율, 통계)와 타입·색·서식지 한국어 매핑 표
app.js                  화면 처리(설정, 타이머, 이미지 로딩, 채점 패널, 결과 통계·산점도)
data/pokemon-data.js    앱이 읽는 데이터
data/pokemon.json       같은 데이터(JSON)
scripts/fetch_pokemon.py  데이터 수집 스크립트
scripts/.cache/         수집 캐시(git 에서 제외)
```

## 레코드 형식

```json
{"id":25,"nameKo":"피카츄","nameEn":"Pikachu","genusKo":"쥐포켓몬","types":["electric"],
 "heightM":0.4,"weightKg":6.0,"abilities":[{"nameKo":"정전기","nameEn":"Static","hidden":false}],
 "stats":{"hp":35,"attack":55,"defense":40,"spAttack":50,"spDefense":50,"speed":90},
 "generation":1,"color":"yellow","habitat":"forest","isLegendary":false,"isMythical":false,"isBaby":false,
 "evolvesFromKo":"피츄","flavorText":"...","image":"https://...","imageFallback":"https://..."}
```

값이 없으면 `null` 입니다. `habitat` 은 3세대까지만 있어 이후 세대는 `null` 이 정상입니다.
