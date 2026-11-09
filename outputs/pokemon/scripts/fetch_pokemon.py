#!/usr/bin/env python3
"""PokeAPI v2 에서 전국도감 1~1025번 기본 폼 데이터를 받아 퀴즈용 데이터 파일을 만든다.

표준 라이브러리만 사용한다.

    python scripts/fetch_pokemon.py            # 이어 받기(캐시 사용)
    python scripts/fetch_pokemon.py --fresh    # 캐시를 지우고 처음부터

출력: data/pokemon-data.js (window.POKEMON_DATA = [...]), data/pokemon.json
캐시: scripts/.cache/*.jsonl (필요한 필드만 뽑은 결과를 한 줄씩 덧붙인다)
"""

import argparse
import json
import os
import re
import shutil
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://pokeapi.co/api/v2"
FIRST_ID, LAST_ID = 1, 1025
MIN_SUCCESS = 500
WORKERS = 6
TIMEOUT = 20  # 초
RETRIES = 3  # 첫 시도 뒤 재시도 횟수
BACKOFF = 1.0  # 1, 2, 4초
USER_AGENT = "pokemon-weight-quiz-fetcher/1.0 (+https://pokeapi.co; educational project)"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "scripts", ".cache")
DATA_DIR = os.path.join(ROOT, "data")

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9}
STAT_KEYS = {
    "hp": "hp",
    "attack": "attack",
    "defense": "defense",
    "special-attack": "spAttack",
    "special-defense": "spDefense",
    "speed": "speed",
}
# 몸무게를 드러내는 표현. 하나라도 있으면 그 도감 설명은 쓰지 않는다.
WEIGHT_PATTERNS = re.compile(
    r"\d[\d,.]*\s*(?:kg|㎏|킬로|kilo|pound|lb)|(?<![a-z])kg(?![a-z])|㎏|킬로|kilogram|\bpounds?\b|\blbs?\b",
    re.IGNORECASE,
)

EXPECTED = {
    1: {"nameKo": "이상해씨", "weightKg": 6.9, "heightM": 0.7},
    25: {"nameKo": "피카츄", "weightKg": 6.0, "heightM": 0.4},
    143: {"nameKo": "잠만보", "weightKg": 460.0},
}


class FatalError(Exception):
    """인증서·네트워크 문제처럼 계속해도 소용없는 오류."""


class NotFound(Exception):
    pass


# ---------------------------------------------------------------- HTTP

def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last = None
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(url)
            last = e  # 429/5xx 등은 재시도
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLError):
                raise FatalError(f"SSL/인증서 오류: {e.reason} ({url})")
            last = e
        except ssl.SSLError as e:
            raise FatalError(f"SSL/인증서 오류: {e} ({url})")
        except (socket.timeout, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = e
        if attempt < RETRIES:
            time.sleep(BACKOFF * (2 ** attempt))
    raise RuntimeError(f"{url}: {last!r}")


# ---------------------------------------------------------------- 필드 추출

def lang_value(entries, lang, key):
    for e in entries:
        if e["language"]["name"] == lang:
            return e[key]
    return None


def id_from_url(url):
    return int(url.rstrip("/").rsplit("/", 1)[1])


def extract_pokemon(d):
    sprites = d.get("sprites") or {}
    artwork = ((sprites.get("other") or {}).get("official-artwork") or {}).get("front_default")
    stats = {}
    for s in d.get("stats", []):
        key = STAT_KEYS.get(s["stat"]["name"])
        if key:
            stats[key] = s["base_stat"]
    return {
        "id": d["id"],
        "types": [t["type"]["name"] for t in sorted(d.get("types", []), key=lambda t: t["slot"])],
        "height": d.get("height"),
        "weight": d.get("weight"),
        "abilities": [
            {"name": a["ability"]["name"], "hidden": bool(a["is_hidden"])}
            for a in sorted(d.get("abilities", []), key=lambda a: a["slot"])
        ],
        "stats": stats,
        "image": artwork,
        "imageFallback": sprites.get("front_default"),
    }


def extract_species(d):
    gen = None
    if d.get("generation"):
        gen = ROMAN.get(d["generation"]["name"].split("-", 1)[1])
    flavors, seen = [], set()
    for e in d.get("flavor_text_entries", []):
        lang = e["language"]["name"]
        if lang not in ("ko", "en"):
            continue
        text = clean_text(e["flavor_text"])
        if text and (lang, text) not in seen:
            seen.add((lang, text))
            flavors.append({"lang": lang, "version": e["version"]["name"], "text": text})
    evo = d.get("evolves_from_species")
    return {
        "id": d["id"],
        "nameKo": lang_value(d.get("names", []), "ko", "name"),
        "nameEn": lang_value(d.get("names", []), "en", "name"),
        "genusKo": lang_value(d.get("genera", []), "ko", "genus"),
        "generation": gen,
        "color": (d.get("color") or {}).get("name"),
        "habitat": (d.get("habitat") or {}).get("name"),
        "isLegendary": bool(d.get("is_legendary")),
        "isMythical": bool(d.get("is_mythical")),
        "isBaby": bool(d.get("is_baby")),
        "evolvesFromId": id_from_url(evo["url"]) if evo else None,
        "flavors": flavors,
    }


def extract_ability(d):
    return {
        "name": d["name"],
        "nameKo": lang_value(d.get("names", []), "ko", "name"),
        "nameEn": lang_value(d.get("names", []), "en", "name"),
    }


def clean_text(s):
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[\f\n\r\t\v\u000c\u2028\u2029]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def reveals_weight(text):
    return bool(WEIGHT_PATTERNS.search(text))


def pick_flavor(flavors):
    """ko 우선(최신 버전부터), 없으면 en. 몸무게를 드러내는 설명은 건너뛴다."""
    for lang in ("ko", "en"):
        for f in reversed([f for f in flavors if f["lang"] == lang]):
            if not reveals_weight(f["text"]):
                return f["text"]
    return None


# ---------------------------------------------------------------- 캐시

class JsonlCache:
    def __init__(self, name, key):
        self.path = os.path.join(CACHE_DIR, name + ".jsonl")
        self.key = key
        self.items = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 중간에 끊겨 잘린 줄
                    self.items[obj[key]] = obj
        self._fh = None

    def add(self, obj):
        if self._fh is None:
            os.makedirs(CACHE_DIR, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.items[obj[self.key]] = obj

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def run_jobs(label, jobs, on_done):
    """jobs: [(key, url, extractor)]. 실패한 key 목록을 돌려준다."""
    total = len(jobs)
    if total == 0:
        print(f"[{label}] 모두 캐시에 있음")
        return []
    print(f"[{label}] {total}건 요청 시작 (동시 {WORKERS}개)")
    failed, done = [], 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(lambda u, f: f(get_json(u)), url, fn): key for key, url, fn in jobs}
        try:
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    on_done(fut.result())
                except FatalError:
                    raise
                except Exception as e:  # noqa: BLE001
                    failed.append(key)
                    print(f"  ! {label} {key} 실패: {e}", file=sys.stderr)
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  {label}: {done}/{total} ({time.time() - started:.0f}초, 실패 {len(failed)})")
        except (FatalError, KeyboardInterrupt):
            for f in futures:
                f.cancel()
            raise
    return failed


# ---------------------------------------------------------------- 조립

def build_records(pokemon, species, abilities):
    records, failed = [], []
    for pid in range(FIRST_ID, LAST_ID + 1):
        p, s = pokemon.get(pid), species.get(pid)
        if not p or not s:
            failed.append(pid)
            continue
        if p["weight"] is None or p["weight"] <= 0:
            print(f"  - #{pid}: 몸무게가 0 이하라 제외")
            failed.append(pid)
            continue
        evo = species.get(s["evolvesFromId"]) if s["evolvesFromId"] else None
        records.append({
            "id": pid,
            "nameKo": s["nameKo"],
            "nameEn": s["nameEn"],
            "genusKo": s["genusKo"],
            "types": p["types"],
            "heightM": round(p["height"] / 10, 1) if p["height"] is not None else None,
            "weightKg": round(p["weight"] / 10, 1),
            "abilities": [
                {
                    "nameKo": (abilities.get(a["name"]) or {}).get("nameKo"),
                    "nameEn": (abilities.get(a["name"]) or {}).get("nameEn") or a["name"],
                    "hidden": a["hidden"],
                }
                for a in p["abilities"]
            ],
            "stats": p["stats"] if len(p["stats"]) == 6 else None,
            "generation": s["generation"],
            "color": s["color"],
            "habitat": s["habitat"],
            "isLegendary": s["isLegendary"],
            "isMythical": s["isMythical"],
            "isBaby": s["isBaby"],
            "evolvesFromKo": evo["nameKo"] if evo else None,
            "flavorText": pick_flavor(s["flavors"]),
            "image": p["image"],
            "imageFallback": p["imageFallback"],
        })
    return records, failed


def verify(records):
    by_id = {r["id"]: r for r in records}
    ok = True
    for pid, exp in EXPECTED.items():
        r = by_id.get(pid)
        if not r:
            print(f"  ? #{pid}: 데이터 없음(수집 실패)")
            continue
        bad = [(k, v, r[k]) for k, v in exp.items() if r[k] != v]
        for k, v, got in bad:
            ok = False
            print(f"  ✗ #{pid} {k}: 기대 {v!r}, 실제 {got!r}")
        if not bad:
            print(f"  ✓ #{pid} {r['nameKo']} {r['weightKg']}kg {r['heightM']}m")
    leaked = [r["id"] for r in records if r["flavorText"] and reveals_weight(r["flavorText"])]
    if leaked:
        ok = False
        print(f"  ✗ 몸무게가 드러나는 도감 설명: {leaked}")
    return ok


def write_outputs(records):
    os.makedirs(DATA_DIR, exist_ok=True)
    body = "[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records) + "\n]"
    with open(os.path.join(DATA_DIR, "pokemon.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(body + "\n")
    with open(os.path.join(DATA_DIR, "pokemon-data.js"), "w", encoding="utf-8", newline="\n") as f:
        f.write("// scripts/fetch_pokemon.py 가 만든 파일. 직접 고치지 마세요. 출처: PokeAPI (https://pokeapi.co)\n")
        f.write("window.POKEMON_DATA = " + body + ";\n")


def print_summary(records, failed):
    print("\n===== 요약 =====")
    print(f"성공: {len(records)}마리")
    print(f"실패/제외 id ({len(failed)}): {failed if failed else '없음'}")
    if records:
        lo = min(records, key=lambda r: r["weightKg"])
        hi = max(records, key=lambda r: r["weightKg"])
        print(f"몸무게 최소: #{lo['id']} {lo['nameKo']} {lo['weightKg']}kg")
        print(f"몸무게 최대: #{hi['id']} {hi['nameKo']} {hi['weightKg']}kg")
    print("필드별 null 개수:")
    for k in records[0].keys() if records else []:
        n = sum(1 for r in records if r[k] is None)
        print(f"  {k:14s} {n}")
    n_ab = sum(1 for r in records for a in r["abilities"] if a["nameKo"] is None)
    print(f"  {'abilities[].nameKo':14s} {n_ab}")


def main():
    for stream in (sys.stdout, sys.stderr):  # Windows 콘솔(cp949)에서 한글·기호 출력이 깨지지 않게
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="캐시를 지우고 처음부터 받는다")
    args = ap.parse_args()
    if args.fresh and os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    # 사전 점검: 인증서·네트워크 문제면 여기서 멈춘다.
    try:
        get_json(f"{API}/pokemon-species/1")
    except FatalError as e:
        print(f"중단: {e}\n인증서 검증은 끄지 않습니다. 네트워크/프록시/인증서 설정을 확인하세요.", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"중단: PokeAPI 에 연결할 수 없습니다: {e}", file=sys.stderr)
        return 2

    pokemon = JsonlCache("pokemon", "id")
    species = JsonlCache("species", "id")
    abilities = JsonlCache("ability", "name")
    ids = range(FIRST_ID, LAST_ID + 1)
    print(f"캐시: pokemon {len(pokemon.items)}, species {len(species.items)}, ability {len(abilities.items)}")

    try:
        jobs = [(i, f"{API}/pokemon/{i}", extract_pokemon) for i in ids if i not in pokemon.items]
        jobs += [(f"s{i}", f"{API}/pokemon-species/{i}", extract_species) for i in ids if i not in species.items]

        def on_done(obj):
            (species if "flavors" in obj else pokemon).add(obj)

        run_jobs("pokemon+species", jobs, on_done)

        names = sorted({a["name"] for p in pokemon.items.values() for a in p["abilities"]})
        jobs = [(n, f"{API}/ability/{n}", extract_ability) for n in names if n not in abilities.items]
        run_jobs("ability", jobs, abilities.add)
    except FatalError as e:
        print(f"\n중단: {e}\n인증서 검증은 끄지 않습니다. 받은 만큼은 캐시에 남아 있어 다시 실행하면 이어 받습니다.",
              file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n중단됨. 다시 실행하면 캐시에서 이어 받습니다.", file=sys.stderr)
        return 130
    finally:
        for c in (pokemon, species, abilities):
            c.close()

    records, failed = build_records(pokemon.items, species.items, abilities.items)
    print("\n표본 검증:")
    ok = verify(records)
    print_summary(records, failed)
    if not ok:
        print("\n검증 실패: 데이터 파일을 쓰지 않습니다.", file=sys.stderr)
        return 1
    if len(records) < MIN_SUCCESS:
        print(f"\n성공 {len(records)}마리 < {MIN_SUCCESS}: 데이터 파일을 쓰지 않습니다. 다시 실행해 이어 받으세요.",
              file=sys.stderr)
        return 1
    write_outputs(records)
    print(f"\n저장: data/pokemon-data.js, data/pokemon.json ({len(records)}마리)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
