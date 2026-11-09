/* 퀴즈 규칙: 상수, 매핑 표, 순수 함수. DOM 에 의존하지 않는다. */
(function (global) {
  "use strict";

  var C = Object.freeze({
    TOLERANCE_RATIO: 0.2, // 실제 몸무게의 20%
    TOLERANCE_MIN_KG: 0.5, // 최소 허용 오차 0.5kg
    QUESTION_PRESETS: [5, 10, 20, 30],
    QUESTION_MIN: 1,
    QUESTION_MAX: 100,
    QUESTION_DEFAULT: 10,
    TIME_PRESETS: [10, 20, 30, 60],
    TIME_MIN: 5,
    TIME_MAX: 180,
    TIME_DEFAULT: 20,
    ANSWER_MIN_EXCLUSIVE: 0,
    ANSWER_MAX_EXCLUSIVE: 10000,
    IMAGE_WAIT_MS: 3000,
    WARN_SECONDS: 5,
    STAT_MAX: 255,
    WEIGHT_CLASSES: [
      { key: "light", label: "10kg 미만", min: 0, max: 10 },
      { key: "middle", label: "10~100kg", min: 10, max: 100 },
      { key: "heavy", label: "100kg 이상", min: 100, max: Infinity }
    ]
  });

  var TYPES = {
    normal: { ko: "노말", color: "#A8A77A", text: "#fff" },
    fire: { ko: "불꽃", color: "#EE8130", text: "#fff" },
    water: { ko: "물", color: "#6390F0", text: "#fff" },
    electric: { ko: "전기", color: "#F7D02C", text: "#3a3000" },
    grass: { ko: "풀", color: "#7AC74C", text: "#fff" },
    ice: { ko: "얼음", color: "#96D9D6", text: "#0f3b3a" },
    fighting: { ko: "격투", color: "#C22E28", text: "#fff" },
    poison: { ko: "독", color: "#A33EA1", text: "#fff" },
    ground: { ko: "땅", color: "#E2BF65", text: "#3d2e00" },
    flying: { ko: "비행", color: "#A98FF3", text: "#fff" },
    psychic: { ko: "에스퍼", color: "#F95587", text: "#fff" },
    bug: { ko: "벌레", color: "#A6B91A", text: "#fff" },
    rock: { ko: "바위", color: "#B6A136", text: "#fff" },
    ghost: { ko: "고스트", color: "#735797", text: "#fff" },
    dragon: { ko: "드래곤", color: "#6F35FC", text: "#fff" },
    dark: { ko: "악", color: "#705746", text: "#fff" },
    steel: { ko: "강철", color: "#B7B7CE", text: "#25253a" },
    fairy: { ko: "페어리", color: "#D685AD", text: "#fff" }
  };

  var COLORS = {
    black: { ko: "검정", hex: "#333333" },
    blue: { ko: "파랑", hex: "#3b82f6" },
    brown: { ko: "갈색", hex: "#92603a" },
    gray: { ko: "회색", hex: "#9ca3af" },
    green: { ko: "초록", hex: "#22c55e" },
    pink: { ko: "분홍", hex: "#f472b6" },
    purple: { ko: "보라", hex: "#a855f7" },
    red: { ko: "빨강", hex: "#ef4444" },
    white: { ko: "흰색", hex: "#ffffff" },
    yellow: { ko: "노랑", hex: "#facc15" }
  };

  var HABITATS = {
    cave: "동굴",
    forest: "숲",
    grassland: "초원",
    mountain: "산",
    rare: "희귀",
    "rough-terrain": "험한 지형",
    sea: "바다",
    urban: "도시",
    "waters-edge": "물가"
  };

  var STATS = [
    { key: "hp", ko: "HP", color: "#ef4444" },
    { key: "attack", ko: "공격", color: "#f97316" },
    { key: "defense", ko: "방어", color: "#eab308" },
    { key: "spAttack", ko: "특수공격", color: "#3b82f6" },
    { key: "spDefense", ko: "특수방어", color: "#22c55e" },
    { key: "speed", ko: "스피드", color: "#ec4899" }
  ];

  /** Fisher–Yates 셔플. 원본은 건드리지 않는다. */
  function shuffle(arr, rng) {
    rng = rng || Math.random;
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
    return a;
  }

  /** 전체에서 중복 없이 n마리. */
  function pickQuestions(pool, n, rng) {
    return shuffle(pool, rng).slice(0, Math.min(n, pool.length));
  }

  function tolerance(actual) {
    return Math.max(actual * C.TOLERANCE_RATIO, C.TOLERANCE_MIN_KG);
  }

  function isCorrect(guess, actual) {
    // 부동소수 오차로 경계값이 떨어지지 않게 아주 작은 여유를 둔다.
    return Math.abs(guess - actual) <= tolerance(actual) + 1e-9;
  }

  function errorPercent(guess, actual) {
    return ((guess - actual) / actual) * 100;
  }

  /** 입력 문자열 → 유효하면 숫자(소수 첫째 자리까지), 아니면 null. */
  function parseAnswer(str) {
    var s = String(str == null ? "" : str).trim().replace(/,/g, "");
    if (!/^(\d+(\.\d?)?|\.\d)$/.test(s)) return null;
    var v = Number(s);
    if (!isFinite(v) || v <= C.ANSWER_MIN_EXCLUSIVE || v >= C.ANSWER_MAX_EXCLUSIVE) return null;
    return Math.round(v * 10) / 10;
  }

  /** 한 문제 채점. guess 는 숫자 또는 null. */
  function grade(pokemon, guess, timedOut, timeMs) {
    var actual = pokemon.weightKg;
    var hasGuess = typeof guess === "number" && isFinite(guess);
    var correct = !timedOut && hasGuess && isCorrect(guess, actual);
    return {
      pokemon: pokemon,
      guess: hasGuess ? guess : null,
      timedOut: !!timedOut,
      correct: correct,
      errorPct: !timedOut && hasGuess ? errorPercent(guess, actual) : null,
      timeMs: timeMs
    };
  }

  function clampInt(v, min, max) {
    var n = Math.round(Number(v));
    if (!isFinite(n)) return null;
    return n >= min && n <= max ? n : null;
  }

  function validQuestionCount(v, poolSize) {
    return clampInt(v, C.QUESTION_MIN, Math.min(C.QUESTION_MAX, poolSize));
  }

  function validSeconds(v) {
    return clampInt(v, C.TIME_MIN, C.TIME_MAX);
  }

  function median(nums) {
    if (!nums.length) return null;
    var a = nums.slice().sort(function (x, y) { return x - y; });
    var m = Math.floor(a.length / 2);
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  }

  /** 추측/실제 비율의 기하평균. */
  function geometricMeanRatio(pairs) {
    if (!pairs.length) return null;
    var s = 0;
    pairs.forEach(function (p) { s += Math.log(p.guess / p.actual); });
    return Math.exp(s / pairs.length);
  }

  function weightClassOf(kg) {
    for (var i = 0; i < C.WEIGHT_CLASSES.length; i++) {
      var c = C.WEIGHT_CLASSES[i];
      if (kg >= c.min && kg < c.max) return c;
    }
    return C.WEIGHT_CLASSES[C.WEIGHT_CLASSES.length - 1];
  }

  /** 빗나간 정도: 비율 기준(10배 크게 = 10배 작게). */
  function missSize(r) {
    return Math.abs(Math.log(r.guess / r.pokemon.weightKg));
  }

  function computeStats(results) {
    var total = results.length;
    var correct = results.filter(function (r) { return r.correct; }).length;
    var graded = results.filter(function (r) { return !r.timedOut && r.guess != null; });
    var timeouts = results.filter(function (r) { return r.timedOut; }).length;

    var closest = null, farthest = null;
    graded.forEach(function (r) {
      if (!closest || missSize(r) < missSize(closest)) closest = r;
      if (!farthest || missSize(r) > missSize(farthest)) farthest = r;
    });

    var classes = C.WEIGHT_CLASSES.map(function (c) {
      return { key: c.key, label: c.label, total: 0, correct: 0 };
    });
    var typeMap = {};
    results.forEach(function (r) {
      var cls = weightClassOf(r.pokemon.weightKg);
      classes.forEach(function (c) {
        if (c.key === cls.key) { c.total++; if (r.correct) c.correct++; }
      });
      r.pokemon.types.forEach(function (t) {
        if (!typeMap[t]) typeMap[t] = { type: t, total: 0, correct: 0 };
        typeMap[t].total++;
        if (r.correct) typeMap[t].correct++;
      });
    });
    var types = Object.keys(typeMap).map(function (k) { return typeMap[k]; });
    types.sort(function (a, b) { return b.total - a.total || b.correct / b.total - a.correct / a.total; });

    var times = graded.map(function (r) { return r.timeMs; });
    return {
      total: total,
      correct: correct,
      accuracy: total ? correct / total : null,
      medianAbsErrorPct: median(graded.map(function (r) { return Math.abs(r.errorPct); })),
      timeouts: timeouts,
      avgTimeMs: times.length ? times.reduce(function (a, b) { return a + b; }, 0) / times.length : null,
      biasRatio: geometricMeanRatio(graded.map(function (r) { return { guess: r.guess, actual: r.pokemon.weightKg }; })),
      closest: closest,
      farthest: farthest,
      classes: classes,
      types: types,
      graded: graded
    };
  }

  // ---- 표시용 문자열(순수 함수)
  function formatKg(n) {
    return n.toLocaleString("ko-KR", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " kg";
  }

  function formatPctAbs(p) {
    var a = Math.abs(p);
    return (a < 10 ? a.toFixed(1).replace(/\.0$/, "") : String(Math.round(a))) + "%";
  }

  function describeError(p) {
    if (p == null) return "—";
    if (Math.abs(p) < 0.05) return "실제와 같음";
    return "실제보다 " + formatPctAbs(p) + (p > 0 ? " 무거움" : " 가벼움");
  }

  function formatSignedPct(p) {
    if (p == null) return "—";
    if (Math.abs(p) < 0.05) return "0%";
    return (p > 0 ? "+" : "−") + formatPctAbs(p);
  }

  function describeBias(ratio) {
    if (ratio == null) return null;
    var shown = ratio >= 10 ? ratio.toFixed(0) : ratio >= 1 ? ratio.toFixed(1) : ratio.toFixed(2);
    var text = "평균적으로 실제의 " + shown + "배로 추측했어요";
    if (ratio >= 1.05) return text + " (무겁게 보는 편)";
    if (ratio <= 0.95) return text + " (가볍게 보는 편)";
    return text + " (감각이 정확한 편)";
  }

  function formatSeconds(ms) {
    return (ms / 1000).toFixed(1) + "초";
  }

  function dexNo(id) {
    return "No." + String(id).padStart(4, "0");
  }

  global.Rules = {
    C: C,
    TYPES: TYPES,
    COLORS: COLORS,
    HABITATS: HABITATS,
    STATS: STATS,
    shuffle: shuffle,
    pickQuestions: pickQuestions,
    tolerance: tolerance,
    isCorrect: isCorrect,
    errorPercent: errorPercent,
    parseAnswer: parseAnswer,
    grade: grade,
    validQuestionCount: validQuestionCount,
    validSeconds: validSeconds,
    median: median,
    geometricMeanRatio: geometricMeanRatio,
    weightClassOf: weightClassOf,
    computeStats: computeStats,
    formatKg: formatKg,
    describeError: describeError,
    formatSignedPct: formatSignedPct,
    describeBias: describeBias,
    formatSeconds: formatSeconds,
    dexNo: dexNo
  };
})(window);
