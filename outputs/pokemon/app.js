/* 화면 처리: 대문 → 퀴즈 → 결과. 규칙과 계산은 rules.js(window.Rules)에 있다. */
(function () {
  "use strict";

  const R = window.Rules;
  const C = R.C;
  const POOL = (Array.isArray(window.POKEMON_DATA) ? window.POKEMON_DATA : []).filter(
    (p) => p && typeof p.weightKg === "number" && p.weightKg > 0 && Array.isArray(p.types)
  );
  const STORE_KEY = "pokemonWeightQuiz.settings.v1";
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------ DOM 도우미
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        const v = attrs[k];
        if (v == null || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "style") node.style.cssText = v;
        else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
        else node.setAttribute(k, v === true ? "" : v);
      }
    }
    (Array.isArray(children) ? children : children == null ? [] : [children]).forEach((c) => {
      if (c == null || c === false) return;
      node.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    });
    return node;
  }

  function svg(tag, attrs, children) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k in attrs || {}) node.setAttribute(k, attrs[k]);
    (children || []).forEach((c) => node.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return node;
  }

  function typeBadge(t) {
    const info = R.TYPES[t] || { ko: t, color: "#888", text: "#fff" };
    return el("span", { class: "type-badge", style: `background:${info.color};color:${info.text}`, text: info.ko });
  }

  function pct(n) {
    return Math.round(n * 100) + "%";
  }

  function showScreen(name) {
    ["home", "quiz", "result"].forEach((s) => {
      $("screen-" + s).hidden = s !== name;
    });
    document.body.dataset.screen = name;
    window.scrollTo(0, 0);
  }

  // ------------------------------------------------------------ 설정 저장
  function loadSettings() {
    const s = { questions: C.QUESTION_DEFAULT, seconds: C.TIME_DEFAULT };
    try {
      const raw = window.localStorage.getItem(STORE_KEY);
      if (raw) {
        const o = JSON.parse(raw);
        const q = R.validQuestionCount(o && o.questions, POOL.length);
        const t = R.validSeconds(o && o.seconds);
        if (q) s.questions = q;
        if (t) s.seconds = t;
      }
    } catch (e) {
      /* 읽기 실패 → 기본값 */
    }
    s.questions = Math.max(1, Math.min(s.questions, POOL.length || 1));
    return s;
  }

  function saveSettings(s) {
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(s));
    } catch (e) {
      /* 쓰기 실패는 무시 */
    }
  }

  // ------------------------------------------------------------ 대문
  function makePicker(container, opts) {
    // opts: presets, unit, value, validate(v) → 숫자|null, hint
    let mode = opts.presets.indexOf(opts.value) >= 0 ? "preset" : "custom";
    let preset = mode === "preset" ? opts.value : null;
    const buttons = opts.presets.map((v) =>
      el("button", { type: "button", class: "chip", "aria-pressed": "false", text: v + opts.unit, onclick: () => select(v) })
    );
    const customBtn = el("button", { type: "button", class: "chip", "aria-pressed": "false", text: "직접 입력", onclick: () => select(null) });
    const input = el("input", {
      type: "number",
      class: "custom-input",
      inputmode: "numeric",
      min: opts.min,
      max: opts.max,
      step: 1,
      "aria-label": opts.label + " 직접 입력",
      value: mode === "custom" ? opts.value : ""
    });
    const customWrap = el("span", { class: "custom-wrap" }, [input, el("span", { class: "unit", text: opts.unit })]);
    container.append(...buttons, customBtn, customWrap);
    input.addEventListener("input", opts.onChange);

    function select(v) {
      if (v == null) {
        mode = "custom";
        if (!input.value) input.value = preset != null ? preset : opts.value;
      } else {
        mode = "preset";
        preset = v;
      }
      sync();
      if (mode === "custom") input.focus();
      opts.onChange();
    }

    function sync() {
      buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(mode === "preset" && opts.presets[i] === preset)));
      customBtn.setAttribute("aria-pressed", String(mode === "custom"));
      customWrap.hidden = mode !== "custom";
    }

    sync();
    return {
      value: () => (mode === "preset" ? preset : opts.validate(input.value === "" ? NaN : input.value)),
      setInvalid: (bad) => input.classList.toggle("invalid", bad)
    };
  }

  let pickers = null;

  function initHome() {
    const n = POOL.length;
    $("pool-count").textContent = `포켓몬 ${n.toLocaleString("ko-KR")}마리 중 무작위 출제`;
    $("rule-text").textContent =
      `실제 몸무게와의 차이가 ±${Math.round(C.TOLERANCE_RATIO * 100)}% 이내면 정답이에요. ` +
      `(가벼운 포켓몬은 최소 ±${C.TOLERANCE_MIN_KG}kg까지 인정)`;
    const maxQ = Math.min(C.QUESTION_MAX, n);
    $("hint-questions").textContent = `직접 입력: ${C.QUESTION_MIN}~${maxQ}문제`;
    $("hint-seconds").textContent = `직접 입력: ${C.TIME_MIN}~${C.TIME_MAX}초`;

    if (!n) {
      $("setup-error").textContent =
        "포켓몬 데이터(data/pokemon-data.js)를 불러오지 못했어요. scripts/fetch_pokemon.py 를 실행해 데이터를 만들어 주세요.";
      $("start-btn").disabled = true;
      return;
    }

    const saved = loadSettings();
    pickers = {
      questions: makePicker($("picker-questions"), {
        presets: C.QUESTION_PRESETS.filter((v) => v <= n),
        unit: "문제",
        label: "문제 수",
        min: C.QUESTION_MIN,
        max: maxQ,
        value: saved.questions,
        validate: (v) => R.validQuestionCount(v, n),
        onChange: validateSetup
      }),
      seconds: makePicker($("picker-seconds"), {
        presets: C.TIME_PRESETS,
        unit: "초",
        label: "문제당 시간",
        min: C.TIME_MIN,
        max: C.TIME_MAX,
        value: saved.seconds,
        validate: R.validSeconds,
        onChange: validateSetup
      })
    };
    validateSetup();

    $("setup-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const s = validateSetup();
      if (s) startQuiz(s);
    });
  }

  function validateSetup() {
    const q = pickers.questions.value();
    const t = pickers.seconds.value();
    pickers.questions.setInvalid(!q);
    pickers.seconds.setInvalid(!t);
    const msgs = [];
    if (!q) msgs.push(`문제 수는 ${C.QUESTION_MIN}~${Math.min(C.QUESTION_MAX, POOL.length)} 사이 정수로 입력해 주세요.`);
    if (!t) msgs.push(`문제당 시간은 ${C.TIME_MIN}~${C.TIME_MAX}초 사이 정수로 입력해 주세요.`);
    $("setup-error").textContent = msgs.join(" ");
    $("start-btn").disabled = msgs.length > 0;
    return msgs.length ? null : { questions: q, seconds: t };
  }

  // ------------------------------------------------------------ 타이머 (Date.now() 차이로 계산)
  const timer = {
    id: null,
    running: false,
    startedAt: 0,
    elapsed: 0,
    duration: 0,
    onExpire: null,
    begin(durationMs, onExpire) {
      this.stop();
      this.duration = durationMs;
      this.elapsed = 0;
      this.startedAt = Date.now();
      this.running = true;
      this.onExpire = onExpire;
      this.id = setInterval(() => this.tick(), 100);
      this.tick();
    },
    elapsedMs() {
      return this.elapsed + (this.running ? Date.now() - this.startedAt : 0);
    },
    remainingMs() {
      return Math.max(0, this.duration - this.elapsedMs());
    },
    pause() {
      if (!this.running) return;
      this.elapsed += Date.now() - this.startedAt;
      this.running = false;
    },
    resume() {
      if (this.running || this.id == null) return;
      this.startedAt = Date.now();
      this.running = true;
      this.tick();
    },
    stop() {
      this.pause();
      if (this.id != null) clearInterval(this.id);
      this.id = null;
    },
    tick() {
      if (!this.running) return;
      const rem = this.remainingMs();
      renderTimer(rem, this.duration);
      if (rem <= 0) {
        this.stop();
        if (this.onExpire) this.onExpire();
      }
    }
  };

  function renderTimer(remMs, durMs) {
    const bar = $("timer-bar");
    bar.style.width = (durMs ? (remMs / durMs) * 100 : 100) + "%";
    const secs = Math.ceil(remMs / 1000);
    $("timer-text").textContent = `${secs}초`;
    $("timer").classList.toggle("danger", remMs <= C.WARN_SECONDS * 1000);
  }

  function renderTimerWaiting(seconds) {
    $("timer-bar").style.width = "100%";
    $("timer-text").textContent = `${seconds}초 · 이미지 불러오는 중`;
    $("timer").classList.remove("danger");
  }

  document.addEventListener("visibilitychange", () => timer.tick());

  // ------------------------------------------------------------ 퀴즈
  let quiz = null;
  let loadToken = 0;
  const preloaded = new Map();

  function startQuiz(settings) {
    saveSettings(settings);
    quiz = {
      settings,
      questions: R.pickQuestions(POOL, settings.questions),
      index: 0,
      results: [],
      phase: "loading"
    };
    preloaded.clear();
    showScreen("quiz");
    showQuestion();
  }

  function current() {
    return quiz.questions[quiz.index];
  }

  function renderScoreboard() {
    const total = quiz.questions.length;
    const done = quiz.results.length;
    const correct = quiz.results.filter((r) => r.correct).length;
    $("sb-question").textContent = `문제 ${Math.min(quiz.index + 1, total)} / ${total}`;
    $("sb-correct").textContent = `맞힌 문제 ${correct} / ${done}`;
    const dots = $("sb-dots");
    dots.textContent = "";
    for (let i = 0; i < total; i++) {
      const r = quiz.results[i];
      let cls = "dot";
      let label = `${i + 1}번: 남음`;
      if (r) {
        cls += r.correct ? " ok" : " ng";
        label = `${i + 1}번: ${r.correct ? "정답" : r.timedOut ? "시간 초과" : "오답"}`;
      } else if (i === quiz.index) {
        cls += " now";
        label = `${i + 1}번: 지금 문제`;
      }
      dots.appendChild(el("li", { class: cls, title: label, "aria-label": label }));
    }
    document.documentElement.style.setProperty("--sb-h", $("scoreboard").offsetHeight + "px");
  }

  function preload(p) {
    if (!p || preloaded.has(p.id)) return;
    const img = new Image();
    img.onerror = () => {
      if (p.imageFallback && img.src !== p.imageFallback) img.src = p.imageFallback;
    };
    img.src = p.image || p.imageFallback || "";
    preloaded.set(p.id, img);
  }

  /** 이미지를 넣고, 뜨거나 최종 실패하거나 최대 3초가 지나면 resolve. */
  function loadArt(p) {
    const wrap = $("art-wrap");
    wrap.textContent = "";
    wrap.classList.add("loading");
    const img = el("img", { class: "art", alt: `${p.nameKo || p.nameEn} 공식 일러스트`, decoding: "async", hidden: true });
    const placeholder = el("div", { class: "art-placeholder", hidden: true }, [
      el("span", { class: "ball big", "aria-hidden": "true" }),
      el("span", { text: "이미지를 불러오지 못했어요" })
    ]);
    wrap.append(img, placeholder);
    const sources = [p.image, p.imageFallback].filter(Boolean);
    let i = 0;
    return new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(t);
        resolve();
      };
      const t = setTimeout(finish, C.IMAGE_WAIT_MS);
      const fail = () => {
        wrap.classList.remove("loading");
        img.hidden = true;
        placeholder.hidden = false;
        finish();
      };
      img.onload = () => {
        wrap.classList.remove("loading");
        img.hidden = false;
        if (i > 0) img.classList.add("pixel");
        finish();
      };
      img.onerror = () => {
        i++;
        if (i < sources.length) img.src = sources[i];
        else fail();
      };
      if (sources.length) img.src = sources[0];
      else fail();
    });
  }

  function renderHead(p) {
    $("dex-no").textContent = R.dexNo(p.id);
    $("dex-name-ko").textContent = p.nameKo || p.nameEn || "";
    $("dex-name-en").textContent = p.nameEn && p.nameKo ? p.nameEn : "";
    $("dex-genus").textContent = p.genusKo || "";
    $("dex-genus").hidden = !p.genusKo;
    const types = $("dex-types");
    types.textContent = "";
    p.types.forEach((t) => types.appendChild(typeBadge(t)));
    $("fact-height").hidden = p.heightM == null;
    $("dex-height").textContent = p.heightM != null ? `${p.heightM.toFixed(1)} m` : "";
    const w = $("dex-weight");
    w.textContent = "??? kg";
    w.classList.remove("revealed");
  }

  function detailRow(label, content, extraClass) {
    return el("div", { class: "detail-row" + (extraClass ? " " + extraClass : "") }, [
      el("div", { class: "detail-label", text: label }),
      el("div", { class: "detail-value" }, content)
    ]);
  }

  function renderDetails(p) {
    const box = $("dex-details");
    box.textContent = "";
    const rows = [];

    const badges = [];
    if (p.isLegendary) badges.push(el("span", { class: "flag flag-legend", text: "전설" }));
    if (p.isMythical) badges.push(el("span", { class: "flag flag-myth", text: "환상" }));
    if (p.isBaby) badges.push(el("span", { class: "flag flag-baby", text: "베이비" }));
    if (badges.length) rows.push(el("div", { class: "flags" }, badges));

    if (p.flavorText) rows.push(el("blockquote", { class: "flavor", text: p.flavorText }));

    if (p.abilities && p.abilities.length) {
      rows.push(
        detailRow(
          "특성",
          el(
            "div",
            { class: "abilities" },
            p.abilities.map((a) =>
              el("span", { class: "ability" + (a.hidden ? " hidden-ability" : "") }, [
                a.nameKo || a.nameEn,
                a.hidden ? el("em", { text: "숨은 특성" }) : null
              ])
            )
          )
        )
      );
    }
    if (p.generation != null) rows.push(detailRow("세대", `${p.generation}세대`));
    if (p.evolvesFromKo) rows.push(detailRow("진화 전", p.evolvesFromKo));
    if (p.color) {
      const c = R.COLORS[p.color] || { ko: p.color, hex: "#ccc" };
      rows.push(detailRow("색", [el("span", { class: "swatch", style: `background:${c.hex}` }), c.ko]));
    }
    if (p.habitat) rows.push(detailRow("서식지", R.HABITATS[p.habitat] || p.habitat));

    if (p.stats) {
      let sum = 0;
      const bars = R.STATS.map((s) => {
        const v = p.stats[s.key];
        if (typeof v !== "number") return null;
        sum += v;
        const w = Math.min(100, (v / C.STAT_MAX) * 100);
        return el("div", { class: "stat" }, [
          el("span", { class: "stat-name", text: s.ko }),
          el("span", { class: "stat-num", text: v }),
          el("span", { class: "stat-track" }, el("span", { class: "stat-bar", style: `width:${w}%;background:${s.color}` }))
        ]);
      });
      rows.push(
        el("div", { class: "stats-block" }, [
          el("div", { class: "detail-label", text: "기본 능력치" }),
          el("div", { class: "stats" }, bars),
          el("div", { class: "stat-total" }, ["합계 ", el("strong", { text: sum })])
        ])
      );
    }
    box.append(...rows);
    box.hidden = rows.length === 0;
  }

  function resetAnswer() {
    const input = $("answer-input");
    input.value = "";
    input.disabled = false;
    input.classList.remove("invalid");
    $("answer-form").hidden = false;
    $("feedback").hidden = true;
    $("feedback").textContent = "";
    updateSubmitState();
    input.focus();
  }

  function updateSubmitState() {
    const input = $("answer-input");
    const v = R.parseAnswer(input.value);
    const typed = input.value.trim() !== "";
    input.classList.toggle("invalid", typed && v == null);
    $("submit-btn").disabled = !quiz || quiz.phase !== "answering" || v == null;
  }

  function showQuestion() {
    const p = current();
    quiz.phase = "loading";
    renderScoreboard();
    renderHead(p);
    renderDetails(p);
    renderTimerWaiting(quiz.settings.seconds);
    resetAnswer();
    const token = ++loadToken;
    loadArt(p).then(() => {
      if (token !== loadToken || !quiz || quiz.phase !== "loading") return;
      quiz.phase = "answering";
      updateSubmitState();
      timer.begin(quiz.settings.seconds * 1000, () => finishQuestion(R.parseAnswer($("answer-input").value), true));
    });
    preload(quiz.questions[quiz.index + 1]);
  }

  function finishQuestion(guess, timedOut) {
    if (!quiz || quiz.phase !== "answering") return;
    timer.stop();
    const dur = quiz.settings.seconds * 1000;
    const timeMs = timedOut ? dur : Math.min(timer.elapsedMs(), dur);
    const r = R.grade(current(), guess, timedOut, timeMs);
    quiz.results.push(r);
    quiz.phase = "feedback";
    if (timedOut) renderTimer(0, dur);
    $("answer-input").disabled = true;
    $("submit-btn").disabled = true;
    renderScoreboard();
    renderFeedback(r);
  }

  function renderFeedback(r) {
    const p = r.pokemon;
    const w = $("dex-weight");
    w.textContent = R.formatKg(p.weightKg);
    w.classList.add("revealed");

    const status = r.correct ? "정답!" : r.timedOut ? "시간 초과" : "오답";
    const cls = r.correct ? "ok" : r.timedOut ? "timeout" : "ng";
    const tol = R.tolerance(p.weightKg);
    const isLast = quiz.index + 1 >= quiz.questions.length;
    const nextBtn = el("button", {
      type: "button",
      class: "btn btn-primary btn-lg next-btn",
      text: isLast ? "결과 보기" : "다음 문제",
      onclick: next
    });
    const fb = $("feedback");
    fb.textContent = "";
    fb.className = "feedback " + cls;
    fb.append(
      el("div", { class: "fb-status", text: status }),
      el("dl", { class: "fb-rows" }, [
        el("div", null, [el("dt", { text: "실제 몸무게" }), el("dd", { class: "fb-actual", text: R.formatKg(p.weightKg) })]),
        el("div", null, [el("dt", { text: "내 답" }), el("dd", { text: r.guess != null ? R.formatKg(r.guess) : "입력 없음" })]),
        el("div", null, [
          el("dt", { text: "오차율" }),
          el("dd", { text: r.timedOut ? "— (시간 초과는 오답 처리)" : R.describeError(r.errorPct) })
        ])
      ]),
      el("p", {
        class: "fb-range",
        text: `정답 범위: ${R.formatKg(Math.max(0, p.weightKg - tol))} ~ ${R.formatKg(p.weightKg + tol)}`
      }),
      nextBtn
    );
    fb.hidden = false;
    $("answer-form").hidden = true;
    nextBtn.focus({ preventScroll: true });
    if (window.matchMedia("(max-width: 860px)").matches) fb.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function next() {
    if (!quiz || quiz.phase !== "feedback") return;
    if (quiz.index + 1 >= quiz.questions.length) {
      showResults();
      return;
    }
    quiz.index++;
    showQuestion();
    window.scrollTo(0, 0);
  }

  $("answer-input").addEventListener("input", updateSubmitState);
  $("answer-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!quiz || quiz.phase !== "answering") return;
    const v = R.parseAnswer($("answer-input").value);
    if (v != null) finishQuestion(v, false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.repeat || e.isComposing) return;
    if (quiz && quiz.phase === "feedback" && !(e.target instanceof HTMLButtonElement)) {
      e.preventDefault();
      next();
    }
  });
  $("quit-btn").addEventListener("click", () => {
    if (!quiz || quiz.phase === "done") return;
    timer.pause();
    const ok = window.confirm("퀴즈를 그만두고 지금까지 푼 문제로 결과를 볼까요?");
    if (!ok) {
      timer.resume();
      return;
    }
    showResults();
  });

  // ------------------------------------------------------------ 결과
  function thumb(p, small) {
    const srcs = small ? [p.imageFallback, p.image] : [p.image, p.imageFallback];
    const list = srcs.filter(Boolean);
    const img = el("img", { class: small ? "thumb" : "pick-art", alt: p.nameKo || p.nameEn, loading: "lazy" });
    let i = 0;
    img.onerror = () => {
      i++;
      if (i < list.length) img.src = list[i];
      else img.style.visibility = "hidden";
    };
    if (list.length) img.src = list[0];
    return img;
  }

  function tile(label, value, sub) {
    return el("div", { class: "tile" }, [
      el("div", { class: "tile-label", text: label }),
      el("div", { class: "tile-value", text: value }),
      sub ? el("div", { class: "tile-sub", text: sub }) : null
    ]);
  }

  function pickCard(title, r, cls) {
    const p = r.pokemon;
    return el("div", { class: "card pick " + cls }, [
      el("div", { class: "pick-title", text: title }),
      el("div", { class: "pick-body" }, [
        el("div", { class: "pick-art-wrap" }, thumb(p, false)),
        el("div", { class: "pick-info" }, [
          el("div", { class: "pick-name" }, [p.nameKo || p.nameEn, " ", el("small", { text: R.dexNo(p.id) })]),
          el("div", { text: `실제 ${R.formatKg(p.weightKg)}` }),
          el("div", { text: `내 답 ${R.formatKg(r.guess)}` }),
          el("div", { class: "pick-err", text: R.describeError(r.errorPct) })
        ])
      ])
    ]);
  }

  function rateBar(label, c, t, labelNode) {
    const rate = t ? c / t : 0;
    return el("div", { class: "rate" }, [
      el("div", { class: "rate-label" }, labelNode || label),
      el("div", { class: "rate-track" }, el("div", { class: "rate-bar", style: `width:${rate * 100}%` })),
      el("div", { class: "rate-num", text: t ? `${pct(rate)} (${c}/${t})` : "출제 없음" })
    ]);
  }

  function scatter(graded) {
    const W = 360, H = 320, L = 52, Rm = 14, T = 14, B = 44;
    const vals = [];
    graded.forEach((r) => vals.push(r.guess, r.pokemon.weightKg));
    let lo = Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, vals))));
    let hi = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, vals))));
    if (hi <= lo) hi = lo * 10;
    const llo = Math.log10(lo), lhi = Math.log10(hi);
    const x = (v) => L + ((Math.log10(v) - llo) / (lhi - llo)) * (W - L - Rm);
    const y = (v) => H - B - ((Math.log10(v) - llo) / (lhi - llo)) * (H - T - B);
    const kids = [];
    for (let e = Math.round(llo); e <= Math.round(lhi); e++) {
      const v = Math.pow(10, e);
      const label = v >= 1 ? v.toLocaleString("ko-KR") : String(Number(v.toPrecision(1)));
      kids.push(
        svg("line", { x1: x(v), x2: x(v), y1: T, y2: H - B, class: "grid" }),
        svg("line", { x1: L, x2: W - Rm, y1: y(v), y2: y(v), class: "grid" }),
        svg("text", { x: x(v), y: H - B + 16, "text-anchor": "middle", class: "tick" }, [label]),
        svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", class: "tick" }, [label])
      );
    }
    kids.push(
      svg("line", { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), class: "diag" }),
      svg("text", { x: x(hi) - 4, y: y(hi) + 14, "text-anchor": "end", class: "diag-label" }, ["추측 = 실제"]),
      svg("text", { x: (L + W - Rm) / 2, y: H - 6, "text-anchor": "middle", class: "axis" }, ["실제 몸무게 (kg, 로그)"]),
      svg("text", { x: 14, y: (T + H - B) / 2, "text-anchor": "middle", class: "axis", transform: `rotate(-90 14 ${(T + H - B) / 2})` }, [
        "내 추측 (kg, 로그)"
      ])
    );
    graded.forEach((r) => {
      kids.push(
        svg("circle", { cx: x(r.pokemon.weightKg), cy: y(r.guess), r: 6, class: r.correct ? "pt ok" : "pt ng" }, [
          svg("title", {}, [`${r.pokemon.nameKo}: 실제 ${R.formatKg(r.pokemon.weightKg)}, 추측 ${R.formatKg(r.guess)}`])
        ])
      );
    });
    return svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "scatter", role: "img", "aria-label": "추측 대 실제 몸무게 산점도" }, kids);
  }

  function resultTable(results) {
    const head = el("tr", null, ["번호", "", "이름", "실제", "내 답", "오차율", "걸린 시간", "결과"].map((h) => el("th", { text: h })));
    const rows = results.map((r, i) => {
      const p = r.pokemon;
      const res = r.correct ? "정답" : r.timedOut ? "시간 초과" : "오답";
      return el("tr", { class: r.correct ? "ok" : r.timedOut ? "timeout" : "ng" }, [
        el("td", { class: "c-no", text: i + 1 }),
        el("td", { class: "c-thumb" }, thumb(p, true)),
        el("td", { class: "c-name" }, [p.nameKo || p.nameEn, el("small", { text: R.dexNo(p.id) })]),
        el("td", { class: "c-actual", "data-label": "실제", text: R.formatKg(p.weightKg) }),
        el("td", { class: "c-guess", "data-label": "내 답", text: r.guess != null ? R.formatKg(r.guess) : "—" }),
        el("td", { class: "c-err", "data-label": "오차율", text: R.formatSignedPct(r.errorPct) }),
        el("td", { class: "c-time", "data-label": "시간", text: R.formatSeconds(r.timeMs) }),
        el("td", { class: "c-res" }, el("span", { class: "res-badge", text: res }))
      ]);
    });
    return el("div", { class: "table-wrap" }, el("table", { class: "result-table" }, [el("thead", null, head), el("tbody", null, rows)]));
  }

  function section(title, body) {
    return el("section", { class: "card result-section" }, [el("h3", { text: title }), body]);
  }

  function showResults() {
    if (!quiz) return;
    timer.stop();
    loadToken++;
    quiz.phase = "done";
    const results = quiz.results;
    const root = $("result-root");
    root.textContent = "";
    showScreen("result");

    if (!results.length) {
      root.append(
        el("div", { class: "card result-hero" }, [
          el("h2", { text: "결과" }),
          el("p", { class: "empty", text: "푼 문제가 없어요. 다시 도전해 보세요!" })
        ])
      );
      return;
    }
    const s = R.computeStats(results);
    const quitEarly = results.length < quiz.questions.length;

    root.append(
      el("div", { class: "card result-hero" }, [
        el("h2", { text: "결과" }),
        el("div", { class: "big-score" }, [el("strong", { text: s.correct }), ` / ${s.total} 맞힘`]),
        el("div", { class: "big-rate", text: `정답률 ${pct(s.accuracy)}` }),
        quitEarly ? el("p", { class: "muted", text: `${quiz.questions.length}문제 중 ${results.length}문제까지 풀고 그만뒀어요.` }) : null,
        s.biasRatio != null ? el("p", { class: "bias", text: R.describeBias(s.biasRatio) }) : null
      ]),
      el("div", { class: "tiles" }, [
        tile("오차율 중앙값", s.medianAbsErrorPct != null ? `±${Math.round(s.medianAbsErrorPct)}%` : "—", "시간 초과 제외"),
        tile("시간 초과", `${s.timeouts}회`),
        tile("평균 응답 시간", s.avgTimeMs != null ? R.formatSeconds(s.avgTimeMs) : "—", "시간 초과 제외")
      ])
    );

    if (s.closest) {
      const picks = [pickCard("가장 가까웠던 문제", s.closest, "best")];
      if (s.farthest && s.farthest !== s.closest) picks.push(pickCard("가장 빗나간 문제", s.farthest, "worst"));
      root.append(el("div", { class: "picks" }, picks));
    }

    root.append(
      el("div", { class: "two-col" }, [
        section("체급별 정답률", el("div", { class: "rates" }, s.classes.map((c) => rateBar(c.label, c.correct, c.total)))),
        section(
          "타입별 정답률",
          el("div", { class: "rates" }, s.types.map((t) => rateBar(null, t.correct, t.total, typeBadge(t.type))))
        )
      ])
    );

    if (s.graded.length) {
      root.append(
        section(
          "추측 vs 실제",
          el("div", { class: "scatter-wrap" }, [
            scatter(s.graded),
            el("div", { class: "legend" }, [
              el("span", { class: "lg ok", text: "정답" }),
              el("span", { class: "lg ng", text: "오답" }),
              el("span", { class: "lg diag", text: "기준선: 추측 = 실제 (선 위쪽은 무겁게, 아래쪽은 가볍게 추측)" })
            ])
          ])
        )
      );
    }

    root.append(section("문제별 결과", resultTable(results)));
    $("retry-btn").focus({ preventScroll: true });
  }

  $("retry-btn").addEventListener("click", () => {
    if (quiz) startQuiz(quiz.settings);
  });
  $("home-btn").addEventListener("click", () => {
    quiz = null;
    timer.stop();
    showScreen("home");
    validateSetup();
  });

  initHome();
  showScreen("home");
})();
