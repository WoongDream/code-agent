(function () {
  'use strict';

  // 앞뒤 공백, 띄어쓰기, 문장부호·기호, 영문 대소문자를 무시한다.
  function normalize(text) {
    return String(text == null ? '' : text)
      .normalize('NFC')
      .toLowerCase()
      .replace(/[\s\p{P}\p{S}]/gu, '');
  }

  function isCorrect(input, answers) {
    const key = normalize(input);
    if (!key) return false;
    return answers.some(function (a) { return normalize(a) === key; });
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  // 테스트에서 쓸 수 있도록 노출
  window.AzQuiz = { normalize: normalize, isCorrect: isCorrect };

  function init() {
    const $ = function (id) { return document.getElementById(id); };
    const el = {
      question: $('question'),
      category: $('category'),
      timer: document.querySelector('.timer'),
      timerFill: $('timerFill'),
      timerNum: $('timerNum'),
      form: $('answerForm'),
      input: $('answerInput'),
      submit: $('submitBtn'),
      feedback: $('feedback'),
      verdict: $('verdict'),
      answerLine: $('answerLine'),
      explainLine: $('explainLine'),
      next: $('nextBtn'),
      stats: $('stats'),
      reset: $('resetBtn'),
      timeNote: $('timeNote'),
      timeBtns: document.querySelectorAll('.time-btn')
    };
    if (!el.question) return;

    const quizzes = (typeof QUIZZES !== 'undefined' && Array.isArray(QUIZZES)) ? QUIZZES : [];
    if (quizzes.length === 0) {
      el.question.textContent = '문제 데이터(quizzes.js)를 불러오지 못했습니다.';
      return;
    }

    const state = {
      deck: [],
      pos: 0,
      lastIndex: -1,
      current: null,
      selectedLimit: 20,
      limit: 20,
      deadline: 0,
      timerId: null,
      answered: true,
      answeredAt: 0,
      solved: 0,
      correct: 0
    };

    function newDeck() {
      const deck = shuffle(quizzes.map(function (_, i) { return i; }));
      // 바퀴가 바뀔 때 방금 낸 문제가 바로 다시 나오지 않게 한다.
      if (deck.length > 1 && deck[0] === state.lastIndex) {
        const t = deck[0]; deck[0] = deck[1]; deck[1] = t;
      }
      state.deck = deck;
      state.pos = 0;
    }

    function updateStats() {
      const rate = state.solved === 0 ? 0 : Math.round(state.correct / state.solved * 100);
      el.stats.textContent = '푼 문제 ' + state.solved + '개 중 ' + state.correct + '개 정답 (정답률 ' + rate + '%)';
    }

    function renderTimer(remainingMs) {
      const ratio = Math.max(0, remainingMs) / (state.limit * 1000);
      const sec = Math.max(0, Math.ceil(remainingMs / 1000));
      el.timerFill.style.width = (ratio * 100) + '%';
      el.timerNum.textContent = sec;
      el.timer.classList.toggle('danger', sec <= 5);
    }

    function stopTimer() {
      if (state.timerId !== null) {
        clearInterval(state.timerId);
        state.timerId = null;
      }
    }

    function tick() {
      const remaining = state.deadline - Date.now();
      renderTimer(remaining);
      if (remaining <= 0) finish(false, true);
    }

    function nextQuestion() {
      stopTimer();
      if (state.pos >= state.deck.length) newDeck();
      const index = state.deck[state.pos++];
      state.lastIndex = index;
      state.current = quizzes[index];
      state.limit = state.selectedLimit;
      state.answered = false;

      el.category.textContent = state.current.category;
      el.question.textContent = state.current.question;
      el.feedback.hidden = true;
      el.feedback.classList.remove('ok', 'bad');
      el.timeNote.hidden = true;
      el.input.value = '';
      el.input.disabled = false;
      el.submit.disabled = false;
      el.input.focus();

      state.deadline = Date.now() + state.limit * 1000;
      renderTimer(state.limit * 1000);
      state.timerId = setInterval(tick, 100);
    }

    function finish(ok, timeUp) {
      if (state.answered) return;
      state.answered = true;
      state.answeredAt = Date.now();
      stopTimer();

      el.input.disabled = true;
      el.submit.disabled = true;

      state.solved++;
      if (ok) state.correct++;
      updateStats();

      const q = state.current;
      el.feedback.classList.add(ok ? 'ok' : 'bad');
      el.verdict.textContent = ok ? '정답!' : (timeUp ? '땡! 시간 초과' : '땡!');
      el.answerLine.textContent = '정답: ' + q.answers[0];
      el.explainLine.textContent = '풀이: ' + q.explain;
      el.feedback.hidden = false;
      el.next.focus();
    }

    function goNext() {
      // 답을 낸 Enter 가 곧바로 "다음 문제"로 이어지지 않게 잠깐 막는다.
      if (!state.answered || Date.now() - state.answeredAt < 300) return;
      nextQuestion();
    }

    el.form.addEventListener('submit', function (e) {
      e.preventDefault();
      if (state.answered) return;
      if (normalize(el.input.value) === '') {
        el.input.focus();
        return;
      }
      finish(isCorrect(el.input.value, state.current.answers), false);
    });

    el.next.addEventListener('click', goNext);

    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' || e.isComposing) return;
      if (state.answered && e.target !== el.next) {
        e.preventDefault();
        goNext();
      }
    });

    el.timeBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        state.selectedLimit = Number(btn.dataset.sec);
        el.timeBtns.forEach(function (b) { b.classList.toggle('active', b === btn); });
        el.timeNote.hidden = state.answered || state.selectedLimit === state.limit;
        if (!state.answered) el.input.focus();
      });
    });

    el.reset.addEventListener('click', function () {
      state.solved = 0;
      state.correct = 0;
      updateStats();
      newDeck();
      nextQuestion();
    });

    el.timeBtns.forEach(function (b) {
      b.classList.toggle('active', Number(b.dataset.sec) === state.selectedLimit);
    });
    updateStats();
    newDeck();
    nextQuestion();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
