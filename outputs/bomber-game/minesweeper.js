"use strict";

/* =========================================================
 * 게임 로직 (DOM 에 의존하지 않음)
 * ========================================================= */

const DIFFICULTIES = {
  easy: { rows: 9, cols: 9, mines: 10 },
  normal: { rows: 16, cols: 16, mines: 40 },
  hard: { rows: 16, cols: 30, mines: 99 },
};

function createBoard(rows, cols) {
  const cells = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      row.push({ mine: false, adjacent: 0, open: false, flagged: false });
    }
    cells.push(row);
  }
  return { rows, cols, cells };
}

function forEachNeighbor(board, row, col, fn) {
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const r = row + dr;
      const c = col + dc;
      if (r >= 0 && r < board.rows && c >= 0 && c < board.cols) {
        fn(r, c);
      }
    }
  }
}

/** 지뢰를 배치한다. (safeRow, safeCol) 칸에는 절대 놓지 않는다. */
function placeMines(board, mineCount, safeRow, safeCol, random = Math.random) {
  const candidates = [];
  for (let r = 0; r < board.rows; r++) {
    for (let c = 0; c < board.cols; c++) {
      if (r !== safeRow || c !== safeCol) candidates.push([r, c]);
    }
  }
  // Fisher-Yates 부분 셔플로 mineCount 개를 무작위로 고른다.
  for (let i = 0; i < mineCount; i++) {
    const j = i + Math.floor(random() * (candidates.length - i));
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    const [r, c] = candidates[i];
    board.cells[r][c].mine = true;
  }
  calculateAdjacentCounts(board);
}

/** 모든 칸의 주변 8칸 지뢰 수를 계산한다. */
function calculateAdjacentCounts(board) {
  for (let r = 0; r < board.rows; r++) {
    for (let c = 0; c < board.cols; c++) {
      let count = 0;
      forEachNeighbor(board, r, c, (nr, nc) => {
        if (board.cells[nr][nc].mine) count++;
      });
      board.cells[r][c].adjacent = count;
    }
  }
}

/**
 * 칸을 연다. 주변 지뢰가 0 인 칸이면 이어진 빈 칸을 펼친다.
 * 반환값: { hitMine: 지뢰를 밟았는지, opened: 새로 열린 칸 [r, c] 목록 }
 */
function revealCell(board, row, col) {
  const start = board.cells[row][col];
  if (start.open || start.flagged) return { hitMine: false, opened: [] };

  if (start.mine) {
    start.open = true;
    return { hitMine: true, opened: [[row, col]] };
  }

  const opened = [];
  const stack = [[row, col]];
  while (stack.length > 0) {
    const [r, c] = stack.pop();
    const cell = board.cells[r][c];
    if (cell.open || cell.flagged || cell.mine) continue;
    cell.open = true;
    opened.push([r, c]);
    if (cell.adjacent === 0) {
      forEachNeighbor(board, r, c, (nr, nc) => stack.push([nr, nc]));
    }
  }
  return { hitMine: false, opened };
}

/** 깃발을 꽂거나 뺀다. 이미 열린 칸이면 false 를 반환한다. */
function toggleFlag(board, row, col) {
  const cell = board.cells[row][col];
  if (cell.open) return false;
  cell.flagged = !cell.flagged;
  return true;
}

function countFlags(board) {
  let count = 0;
  for (const row of board.cells) {
    for (const cell of row) if (cell.flagged) count++;
  }
  return count;
}

/** 지뢰가 아닌 칸이 모두 열렸으면 승리. */
function isWin(board) {
  for (const row of board.cells) {
    for (const cell of row) {
      if (!cell.mine && !cell.open) return false;
    }
  }
  return true;
}

/* =========================================================
 * 화면 (UI)
 * ========================================================= */

document.addEventListener("DOMContentLoaded", () => {
  const boardEl = document.getElementById("board");
  const difficultyEl = document.getElementById("difficulty");
  const minesLeftEl = document.getElementById("mines-left");
  const timerEl = document.getElementById("timer");
  const newGameBtn = document.getElementById("new-game");
  const messageEl = document.getElementById("message");

  const state = {
    config: null,
    board: null,
    minesPlaced: false,
    over: false,
    exploded: null,
    timerId: null,
    startTime: 0,
    elapsed: 0,
  };
  let cellEls = [];

  function newGame() {
    stopTimer();
    state.config = DIFFICULTIES[difficultyEl.value];
    const { rows, cols } = state.config;
    state.board = createBoard(rows, cols);
    state.minesPlaced = false;
    state.over = false;
    state.exploded = null;
    state.elapsed = 0;

    boardEl.style.setProperty("--cols", cols);
    boardEl.classList.remove("ended");
    boardEl.innerHTML = "";
    cellEls = [];
    for (let r = 0; r < rows; r++) {
      const rowEls = [];
      for (let c = 0; c < cols; c++) {
        const el = document.createElement("div");
        el.className = "cell";
        el.dataset.row = r;
        el.dataset.col = c;
        boardEl.appendChild(el);
        rowEls.push(el);
      }
      cellEls.push(rowEls);
    }

    setMessage("", "");
    render();
  }

  function startTimer() {
    state.startTime = Date.now();
    state.timerId = setInterval(() => {
      state.elapsed = Math.floor((Date.now() - state.startTime) / 1000);
      timerEl.textContent = state.elapsed;
    }, 1000);
  }

  function stopTimer() {
    if (state.timerId !== null) {
      clearInterval(state.timerId);
      state.timerId = null;
    }
  }

  function setMessage(text, type) {
    messageEl.textContent = text;
    messageEl.className = "message" + (type ? " " + type : "");
  }

  function endGame(won) {
    state.over = true;
    stopTimer();
    state.elapsed = Math.floor((Date.now() - state.startTime) / 1000);
    boardEl.classList.add("ended");
    if (won) {
      setMessage(`승리! ${state.elapsed}초 만에 모든 지뢰를 피했습니다.`, "win");
    } else {
      setMessage("게임 오버! 지뢰를 밟았습니다.", "lose");
    }
  }

  function handleOpen(row, col) {
    if (state.over) return;
    const cell = state.board.cells[row][col];
    if (cell.flagged || cell.open) return;

    if (!state.minesPlaced) {
      placeMines(state.board, state.config.mines, row, col);
      state.minesPlaced = true;
      startTimer();
    }

    const result = revealCell(state.board, row, col);
    if (result.hitMine) {
      state.exploded = { row, col };
      endGame(false);
    } else if (isWin(state.board)) {
      endGame(true);
    }
    render();
  }

  function handleFlag(row, col) {
    if (state.over) return;
    if (toggleFlag(state.board, row, col)) render();
  }

  function render() {
    const { board } = state;
    const lost = state.over && state.exploded !== null;

    for (let r = 0; r < board.rows; r++) {
      for (let c = 0; c < board.cols; c++) {
        const cell = board.cells[r][c];
        const el = cellEls[r][c];
        let text = "";
        let cls = "cell";

        if (cell.mine && (cell.open || lost)) {
          text = "💣";
          cls += " mine";
          if (state.exploded && state.exploded.row === r && state.exploded.col === c) {
            cls += " exploded";
          }
        } else if (cell.open) {
          cls += " open";
          if (cell.adjacent > 0) {
            text = cell.adjacent;
            cls += " n" + cell.adjacent;
          }
        } else if (cell.flagged) {
          text = "🚩";
        }

        el.className = cls;
        el.textContent = text;
      }
    }

    minesLeftEl.textContent = state.config.mines - countFlags(board);
    timerEl.textContent = state.elapsed;
  }

  function cellPosition(target) {
    const el = target.closest(".cell");
    if (!el) return null;
    return { row: Number(el.dataset.row), col: Number(el.dataset.col) };
  }

  boardEl.addEventListener("click", (e) => {
    const pos = cellPosition(e.target);
    if (pos) handleOpen(pos.row, pos.col);
  });

  boardEl.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    const pos = cellPosition(e.target);
    if (pos) handleFlag(pos.row, pos.col);
  });

  difficultyEl.addEventListener("change", newGame);
  newGameBtn.addEventListener("click", newGame);

  newGame();
});
