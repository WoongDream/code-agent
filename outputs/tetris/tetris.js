(function () {
  'use strict';

  const COLS = 10;
  const ROWS = 20;
  const BLOCK = 30;
  const NEXT_BLOCK = 24;

  const SHAPES = {
    I: [
      [0, 0, 0, 0],
      [1, 1, 1, 1],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ],
    J: [
      [1, 0, 0],
      [1, 1, 1],
      [0, 0, 0],
    ],
    L: [
      [0, 0, 1],
      [1, 1, 1],
      [0, 0, 0],
    ],
    O: [
      [1, 1],
      [1, 1],
    ],
    S: [
      [0, 1, 1],
      [1, 1, 0],
      [0, 0, 0],
    ],
    T: [
      [0, 1, 0],
      [1, 1, 1],
      [0, 0, 0],
    ],
    Z: [
      [1, 1, 0],
      [0, 1, 1],
      [0, 0, 0],
    ],
  };

  const COLORS = {
    I: '#00d8e8',
    J: '#3f6dff',
    L: '#ff9a2e',
    O: '#ffd83a',
    S: '#3ddc6a',
    T: '#b25cff',
    Z: '#ff4d5e',
  };

  // 회전 시 벽/블록에 막히면 순서대로 시도할 위치 보정값 [dx, dy]
  const KICKS = [[0, 0], [-1, 0], [1, 0], [-2, 0], [2, 0], [0, -1]];

  // 한 번에 지운 줄 수별 기본 점수 (레벨을 곱함)
  const LINE_SCORES = [0, 100, 300, 500, 800];

  const boardCtx = document.getElementById('board').getContext('2d');
  const nextCtx = document.getElementById('next').getContext('2d');
  const scoreEl = document.getElementById('score');
  const linesEl = document.getElementById('lines');
  const levelEl = document.getElementById('level');
  const overlayEl = document.getElementById('overlay');
  const finalScoreEl = document.getElementById('final-score');
  const restartBtn = document.getElementById('restart');

  let board;
  let current;
  let nextType;
  let bag;
  let score;
  let lines;
  let level;
  let dropCounter;
  let lastTime;
  let gameOver;

  function createBoard() {
    return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
  }

  // 7-bag: 7종류를 섞어 하나씩 꺼내 같은 블록이 몰리지 않게 한다
  function takeFromBag() {
    if (bag.length === 0) {
      bag = Object.keys(SHAPES);
      for (let i = bag.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [bag[i], bag[j]] = [bag[j], bag[i]];
      }
    }
    return bag.pop();
  }

  function createPiece(type) {
    const shape = SHAPES[type].map((row) => row.slice());
    return {
      type,
      shape,
      x: Math.floor((COLS - shape[0].length) / 2),
      y: 0,
    };
  }

  function collides(shape, x, y) {
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (!shape[r][c]) continue;
        const bx = x + c;
        const by = y + r;
        if (bx < 0 || bx >= COLS || by >= ROWS) return true;
        if (by >= 0 && board[by][bx]) return true;
      }
    }
    return false;
  }

  function rotateClockwise(shape) {
    const n = shape.length;
    return shape.map((row, r) => row.map((_, c) => shape[n - 1 - c][r]));
  }

  function rotate() {
    if (current.type === 'O') return;
    const rotated = rotateClockwise(current.shape);
    for (const [dx, dy] of KICKS) {
      if (!collides(rotated, current.x + dx, current.y + dy)) {
        current.shape = rotated;
        current.x += dx;
        current.y += dy;
        return;
      }
    }
  }

  function move(dx) {
    if (!collides(current.shape, current.x + dx, current.y)) {
      current.x += dx;
    }
  }

  // 한 칸 내린다. 더 내릴 수 없으면 고정하고 false 를 반환한다
  function stepDown() {
    if (!collides(current.shape, current.x, current.y + 1)) {
      current.y++;
      return true;
    }
    lockPiece();
    return false;
  }

  function softDrop() {
    if (stepDown()) score += 1;
    dropCounter = 0;
  }

  function hardDrop() {
    let distance = 0;
    while (!collides(current.shape, current.x, current.y + 1)) {
      current.y++;
      distance++;
    }
    score += distance * 2;
    lockPiece();
    dropCounter = 0;
  }

  function lockPiece() {
    const { shape, x, y, type } = current;
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c] && y + r < 0) {
          endGame();
          return;
        }
      }
    }
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c]) board[y + r][x + c] = type;
      }
    }
    clearLines();
    spawn();
  }

  function clearLines() {
    let cleared = 0;
    for (let r = ROWS - 1; r >= 0; r--) {
      if (board[r].every((cell) => cell !== null)) {
        board.splice(r, 1);
        board.unshift(Array(COLS).fill(null));
        cleared++;
        r++; // 위에서 내려온 줄을 같은 위치에서 다시 검사
      }
    }
    if (cleared > 0) {
      score += LINE_SCORES[cleared] * level;
      lines += cleared;
      level = Math.floor(lines / 10) + 1;
    }
  }

  function spawn() {
    current = createPiece(nextType);
    nextType = takeFromBag();
    if (collides(current.shape, current.x, current.y)) {
      endGame();
    }
  }

  function dropInterval() {
    return Math.max(100, 1000 - (level - 1) * 90);
  }

  function endGame() {
    gameOver = true;
    finalScoreEl.textContent = score;
    overlayEl.classList.remove('hidden');
  }

  function restart() {
    board = createBoard();
    bag = [];
    score = 0;
    lines = 0;
    level = 1;
    dropCounter = 0;
    lastTime = performance.now();
    gameOver = false;
    overlayEl.classList.add('hidden');
    nextType = takeFromBag();
    spawn();
  }

  function drawCell(ctx, x, y, size, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, size, size);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.fillRect(x, y, size, 3);
    ctx.fillRect(x, y, 3, size);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
  }

  function drawBoard() {
    boardCtx.fillStyle = '#0d0e16';
    boardCtx.fillRect(0, 0, COLS * BLOCK, ROWS * BLOCK);

    boardCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    boardCtx.lineWidth = 1;
    for (let c = 1; c < COLS; c++) {
      boardCtx.beginPath();
      boardCtx.moveTo(c * BLOCK + 0.5, 0);
      boardCtx.lineTo(c * BLOCK + 0.5, ROWS * BLOCK);
      boardCtx.stroke();
    }
    for (let r = 1; r < ROWS; r++) {
      boardCtx.beginPath();
      boardCtx.moveTo(0, r * BLOCK + 0.5);
      boardCtx.lineTo(COLS * BLOCK, r * BLOCK + 0.5);
      boardCtx.stroke();
    }

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (board[r][c]) {
          drawCell(boardCtx, c * BLOCK, r * BLOCK, BLOCK, COLORS[board[r][c]]);
        }
      }
    }

    const { shape, x, y, type } = current;
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c] && y + r >= 0) {
          drawCell(boardCtx, (x + c) * BLOCK, (y + r) * BLOCK, BLOCK, COLORS[type]);
        }
      }
    }
  }

  function drawNext() {
    const w = nextCtx.canvas.width;
    const h = nextCtx.canvas.height;
    nextCtx.fillStyle = '#0d0e16';
    nextCtx.fillRect(0, 0, w, h);

    const shape = SHAPES[nextType];
    let minR = Infinity, maxR = -1, minC = Infinity, maxC = -1;
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c]) {
          minR = Math.min(minR, r);
          maxR = Math.max(maxR, r);
          minC = Math.min(minC, c);
          maxC = Math.max(maxC, c);
        }
      }
    }
    const offsetX = (w - (maxC - minC + 1) * NEXT_BLOCK) / 2;
    const offsetY = (h - (maxR - minR + 1) * NEXT_BLOCK) / 2;
    for (let r = minR; r <= maxR; r++) {
      for (let c = minC; c <= maxC; c++) {
        if (shape[r][c]) {
          drawCell(
            nextCtx,
            offsetX + (c - minC) * NEXT_BLOCK,
            offsetY + (r - minR) * NEXT_BLOCK,
            NEXT_BLOCK,
            COLORS[nextType]
          );
        }
      }
    }
  }

  function draw() {
    drawBoard();
    drawNext();
    scoreEl.textContent = score;
    linesEl.textContent = lines;
    levelEl.textContent = level;
  }

  function update(time) {
    const dt = time - lastTime;
    lastTime = time;
    if (!gameOver) {
      dropCounter += dt;
      if (dropCounter >= dropInterval()) {
        stepDown();
        dropCounter = 0;
      }
    }
    draw();
    requestAnimationFrame(update);
  }

  document.addEventListener('keydown', (e) => {
    if (gameOver) {
      if (e.code === 'Enter' || e.code === 'NumpadEnter') {
        e.preventDefault();
        restart();
      }
      return;
    }
    switch (e.code) {
      case 'ArrowLeft':
        move(-1);
        break;
      case 'ArrowRight':
        move(1);
        break;
      case 'ArrowDown':
        softDrop();
        break;
      case 'ArrowUp':
        rotate();
        break;
      case 'Space':
        if (!e.repeat) hardDrop();
        break;
      default:
        return;
    }
    e.preventDefault();
  });

  restartBtn.addEventListener('click', () => {
    restartBtn.blur();
    restart();
  });

  restart();
  requestAnimationFrame(update);
})();
