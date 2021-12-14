// NOTE: this example uses the chess.js library:
// https://github.com/jhlywa/chess.js

let board = null;
const game = new Chess();
const $status = $('#status');
const $pgn = $('#pgn');
const $uci = $('#uci');

/** checks if game is over if valid player move
 * @param {Object} source location
 * @param {Object} piece current selection
 * @param {Object} position location
 * @param {Object} orientation location
 * @return {boolean} valid move
*/
function onDragStart(source, piece, position, orientation) {
  // do not pick up pieces if the game is over
  if (game.game_over()) return false;

  // only pick up pieces for the side to move
  if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
      (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
    return false;
  }
}

/** determines if move is promotion or valid and makes moves
 * @param {Object} source move from source location
 * @param {Object} target move to target location
 * @return {boolean} valid move
*/
function onDrop(source, target) {
  // see if the move is a promotion
  if (isPromotion({
    game: game,
    move: { from: source, to: target },
  })) {
    processUserInput(source, target);
  } else { // normal move
    var move = game.move({
      from: source,
      to: target,
    });
  }
  // illegal move
  if (move === null) return 'snapback';

  updateStatus();
}

/** callback function that promotes pawn to selected piece
 * @param {Object} piece that pawn will be promoted to
 * @param {Object} s move from source location
 * @param {Object} t move to target location
*/
function selection(piece, s, t) {
  game.move({ from: s, to: t, promotion: piece });
  updateStatus();
}

/** callback function that obtains value from popup and sends to selection
 * function
 * @param {Object} s move from source location
 * @param {Object} t move to target location
*/
async function processUserInput(s, t) {
  const piece = await askQuestion();
  selection(piece, s, t);
  board.position(game.fen());
}

/** returns promise after set amount of time
 * @param {Object} ms time in milliseconds
 * @return {Object} promise value
*/
function wait(ms = 0) {
  return new Promise((resolve) => (setTimeout(resolve, ms)));
};

/** destroys popup after selection is obtained
 * @param {Object} popup to be destroyed
*/
async function destroyPopup(popup) {
  let myPopup = popup;
  popup.classList.remove('open');
  await wait(1000);
  // REMOVE THE POPUP ENTIRELY
  popup.remove();
  myPopup = null;
};

/** prompts popup and returns value selected by user **/
async function ask() {
  return new Promise(async function (resolve) {
    // FIRST CREATE A POPUP WITH ALL THE FIELDS IN IT
    const popup = document.createElement('form');
    popup.classList.add('popup');
    if (game.turn() === 'w') {
      popup.insertAdjacentHTML(
        'afterbegin',
        `
        <fieldset>
          <button class = "button" value ="q" style="background-image: 
          url(img/chesspieces/wikipedia/wQ.png);"></button>
          <button class = "button" value ="r" style="background-image: 
          url(img/chesspieces/wikipedia/wR.png);"></button>
          <button class = "button" value ="b" style="background-image: 
          url(img/chesspieces/wikipedia/wB.png);"></button>
          <button class = "button" value ="n" style="background-image: 
          url(img/chesspieces/wikipedia/wN.png);"></button>
        </fieldset>
        `,
      )
    }
    else {
      popup.insertAdjacentHTML(
        'afterbegin',
        `
        <fieldset>
          <button class = "button" value ="q" style="background-image: 
          url(img/chesspieces/wikipedia/bQ.png);"></button>
          <button class = "button" value ="r" style="background-image: 
          url(img/chesspieces/wikipedia/bR.png);"></button>
          <button class = "button" value ="b" style="background-image: 
          url(img/chesspieces/wikipedia/bB.png);"></button>
          <button class = "button" value ="n" style="background-image: 
          url(img/chesspieces/wikipedia/bN.png);"></button>
        </fieldset>
        `,
      )
    };
    // LISTEN FOR THE SUBMIT EVENT ON THE INPUTS
    popup.addEventListener('click', function (event) {
      event.preventDefault();
      resolve(event.target.value);
      destroyPopup(popup);
    },
    { once: true });
    // INSERT THAT POPUP INTO THE DOM
    document.body.appendChild(popup);
    // PUT A SMALL TIMEOUT BEFORE WE ADD THE OPEN CLASS
    await wait(50);
    popup.classList.add('open');
  });
};


/** calls and returns value from ask function **/
async function askQuestion() {
  return await ask({ cancel: true });
};

/**
 * checks if the move will result in a pawn promotion
 * @param {Object} cfg game and move
 * @return {boolean} promotion or not?
*/
function isPromotion(cfg) {
  const piece = cfg.game.get(cfg.move.from);
  // Check if pawn is white or black and being promoted
  if ((cfg.game.turn() == 'w' &&
     cfg.move.from.charAt(1) == 7 &&
     cfg.move.to.charAt(1) == 8 &&
     piece.type == 'p' &&
     piece.color == 'w') ||
    (cfg.game.turn() == 'b' &&
     cfg.move.from.charAt(1) == 2 &&
     cfg.move.to.charAt(1) == 1 &&
     piece.type == 'p' &&
     piece.color == 'b')
  ) {
    const tempChess = new Chess(game.fen());
    if (tempChess.move({ from: cfg.move.from, to: cfg.move.to, promotion: 'q' }) ||
    tempChess.move({ from: cfg.move.from, to: cfg.move.to, promotion: 'r' }) ||
    tempChess.move({ from: cfg.move.from, to: cfg.move.to, promotion: 'n' }) ||
    tempChess.move({ from: cfg.move.from, to: cfg.move.to, promotion: 'b' })) {
      return true;
    } else {
      return false;
    }
  }
}

/** update the board position after the piece snap
* for castling, en passant, pawn promotion
**/
function onSnapEnd() {
  board.position(game.fen());
}

/** update status and prints out the player's move, pgn, and gan **/
function updateStatus() {
  let status = '';
  const moveColor = (game.turn() == 'b') ? 'Black' : 'White';
  // checkmate?
  if (game.in_checkmate()) {
    status = 'Game over, ' + moveColor + ' is in checkmate.';
  }
  // draw?
  else if (game.in_draw()) {
    status = 'Game over, drawn position';
  }
  // game still on
  else {
    status = moveColor + ' to move';

    // check?
    if (game.in_check()) {
      status += ', ' + moveColor + ' is in check';
    }
  }
  $status.html(status);
  $pgn.html(game.pgn());
  $uci.html(game.uci());
}

const config = {
  draggable: true,
  position: 'start',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd,
};
board = ChessBoard('myBoard', config);

updateStatus();
