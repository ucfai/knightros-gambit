// NOTE: this example uses the chess.js library:
// https://github.com/jhlywa/chess.js

var board = null
var game = new Chess()
game.load("rnbqkbnr/ppp1pppp/8/4PP2/8/P7/1PpP2PP/RNBQKBNR b KQkq - 0 5");
var $status = $('#status')
var $pgn = $('#pgn')
var $gan = $('#gan')

function onDragStart (source, piece, position, orientation) {
  // do not pick up pieces if the game is over
  if (game.game_over()) return false

  // only pick up pieces for the side to move
  if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
      (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
    return false
  }
}


function onDrop (source, target) {
  // see if the move is a promotion
  if (is_promotion({
    game: game,
    move: {from: source, to: target}
  }) ) {  
   
    processUserInput(source, target)
    //doesn't work b/c on drop there is no value for promotion so only changes when you drop again, must find a way to move the piece when there is no drop 
   /*choose_piece({
        onchosen: function(idPost) {
              game.move({ from: source, to: target, promotion: idPost });
         }
   }*/
   }
   else{ // normal move
    var move = game.move({
      from: source,
      to: target
    })
   }

  // illegal move
  if (move === null) return 'snapback'

  updateStatus()
}

//callback functions to select the piece
function selection(piece, s, t) {
  console.log("HERE");
  console.log("Source is: "+s);
  console.log("Target is: "+t);
  console.log("Piece is: "+piece);
  game.move({ from: s, to: t, promotion:  piece});
}

async function processUserInput(s, t) {
  let piece = await askQuestion();
  selection(piece, s, t);
  board.position(game.fen());
}

function wait (ms = 0) {
  return new Promise(resolve => setTimeout(resolve, ms));
};
async function destroyPopup(popup){
  let myPopup = popup;
  popup.classList.remove('open');
  await wait(1000);
  //REMOVE THE POPUP ENTIRELY
  popup.remove();
  myPopup = null;
};
async function ask(options){
  return new Promise( async function(resolve) {
  //FIRST CREATE A POPUP WITH ALL THE FIELDS IN IT
  const popup = document.createElement('form');
  popup.classList.add('popup');
  popup.insertAdjacentHTML(
      'afterbegin', 
      `
          <fieldset>
              <h2>Pawn Promotion Selection</h2>
          <select name="select" id="select">
            <option value="q">Queen</option>
            <option value="n">Knight</option>
            <option value="b">Bishop</option>
            <option value="r">Rook</option>
        </select>
        <br />
        <br />
        <button type="submit">Submit</button>
          </fieldset>
      `
      );
  //LISTEN FOR THE SUBMIT EVENT ON THE INPUTS
  popup.addEventListener('submit', function(event){
      event.preventDefault();
      console.log(event.target); // gives us the form
      console.log(event.target.select.value); // gives us the input value
      console.log('Submitted');
      resolve(event.target.select.value);
      destroyPopup(popup);
      },
      {once: true});
  //INSERT THAT POPUP INTO THE DOM
  document.body.appendChild(popup);
  // PUT A SMALL TIMEOUT BEFORE WE ADD THE OPEN CLASS
  await wait(50);
  popup.classList.add('open');
  });
};
//prompts the popup and returns the value selected by the user
 async function askQuestion () {
    return await ask({cancel: true});
  };

//determines if the move will result in a pawn promotion
function is_promotion(cfg) {
  var piece = cfg.game.get(cfg.move.from);
  if ( //pawn is white or black and being promoted 
         (cfg.game.turn() == 'w' &&
         cfg.move.from.charAt(1) == 7 &&
         cfg.move.to.charAt(1) == 8 &&
         piece.type == 'p' &&
         piece.color == 'w') 
         || 
         (cfg.game.turn() == 'b' &&
         cfg.move.from.charAt(1) == 2 &&
         cfg.move.to.charAt(1) == 1 &&
         piece.type == 'p' &&
         piece.color == 'b' )
  ) {
         var temp_chess= new Chess(game.fen());
         if ( temp_chess.move({from: cfg.move.from, to: cfg.move.to, promotion: 'q'}) || temp_chess.move({from: cfg.move.from, to: cfg.move.to, promotion: 'r'}) || temp_chess.move({from: cfg.move.from, to: cfg.move.to, promotion: 'n'}) || temp_chess.move({from: cfg.move.from, to: cfg.move.to, promotion: 'b'})  ) {
               return true;
         } else {
               return false;
         }
  }
}

// update the board position after the piece snap
// for castling, en passant, pawn promotion
function onSnapEnd () {
  console.log(game.fen())
  board.position(game.fen())
}

function updateStatus () {
  var status = ''

  var moveColor = (game.turn() == 'b') ? "Black" : "White";

  // checkmate?
  if (game.in_checkmate()) {
    status = 'Game over, ' + moveColor + ' is in checkmate.'
  }

  // draw?
  else if (game.in_draw()) {
    status = 'Game over, drawn position'
  }

  // game still on
  else {
    status = moveColor + ' to move'

    // check?
    if (game.in_check()) {
      status += ', ' + moveColor + ' is in check'
    }
  }
  
  $status.html(status)
  $pgn.html(game.pgn())
  $gan.html(game.gan())
}

var config = {
  draggable: true,
  position: 'start',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
}
board = ChessBoard('myBoard', config)
board.position(game.fen())

updateStatus()