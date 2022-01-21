# KG Dataset creation
## Procedure to get labeled data from game i (`game_i`)
- Play through `game_i` on lichess
- Save pgn from the game to the `game_i` subdirectory
- Create a `full_game` subdirectory (`game_i/full_game`)
- Use [`pgn-extract`](https://www.howtoinstall.me/ubuntu/18-04/pgn-extract/) command line tool to extract one fen for each PGN into the `full_game` subdirectory
	- Label these fens `game_i_fen_j` where `j` corresponds to the move number of `game_i` when this fen is the board state (Note: `j` starts at 0)
- Play through `game_i` on physical chessboard and take pictures after each move using script `get_images_of_game.py` (each image is labeled `game_i_img_j` as above)
- Use `extract_labels_from_game.py` which
	1. Pairs fen string with the corresponding image of that board state
	2. Creates a subdirectory for that position
- Pair each image of the board with the correct corresponding label
	1. 0, 1, 2; these correspond to '.', 'w', 'b')
	2. 0, 1, 2, 3, 4; these correspond to 'p', 'q', 'b', 'n', 'r'
