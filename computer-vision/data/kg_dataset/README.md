# KG Dataset creation
## Procedure to get labeled data from game i (`game_i`)
- Play through `game_i` on lichess
- Save pgn from the game to the `game_i` subdirectory
- Create a `full_game` subdirectory (`game_i/full_game`)
- Use [`pgn-extract`](https://www.howtoinstall.me/ubuntu/18-04/pgn-extract/) command line tool to extract one fen for each PGN into the `full_game` subdirectory
	- Label these fens `j.fen` where `j` corresponds to the move number of `game_i` when this fen is the board state (Note: `j` starts at 0)
- Play through `game_i` on physical chessboard and take pictures after each move using script `get_images_of_game.py` (each image is labeled `j.png` as above and saved to the `full_game` subdirectory)
- Use `extract_labels_from_game.py` with input parameter `"game_i"` which
	1. Creates a subdirectory `game_i/j` for each move
	1. Uses fen string and segmented image of board state (where each sq on the board creates a single image) to pair each image with a label. Each cell yields a single image with path `game_i/j/<sq>` where sq is the cell on the chessboard for that piece (e.g. `e4`). Two labels are generated for each `sq`
		1. 0, 1, 2; these correspond to '.', 'w', 'b') and are used for developing square occupancy capability
		2. 0, 1, 2, 3, 4; these correspond to 'p', 'q', 'b', 'n', 'r' and are used for developing piece classification capability
