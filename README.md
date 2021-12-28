# Knightr0's Gambit

## Overview
This repository hosts the code for Knightr0's Gambit, a [University of Central Florida](https://www.ucf.edu) [AI](https://ucfai.org) / [IEEE](https://ieee.cecs.ucf.edu) collaboration to create an automatic chessboard (similar to [Harry Potter's Wizard Chess](https://www.youtube.com/watch?v=s3cjWc-XXCg) with less violence) powered by a custom chess AI.

<!-- ![Image of chessboard](blah.jpg) -->

<!-- Video of chessboard in action: https://youtu.be/blah -->

[Project webpage](https://ucfai.github.io/knightros-gambit/), [Project overview doc](https://docs.google.com/document/d/1CPY9yEVWDVO99mbckBnCJi-gKKobkBnB4XnjrLc4HFs/edit#)

<!-- Build guide: n/a -->

## How it works
<!-- ![Overview diagram](docs/imgs/overview-diagram.jpg) -->
The main program is `game.py` (located in `firmware/raspi/`). This code is the entry point to interacting with the chessboard; it runs a loop that keeps track of game state (whose turn it is, past moves, etc.), computes moves using a custom chess AI, and processes images taken of the board to compute board state after the player moves a piece. This driver program has options for over-the-board play, a command-line interface, a voice interface, and a web interface.

The `game.py` program runs on a Raspberry Pi, and interacts with arduino code that controls the actuators of the chessboard, the four interfaces (one-at-a-time) described above, a custom AI modeled after Deepmind's AlphaZero, and a computer vision system which keeps track of the current chessboard state.

The arduino code (located in `firmware/arduino/chessboard`) runs on an ESP32 microcontroller. 
Once the AI move has been computed on the Raspberry Pi, it is converted to a standardized message format and sent to the ESP32 microcontroller using UART serial communication. The microcontroller then actuates the physical board mechanism (motors and electromagnet) and sends back status messages indicating current state of the board/microcontroller. UART serial communication is used as it allows us to easily send data in both directions.

The AI is modeled after [Deepmind's AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go).
> To learn each game, an untrained neural network plays millions of games against itself via a process of trial and error called reinforcement learning. At first, it plays completely randomly, but over time the system learns from wins, losses, and draws to adjust the parameters of the neural network, making it more likely to choose advantageous moves in the future.

> The trained network is used to guide a search algorithm – known as Monte-Carlo Tree Search (MCTS) – to select the most promising moves in games. For each move, AlphaZero searches only a small fraction of the positions considered by traditional chess engines. In Chess, for example, it searches only 60 thousand positions per second in chess, compared to roughly 60 million for Stockfish.

<!-- A system diagram of the Arduino is shown below. -->
<!-- ![Arduino diagram](imgs/arduino-diagram.jpg) -->


<!-- ## How to Build Chessboard
Main documentation: blah.com

You can find the bill of materials, assembly instructions, software setup, etc at this website.
 -->
 