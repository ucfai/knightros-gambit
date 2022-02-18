There are two types of text files used for testing.

In both types of files, the first line(s) describes the type of test. Any line that begins with a % symbol is discarded and not used when running the test file.

1) In PGN files, the noncommented lines are the PGN moves of the game. These are parsed to create a list of `OpCode` type `Message`s

2) `OpCode` message files, where each noncommented line is a `Message`