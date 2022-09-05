# Running training loop
Use the bash script `train.sh` to invoke the main training program.

## Using Streamlit:

Example use:
```bash
# runs train using streamlit dashboard for hyperparameter initialization
./train.sh streamlit

```

## Using Params.json

`-m` : make a new dataset

`-s` : disable stockfish training

`-t` : disable mcts training

Example use:
```bash

# runs train using json file for hyperparameter initialization
./train.sh json

# train and create a new dataset
./train.sh json -m

# Train without using Stockfil
./train.sh json -s

# Train without using mcts
./train.sh json -t

```

