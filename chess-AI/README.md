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

# Train without using Stockfish
./train.sh json -s

# Train without using mcts
./train.sh json -t

```


# Params.json

`dataset_saving` : configuration specifying how the stockfish generated dataset will be loaded and saved

`data_dir` : local directory where the dataset will be saved and loaded from

`file_name` : file name of the dataset 

`figshare_load` : flag specifying if the dataset should be loaded from figshare

`local_load` : flag specifying if the dataset should be loaded from a local directory

`figshare_save` : flag specifying if generated dataset should be saved to figshare

---

`model_saving` : configuration specifying how the model params will be loaded and saved

`model_dir` : local directory where the model params will be saved and loaded from

`file_name` : file name of the model params

`figshare_load` : flag specifying if the model params should be loaded from figshare

`local_load` : flag specifying if the model params should be loaded from a local directory

`figshare_save` : flag specifying if the model params should be saved to figshare

`mcts_check_freq` : how many games the model params should be saved after during MCTS training 

---

`misc_parameters` : miscellaneous hyperparameters for training

`lr` : learning rate for neural network backpropagation

`momentum` : hyperparameter used in gradient descent for training

`weight_decay` : hyperparameter used in gradient descent to reduce overfitting

---

`stockfish` : configuration for training and building the stockfish dataset

`epochs` : the number of full iterations over the dataset during training

`batch_size` : the number of samples to train on for each weight update

`games` : the amount of games played using stockfish

`elo ` : the skill level of the stockfish agent

`depth ` : how many moves ahead the stockfish agent should search at each state of the game. A larger depth is slower but will result in a better player 

---

`mcts` : configuration for training using MCTS

`epochs` : the number of full iterations over the dataset 

`batch_size` :  the number of samples to train on for each weight update

`games` : the amount of full games to play using MCTS

`exploration` : hyperparameter for MCTS that affects the UCT formula for move selection

`training_episodes` :  amount of training iterations for MCTS. Each iteration involves building a new set of move probabilities

`simulations` : the amount of MCTS tree searchs to perform at a certain state

---

`currently_unused` : parameters that are currently not being used in training. 

`move_epsilon` : N/A

`probability_scalar` : N/A

`sigmoid_scalar`  : N/A

`temperature` : N/A