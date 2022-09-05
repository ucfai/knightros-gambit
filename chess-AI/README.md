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

`dataset_saving` : configurations for how the stockfish dataset will be loaded and saved

`data_dir` : local directory for where the dataset will be saved

`file_name` : file name to load from figshare or local dir

`figshare_load` : flag for rather the dataset should be loaded from figshare

`local_load` : flag for if the dataset should be loaded from your local directory

`figshare_save` : flag for if you should save the generated dataset to figshare

---

`model_saving` : configurations for how the model parameters will be loaded and saved

`model_dir` : local directory for where the model params will be saved

`file_name` : file name to load from figshare or local dir

`figshare_load` : flag for rather the model params should be loaded from figshare

`local_load` : flag for if the model params should be loaded from your local directory

`figshare_save` : flag for if the trained model should be saved to figshare

`mcts_check_freq` : checkpointing frequency during training

---

`misc_parameters` : miscellaneous hyperparameters for training

`lr` : learning rate for the model

`momentum` : momentum for training

`weight_decay` : weight decay for training

---

`stockfish` : configurations for building a stockfish dataset

`epochs` : 

`batch_size` : 

`games` : 

`elo ` : 

`depth ` : 

---

`mcts` : 

`epochs` : 

`batch_size` : 

`games` : 

`exploration` : 

`training_episodes` : 

`simulations` : 

---

`currently_unused` : 

`move_epsilon`: 

`probability_scalar` : 

`sigmoid_scalar` : 

`temperature` : 