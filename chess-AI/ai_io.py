import os

import torch

def save_model(nnet, save_path, num_saved_models, overwrite):
    """Save given model parameters to external file
    """
    if save_path is not None:
        torch.save(nnet.state_dict(), save_path)
    else:
        # Iterate through the number of models saved
        for i in range(num_saved_models):
            if not (os.path.isfile(f'./models/models-{i + 1}.pt')):
                if overwrite and i != 0:
                    torch.save(nnet.state_dict(), f'./models/models-{i}.pt')
                    break
                torch.save(nnet.state_dict(), f'./models/models-{i + 1}.pt')
                break
            if i == num_saved_models - 1:
                torch.save(nnet.state_dict(), f'./models/models-{num_saved_models}.pt')


def load_model(nnet, model_path, num_saved_models):
    """Load model parameters into given network from external file
    """
    if model_path is not None:
        nnet.load_state_dict(torch.load(model_path))
    else:
        for i in range(num_saved_models):
            if not (os.path.isfile(f'./models-{i + 2}.pt')):
                if i != 0:
                    nnet.load_state_dict(torch.load(f'./models-{i + 1}.pt'))
                break


def main():
    pass

if __name__ == "__main__":
    main()
    