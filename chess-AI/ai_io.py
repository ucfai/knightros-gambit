import torch

def save_model(nnet, save_path):
    '''Save given model parameters to external file
    '''
    torch.save(nnet.state_dict(), f'{save_path}')

def load_model(nnet, save_path):
    '''Load model parameters into given network from external file
    '''
    nnet.load_state_dict(torch.load(f'{save_path}'))
    return nnet

def main():
    pass

if __name__ == "__main__":
    main()
    