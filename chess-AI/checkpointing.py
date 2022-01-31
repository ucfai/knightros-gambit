import torch

def save_model(nnet):
    '''Save given model parameters to external file
    '''
    torch.save(nnet.state_dict(), 'chess-AI/models.pt')

def load_model(nnet):
    '''Load model parameters into given network from external file
    '''
    nnet.load_state_dict(torch.load('chess-AI/models.pt'))
    return nnet

def main():
    pass

if __name__ == "__main__":
    main()
    