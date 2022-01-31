import torch

def save_model(nnet):
    torch.save(nnet.state_dict(), 'chess-AI/models.pt')

def load_model(nnet):
    nnet.load_state_dict(torch.load('chess-AI/models.pt'))
    return nnet

def main():
    pass

if __name__ == "__main__":
    main()