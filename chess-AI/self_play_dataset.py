from torch.utils.data import Dataset
import torch

class SelfPlayDataset(Dataset):
    def __init__(self, mcts_probs, mcts_evals, boards):
        self.mcts_probs = torch.tensor(mcts_probs)
        self.mcts_evals = torch.tensor(mcts_evals)
        self.boards = boards
        self.num_moves = self.mcts_evals.size
    
    def __getitem__(self, index):
        return self.mcts_probs[index], self.mcts_evals[index], self.boards[index]
    
    def __len__(self):
        return self.num_moves