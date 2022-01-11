from torch.utils.data import Dataset

class SelfPlayDataset(Dataset):
    def __init__(self, mcts_probs, mcts_evals):
        self.mcts_probs = mcts_probs
        self.mcts_evals = mcts_evals
        self.num_moves = self.mcts_evals.size
    
    def __getitem__(self, index):
        return self.mcts_probs[index], self.mcts_evals[index]
    
    def __len__(self):
        return self.num_moves