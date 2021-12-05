from mcts import Mcts
import random
import chess
from nn_layout import PlayNetwork
from output_representation import PlayNetworkPolicyConverter

class Train: # algorithm for training the model.

    def __init__(self,simulations):
        self.start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.simulations = simulations
        self.trainingSet = []

    def selfPlay():
        # TODO: algorithm for the networks to play against each other
        pass
  
    def executeEpisode(self,nnet):

        self.trainingSet = [] # clearn traing set after each episode
        board = chess.Board(self.start_fen) # creates board object
        fen_string = self.start_fen #sets starting fen
        mcts = Mcts() # instantiate the MCTS class

        while True: 
            for _ in range(self.simulations):
                mcts.search(board,nnet) # performs mcts
            
            converter = PlayNetworkPolicyConverter()
            keys,policy = mcts.findSearchProbs(fen_string) # gets the search probs and associated keys
            choice = random.randint(0,len(policy) - 1) # selects random policy
            policy = converter.compute_full_search_probs(keys, policy, board)

            self.trainingSet.append([fen_string,policy,None]) # adds the training example

            action = keys[choice] # makes a random move
            action = chess.Move.from_uci(action) # gets the action to make
            board.push(action)
             
            fen_string = board.fen()

            if board.is_game_over():  # if hte engine is over then you need to assign rewards to all the examples
                self.assignRewards(keys,policy,board,1)
                return
        
    def assignRewards(self,keys,policy,board,reward): # assigns the reward to training examples

        for i in self.trainingSet:   
            winner = board.Outcome().winner
            if not winner:
                i[2] = 0
            if winner == chess.WHITE:
                i[2] = 1 
            else: 
                i[2] = -1

        print(board)    
        print(board.outcome().termination)
   
def main():
    nnet = PlayNetwork()
    train = Train(100)
    train.executeEpisode(nnet)

if __name__ == "__main__":
    main()