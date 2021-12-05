from mcts import Mcts
import random
import chess
from nn_layout import PlayNetwork

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
            
            keys,policy = mcts.findSearchProbs(fen_string) # gets the search probs and associated keys

            #TODO: For training example need to get (8*8*73) grid oof values

            self.trainingSet.append([fen_string,policy,None]) # append the state, and improved policy, None refers to not knowing the values yet

            choice = random.randint(0,len(policy) - 1) # selects random policy
            action = keys[choice] # getes the random action from the random policy
            action = chess.Move.from_uci(action) # gets the action to make
            board.push(action)
            fen_string = board.fen()

            if board.is_game_over():  # if hte engine is over then you need to assign rewards to all the examples
                self.assignRewards(board,1)
                return
        
    def assignRewards(self,board,reward): # assigns the reward to training examples
        for i in self.trainingSet:
            i[2] = reward
        print(board)    
        print(board.outcome().termination)
        #print(len(self.trainingSet))  
        #print(self.trainingSet)  

def main():
    nnet = PlayNetwork()
    train = Train(10)
    train.executeEpisode(nnet)

if __name__ == "__main__":
    main()