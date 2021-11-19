
class Mcts:
    def search(s,game,nnet):

        if game.gameEnded(s): return  -game.gameReward(s) # returns the negative of the game reward, why negative?

        if s not in visited # if a state has not been visited then you must find the predictions made by the model
            visited.add(s) # will mark as visisted
            P[s],v = nnet.predict(s) # this is like the rollout phase but using the nueral network
            return -value
        
        max_u, best_a = -float("inf"), -1 

        for a in game.getvalidActions(s): # finds the valid actions from current state

            u = Q[s][a] + c_puct*P[s][a]*sqrt(sum(N[s]))/(1+N[s][a]) # formula for calculating the action to take

            if u>max_u:
                max_u = u  # assigns best u
                best_a = a  # assigns best a
        
            a = best_a
            sp = game.nextState(s,a) # want to take the the best action
            v = search(sp,game,nnet) # gets the value of the the action

            Q[s][a] = (N[s][a] * Q[s][a] +v)/N[s][a]+1) # calculates Q
            N[s][a] += 1  # adds 1 to represent the node was visited

            return -v

    """"
    1. Initialize nn with random weights, starting with a random policy and value network
    2. Play a number of games of self play
    3. In each turn of the game perform a fixed number of MCTS simulations from the current state
    4. Pick a move by sampling the improved policy 
    """"

class Train:

    def policyIterSp(game)

        nnet = initNet() # initializes nueral network
        examples = []
        for i in range(numIters):
            for e in range(numEps):
                examples += executeEpisode(game,nnet) # recieves training examples
            new_nnet = trainNNet(examples) # trains new nnet on new training examples
            frac_win = pit(new_nnet,nnet) # play the two nueral networks against each other
            if frac_win > threshold: 
                nnet = new_nnet # pick the winning model
        return nnet


    def executeEpisde(game,nnet):

        examples = []
        s = game.startState() # get the start state of the game
        mcts = Mcts() # instantiate the MCTS class
 
        while True: 

            for _ in range(numMCTSims):
                mcsts.search(s,game,nnet) #performs numMCTSims monte carlo simulations

            examples.append([s,mcts.pi(s),None]) ## append the state, and improved policy, None refers to not knowing the values yet

            a = random.choice(len(mcts.pi(s)), p=mcts.pi(s)) # choose a random move from the improved policy
            s= game.nextState(s,a) # try that random move

            if game.gameEnded(s): # if hte game is over then you need to assign rewards to all the examples
                examples = assignRewards(examples,game.gameReward(s))
                return examples
