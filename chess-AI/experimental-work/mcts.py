## https://towardsdatascience.com/monte-carlo-tree-search-implementing-reinforcement-learning-in-real-time-game-player-a9c412ebeff5

class Node: ## node class

    def __init__(self,move: tuple= None, parent: object = None):

        self.move = move   ## the move to get to the node
        self.parent = parent ## the parent of the node
        self.N = 0 ## initially 0
            self.Q = 0 ## initial 0
            self.children = {}

    def add_children(self,children:dict) --> None:

        for child in children:
            self.children[child.move] = child # adds , the child nodes

    def value(self, explore:float = 0.5) # 0.5 is the exploration value , can be controlled

        if self.N == 0:
            return 0 if explore == 0 else Math.INF
        else
            return self.Q / self.N  + explore * sqrt(2*log(self.parent.N)/self.N) ## UCT formula for finding value

 class mctsAgent:

     self.root = Node() # tree root node, will be the parent of all nodes

     def search(self, time_budget: int) -> None:

        """ 
        Search and update the search tree for a specified amount of time
        """
        num_rollouts = 0

        while num_rollouts < 10000:

            node,state = self.select_node() ## selects a node
            turn = state.turn() ## state refers to the chess engine
            outcome = self.roll_out(state) ## plays random moves until the end is reached finds out if state is part of winning game
            self.back_prop(node,turn,outcome) ## updates all the values
            num_rollouts += 1

    
    def select_node(self) -> tuple:

        node = self.root
        state = gameState
    
        while len(node.children) != 0:

            children = node.children.values() ## gets the values of the child nodes
            max_value = max(children,key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values()]
                if n.value == max_value] ## find nodes with max values
            node = choice(max_nodes) ## choses a random value from the max nodes
            state.play(node.move) ## makes move from the chose node

            if node.N == 0 ## if it hasn't been explored then want to check it out
                return node, state

         if self.expand(node,state): ## will expand if theres no children
            node = choice(list(node.children.values())) ## picks the child with the highest with the best value
            state.play(node.move)

        return node,state

    def expand(parent: Node, state: GameState) -> bool:

        ## generate children of the parent node based on the available moves in the passed game state and add to the tree

        children = []

        if state.winner != True # cannot expand if the game is over 
            return False ## game is over

        for move in state.moves(): ## finds valid moves
            children.append(Node(move,parent))
        
        parent.add_children(children) 
        return True


    def moves(self) -> list:
        """
        Get a list of all moves possible on the current board.
        """
        moves = []
        for y in range(self.size):
            for x in range(self.size):
                if is_valid(self.board[x, y]): ## means valid move
                    moves.append((x, y))
        return moves

    def roll_out (state:GameState) -> int:

        """
        Gets the state of the game and keeps playing random moves
        """

        while state.winner == False ## reached end of the game 
            move = choice(moves)
            state.play(move)
            moves.remove(move)

         return state.winner   


    def back_prop(node: Node, turn: int, outcome:int) -> None:
        """
        Update node stats on the path from the passed node to root to reflet the outcome
        of a randomsly simulated playout

        Node : node we get from select node
        Turn : Indicates the player turn in the state which was the second output of select_node
        Outcome output of simulation phase which is winner of the simlulation

        """
        
        # outcome == turn refers to if the winner matches the current player

        reward = 0 if outcome == turn else 1 ## need to see if the oppenent won teh game

        while node is not Node:

            node.N += 1
            node.Q += reward
            node = node.parent
            reward = 0 if reward = 1 else 1


    def tree_size(self) -> int:
        """
        Count nodes in tree by BFS.
        """
        Q = Queue()
        count = 0
        Q.put(self.root)
        while not Q.empty():
            node = Q.get()
            count += 1
            for child in node.children.values():
                Q.put(child)
        return count


        def best_move(self) -> tuple:
            """
            Return the best move according to the current tree.
            Returns:
                best move in terms of the most simulations number unless the game is over
            """
            if self.root_state.winner != GameMeta.PLAYERS['none']:
                return GameMeta.GAME_OVER

            # choose the move of the most simulated node breaking ties randomly
            max_value = max(self.root.children.values(), key=lambda n: n.N).N
            max_nodes = [n for n in self.root.children.values() if n.N == max_value]
            bestchild = choice(max_nodes)
            return bestchild.move