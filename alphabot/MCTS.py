import math
import numpy as np
import copy
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        the state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(state, create_copy=True)

        s = self.game.stringRepresentation(state)

        counts = [self.Nsa[(s,(a,b))] if (s,(a,b)) in self.Nsa else 0 for a in range(21) for b in range(16)]
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs
        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, state, create_copy):
        """
        NEEDS TO RUN ON DEEPCOPY!!!

        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current state
        """
        if create_copy:
            self.game_copy = copy.deepcopy(self.game.b.game)

        s = self.game.stringRepresentation(state)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(1)
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(state)
            valids = self.game.getValidMoves(1, self.game_copy)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(21):
            for b in range(16):
                if valids[a,b]:
                    if (s,(a,b)) in self.Qsa:
                        u = self.Qsa[(s,(a,b))] + self.args.cpuct*self.Ps[s][a,b]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,(a,b))])
                    else:
                        u = self.args.cpuct*self.Ps[s][a,b]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = (a,b)

        a = best_act
        try:
            next_s, next_player = self.game.getNextState(1, a, self.game_copy)
        except:
            # fixes recursion errors by preventing infinite loop while traversing position tree
            next_s, next_player = self.game.getNextState(1, (19,0), self.game_copy)
            cur_best = -float('inf')
        # next_s = self.game.getState(next_player)

        v = self.search(next_s, create_copy=False) #call recursively
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v
