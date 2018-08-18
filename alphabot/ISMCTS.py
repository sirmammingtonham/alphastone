# This is a very simple Python 2.7 implementation of the Information Set Monte Carlo Tree Search algorithm.
# The function ISMCTS(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# An example GameState classes for Knockout Whist is included to give some idea of how you
# can write your own GameState to use ISMCTS in your hidden information game.
# 
# Written by Peter Cowling, Edward Powley, Daniel Whitehouse (University of York, UK) September 2012 - August 2013.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai
# Also read the article accompanying this code at ***URL HERE***

from math import *
import random, sys
import numpy as np
from copy import deepcopy
from utils import Board as Game
EPS = 1e-8


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
    """
    def __init__(self, nnet, move=None, parent=None, playerJustMoved=None, state=None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.visits = 0
        self.avails = 1
        if state != None:
            self.pi, self.v = nnet.predict(state)
        self.W = 0
        self.Q = 0
        self.playerJustMoved = playerJustMoved # the only part of the state that the Node needs later

    def GetUntriedMoves(self, legalMoves):
        """ Return the elements of legalMoves for which this node does not have children.
        """

        # Find all moves for which this node *does* have children
        triedMoves = [child.move for child in self.childNodes]

        # Return all moves that are legal but have not been tried yet-
        return [move for move in np.argwhere(legalMoves == 1) if move not in triedMoves]

    def SelectChild(self, legalMoves, cpuct = 1):
        """ Use the UCB1 formula to select a child node, filtered by the given list of legal moves.
            exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
        """

        # Filter the list of children by the list of legal moves
        legalChildren = [child for child in self.childNodes if child.move in np.argwhere(legalMoves == 1)]

        # Get the child with the highest UCB score
        if legalChildren.Q == 0:
            s = max(legalChildren, key = lambda c: cpuct * c.pi * sqrt(self.visits + EPS))
        else:
            s = max(legalChildren, key = lambda c: c.Q + (cpuct * c.pi * sqrt(c.visits) / (1 + c.avails)))

        # Update availability counts -- it is easier to do this now than during backpropagation
        for child in legalChildren:
            child.avails += 1

        # Return the child selected above
        return s

    def AddChild(self, m, p, s):
        """ Add a new child node for the move m.
            Return the added child node
        """
        n = Node(move=m, parent=self, playerJustMoved=p, state=s)
        self.childNodes.append(n)
        return n

    def Update(self, terminalGame):
        """ Update this node - increment the visit count by one, and increase the win count by the result of terminalState for self.playerJustMoved.
        """
        self.visits += 1
        self.Q = self.W/self.visits
        if Game.getGameEnded(terminalGame, self.playerJustMoved) != 0:
            self.W += Game.getGameEnded(terminalGame, self.playerJustMoved)
        else:
            self.W += self.v

    def __repr__(self):
        return f"[M:{self.move} W/V/A: {self.wins}/{self.visits}/{self.avails} nnet: {self.v}]"

    def TreeToString(self, indent):
        """ Represent the tree as a string, for debugging purposes.
        """
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


class ISMCTS():
    def __init__(self, nnet):
        self.nnet = nnet

    def CloneAndRandomize(self, game):
        """ Create a deep clone of this game state, randomizing any information not visible to the specified observer player.
        """
        game_copy = deepcopy(game)
        enemy = game.current_player.opponent
        combined = enemy.hand + enemy.deck
        random.shuffle(combined)
        enemy.hand, enemy.deck = combined[:len(enemy.hand)], combined[len(enemy.hand):]

        return game_copy

    def getBestAction(self, rootstate, itermax, verbose=False, temp=1):
        """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
            Return the best move from the rootstate.
        """

        rootnode = Node(self.nnet)

        for i in range(itermax):
            node = rootnode
            # Determinize
            game = self.CloneAndRandomize(rootstate)
            # Select
            while Game.getGameEnded(game.current_player, game) == 0 and node.GetUntriedMoves(Game.GetValidMoves()) == []: # node is fully expanded and non-terminal
                node = node.SelectChild(Game.GetValidMoves())
                Game.getNextState(game, game.current_player, node.move)

            #
            untriedMoves = node.GetUntriedMoves(Game.getValidMoves(game) * node.pi)
            if untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
                m = random.choice(untriedMoves)
                player = game.current_player
                Game.getNextState(game, player, m)
                node = node.AddChild(m, player, Game.getState(game)) # add child and descend tree

            # Simulate
            while Game.getGameEnded(node.playerJustMoved, game) == 0: # while state is non-terminal
                Game.getNextState(game, game.current_player, random.choice(np.argwhere(node.pi == 1)))

            # Backpropagate
            while node != None: # backpropagate from the expanded node and work back to the root node
                node.Update(game)
                node = node.parentNode

        # Output some information about the tree - can be omitted
        # if verbose:
        #     print(rootnode.TreeToString(0){})
        # else:
        #     print(rootnode.ChildrenToString())
        return max(rootnode.childNodes, key = lambda c: c.visits**1/temp).move # return the move that was most visited