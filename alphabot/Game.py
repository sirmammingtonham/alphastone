import random
import numpy as np
import sys
import copy
from fireplace.game import Game
from fireplace.player import Player
from fireplace.utils import random_draft
from fireplace import cards
from fireplace.exceptions import GameOver, InvalidAction
from hearthstone.enums import CardClass, CardType
from utils import Board, UnhandledAction

class YEET():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.
    21 possible actions per move, and 8 possible targets per action + 1 if no targets
    is_basic = True initializes game between priest and rogue only
    """
    
    def __init__(self, is_basic=True):
        self.num_actions = 21
        self.players = ['player1', 'player2']
        self.is_basic = is_basic

    def getInitGame(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        b = Board()
        b.initEnvi()
        b.initGame()
        return b.game

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
                num_actions*8 targets, 152 total
        """
        return [self.num_actions, 9]

    def getNextState(self, player, action):
        """
        Input:
            board: current board (gameUtils)
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)

            all actions executed by copy_player to preserve new game

        """
        b = Board()
        if player == 1:
            current_player = b.players[0]
        elif player == -1:
            current_player = b.players[1]

        b.performAction(action, current_player) ##update this function to support new 19x8 action type
        next_state = b.getState(current_player) ###need to figure out what the game object is
        return (b.game, -player)

    def getValidMoves(self, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        b = Board()
        if player == 1:
            current_player = b.players[0]
        elif player == -1:
            current_player = b.players[1]
        return b.getValidMoves(current_player)



    def getGameEnded(self, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """
        b = Board()
        if player == 1:
            current_player = b.players[0]
        elif player == -1:
            current_player = b.players[1]

        if current_player.playstate == 4:
            return 1
        elif current_player.playstate == 5:
            return -1
        elif current_player.playstate == 6:
            return 0.0001
        return 0

    def getState(self, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            state: see gameUtils.getState for info
        """
        b = Board()
        if player == 1:
            current_player = b.players[0]
        elif player == -1:
            current_player = b.players[1]
        
        return b.getState(current_player)
        # return b.game

    def getSymmetries(self, state, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        assert(len(pi) == len(state))
        pi_board = np.reshape(pi[:-1], (21, 9))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newS = np.rot90(state, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newS = np.fliplr(newS)
                    newPi = np.fliplr(newPi)
                l += [(newS, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, state):
        """
        Input:
            state: np array of state

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return state.tostring()
