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
from .gameUtils import Board
from .exceptions import UnhandledAction

class Game():
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

    def init_envi(self):
        cards.db.initialize()

    def init_game(self):
        if self.is_basic: #create quick simple game
            p1 = 6 #priest
            p2 = 7 #rogue
        else:
            p1 = random.randint(1, 9)
            p2 = random.randint(1, 9)
        deck1 = random_draft(CardClass(p1))
        deck2 = random_draft(CardClass(p2))
        self.players[0] = Player("Player1", deck1, CardClass(p1).default_hero)
        self.players[1] = Player("Player2", deck2, CardClass(p2).default_hero)
        game = Game(players=self.players)
        game.start()

        #Skip mulligan for now
        for player in self.game.players:
            cards_to_mulligan = random.sample(player.choice.cards, 0)
            player.choice.choose(*cards_to_mulligan)

        return game

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        b = Board()
        return np.array(b.getState(game, player))#???

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
                num_actions*8 targets, 152 total
        """
        return [self.num_actions, 8]

    def getNextState(self, player, action):
        """
        Input:
            board: current board (gameUtils)
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)

        """
        b = copy.deepcopy(Board())
        if player == 1:
            current_player = b.player[0]
        elif player == -1:
            current_player = b.player[1]

        b.performAction(action) ##update this function to support new 19x8 action type
        next_state = b.getState(b, current_player) ###need to figure out what the game object is
        return (next_state, -player)

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        if player == 1:
            current_player = self.player[0]
        elif player == -1:
            current_player = self.player[1]



    def getGameEnded(self, board, player):
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

        if current_player.playstate == PlayState.WON:
            return 1
        elif player.playstate == PlayState.LOST:
            return -1
        elif player.PlayState == PlayState.TIED:
            return 0.00001
        return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
