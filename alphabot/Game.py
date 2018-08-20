import numpy as np
from utils import Board, UnhandledAction
from fireplace.exceptions import GameOver

class YEET:
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
        self.b = Board()

    def getInitGame(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # self.b.isolateSet()
        self.b.initGame()
        return self.b.game

    def getNextState(self, player, action, game_instance=Board.game):
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
        if game_instance == None:
            game_instance = Board.game

        try:
            self.b.performAction(action, player, game_instance)
        except GameOver:
            raise GameOver
        next_state = self.b.getState(player, game_instance)
        return (next_state, -player)

    def getValidMoves(self, player, game_instance=Board.game):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        # if player == 1:
        #     current_player = self.b.players[0]
        # elif player == -1:
        #     current_player = self.b.players[1]
        if game_instance == None:
            game_instance = Board.game
        return self.b.getValidMoves(game_instance)



    def getGameEnded(self, player, game_instance=Board.game):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """
        if game_instance == None:
            game_instance = Board.game

        # if player == 1:
        #     current_player = game_instance.players[0]
        # elif player == -1:
        #     current_player = game_instance.players[1]
        current_player = game_instance.current_player

        if current_player.playstate == 4:
            return 1
        elif current_player.playstate == 5:
            return -1
        elif current_player.playstate == 6:
            return 0.0001
        elif game_instance.turn > 180:
            game_instance.ended = True
            return 0.0001
        return 0

    def getState(self, player, game_instance=Board.game):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            state: see gameUtils.getState for info
        """
        # if player == 1:
        #     current_player = self.b.players[0]
        # elif player == -1:
        #     current_player = self.b.players[1]
        if game_instance == None:
            game_instance = Board.game

        return self.b.getState(player, game_instance)
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
        # assert(len(pi) == len(state))
        assert(len(pi) == 168)
        pi_board = np.reshape(pi, (21, 9))
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
