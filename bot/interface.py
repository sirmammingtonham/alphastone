"""
Interface to interact with fireplace simulator.
Includes two classes: Board and SelfPlay
Board interacts with fireplace and returns state, possible actions, etc.
SelfPlay initiliazes self play games
"""
from fireplace.game import Game
from fireplace.player import Player
from fireplace.utils import random_draft
from fireplace import cards
from fireplace.exceptions import GameOver, InvalidAction
from hearthstone.enums import CardClass, CardType
from .exceptions import UnhandledAction
import random
import numpy as np
import sys

"""
TO DO: 
Test do_move function
Def winner function
Redo all of it rip
"""
#current player return function = game.current_player
class Board(object):
    def __init__(self, is_basic=True):
        self.state = np.zeros(263, dtype=np.int32)
        self.players = ['player1', 'player2']
        self.is_basic = is_basic
        # self.game = Game()


    def init_envi(self):
        """
        Initializes the environment. All basic initialization goes here.
        """
        cards.db.initialize()
        self.last_move = -1


    def init_game(self):
        """
        initializes a game between two players
        Returns:
            game: A game entity representing the start of the game after the mulligan phase
        """
        if self.is_basic: #create quick simple game
            p1 = 6 #priest
            p2 = 7 #rogue
            deck1 = random_draft(CardClass(p1))
            deck2 = random_draft(CardClass(p2))
            self.players[0] = Player("Player1", deck1, CardClass(p1).default_hero)
            self.players[1] = Player("Player2", deck2, CardClass(p2).default_hero)
            self.game = Game(players=self.players)
            self.game.start()

            #Skip mulligan
            for player in self.game.players:
                cards_to_mulligan = random.sample(player.choice.cards, 0)
                player.choice.choose(*cards_to_mulligan)

            return self.game

        else:
            p1 = random.randint(1, 9)
            p2 = random.randint(1, 9)
            #initialize players and randomly draft decks
            #pdb.set_trace()
            deck1 = random_draft(CardClass(p1))
            deck2 = random_draft(CardClass(p2))
            self.players[0] = Player("Player1", deck1, CardClass(p1).default_hero)
            self.players[1] = Player("Player2", deck2, CardClass(p2).default_hero)
            #begin the game
            self.game = Game(players=self.players)
            self.game.start()

            #Skip mulligan for now
            for player in self.game.players:
                cards_to_mulligan = random.sample(player.choice.cards, 0)
                player.choice.choose(*cards_to_mulligan)

            return self.game

    def get_actions(self, player):
        """
        generate a list of tuples representing all valid actions
        format:
            (actiontype, index, target)
        card.requires_target!!
        """
        actions = []

        #If the player is being given a choice, return only valid choices
        if player.choice:
            for card in player.choice.cards:
                actions.append(("choose", card, None))

        else:
            # add cards in hand
            for index, card in enumerate(player.hand):
                if card.is_playable():
                    # summonable minions (note some require a target on play)
                    if card.type == 4:
                        if card.requires_target():
                            for target in card.targets:
                                actions.append(("summon", index, target))
                        else:
                            actions.append(("summon", index, None, None))
                    # playable spells and weapons
                    elif card.requires_target():
                        for target in card.targets:
                            actions.append(("spell", index, target))
                    else:
                        actions.append(("spell", index, None))
            # add targets avalible to minions that can attack
            for position, minion in enumerate(player.field):
                if minion.can_attack():
                    for target in minion.attack_targets:
                        actions.append(("attack", position, target))
            # add hero power and targets if applicable
            if player.hero.power.is_usable():
                if player.hero.power.requires_target():
                    for target in player.hero.power.targets:
                        actions.append(("hero_power", None, target))
                else:
                    actions.append(("hero_power", None, None))
            # add hero attacking if applicable
            if player.hero.can_attack():
                for target in player.hero.attack_targets:
                    actions.append(("hero_attack", None, target))
            # add end turn
            actions.append(("end_turn", None, None))
            self.actions = actions
        return actions

    def get_actionsize(self):
        self.action_size = len(self.actions)
        return self.action_size


    def perform_action(self, a, player, game):
        """
        utilty to convert an action tuple
        into an action input
        Args:
            a, a tuple representing (action, index, target)
        """

        try:

            if a[0] == "summon":
                if a[2] is None:
                    player.hand[a[1]].play()
                else:
                    player.hand[a[1]].play(a[2])
            elif a[0] == "spell":
                if a[2] is None:
                    player.hand[a[1]].play()
                else:
                    player.hand[a[1]].play(a[2])
            elif a[0] == "attack":
                player.field[a[1]].attack(a[2])
            elif a[0] == "hero_power":
                if a[2] is None:
                    player.hero.power.use()
                else:
                    player.hero.power.use(a[2])
            elif a[0] == "hero_attack":
                player.hero.attack(a[2])
            elif a[0] == "end_turn":
                game.end_turn()
            elif a[0] == "choose":
                #print("Player choosing card %r, " % a[1])
                player.choice.choose(a[1])
            else:
                raise UnhandledAction
        except UnhandledAction:
            print("Attempted to take an inappropriate action!\n")
            print(a)
        except GameOver:
            raise


    def get_state(self):
        """
        Args:
            game, the current game object
            player, the player from whose perspective to analyze the state
        return:
            a numpy array features extracted from the
            supplied game.
        """
        s = np.zeros(263, dtype=np.int32)
        c = np.zeros(263, dtype=np.int32)
        for player in self.players:
            p1 = player
            p2 = player.opponent

            #0-9 player1 class, we subtract 1 here because the classes are from 1 to 10
            s[p1.hero.card_class-1] = 1
            #10-19 player2 class
            s[10 + p2.hero.card_class-1] = 1
            i = 20
            # 20-21: current health of current player, then opponent
            s[i] = p1.hero.health
            s[i + 1] = p2.hero.health

            # 22: hero power usable y/n
            s[i + 2] = p1.hero.power.is_usable()*1
            # 23-24: # of mana crystals for you opponent
            s[i + 3] = p1.max_mana
            s[i + 4] = p2.max_mana
            # 25: # of crystals still avalible
            s[i + 5] = p1.mana
            #26-31: weapon equipped y/n, pow., dur. for you, then opponent
            s[i + 6] = 0 if p1.weapon is None else 1
            s[i + 7] = 0 if p1.weapon is None else p1.weapon.damage
            s[i + 8] = 0 if p1.weapon is None else p1.weapon.durability

            s[i + 9] = 0 if p2.weapon is None else 1
            s[i + 10] = 0 if p2.weapon is None else p2.weapon.damage
            s[i + 11] = 0 if p2.weapon is None else p2.weapon.durability

            # 32: number of cards in opponents hand
            s[i + 12] = len(p2.hand)
            #in play minions

            i = 33
            #33-102, your monsters on the field
            p1_minions = len(p1.field)
            for j in range(0, 7):
                if j < p1_minions:
                    # filled y/n, pow, tough, current health, can attack
                    s[i] = 1
                    s[i + 1] = p1.field[j].atk
                    s[i + 2] = p1.field[j].max_health
                    s[i + 3] = p1.field[j].health
                    s[i + 4] = p1.field[j].can_attack()*1
                    # deathrattle, div shield, taunt, stealth y/n
                    s[i + 5] = p1.field[j].has_deathrattle*1
                    s[i + 6] = p1.field[j].divine_shield*1
                    s[i + 7] = p1.field[j].taunt*1
                    s[i + 8] = p1.field[j].stealthed*1
                    s[i + 9] = p1.field[j].silenced*1
                i += 10

            #103-172, enemy monsters on the field
            p2_minions = len(p2.field)
            for j in range(0, 7):
                if j < p2_minions:
                    # filled y/n, pow, tough, current health, can attack
                    s[i] = 1
                    s[i + 1] = p2.field[j].atk
                    s[i + 2] = p2.field[j].max_health
                    s[i + 3] = p2.field[j].health
                    s[i + 4] = p2.field[j].can_attack()*1
                    # deathrattle, div shield, taunt, stealth y/n
                    s[i + 5] = p2.field[j].has_deathrattle*1
                    s[i + 6] = p2.field[j].divine_shield*1
                    s[i + 7] = p2.field[j].taunt*1
                    s[i + 8] = p2.field[j].stealthed*1
                    s[i + 9] = p2.field[j].silenced*1
                i += 10

            #in hand

            #173-262, your cards in hand
            p1_hand = len(p1.hand)
            for j in range(0, 10):
                if j < p1_hand:
                    #card y/n
                    s[i] = 1
                    # minion y/n, attk, hp, battlecry, div shield, deathrattle, taunt
                    s[i + 1] = 1 if p1.hand[j].type == 4 else 0
                    s[i + 2] = p1.hand[j].atk if s[i + 1] == 1 else 0
                    s[i + 2] = p1.hand[j].health if s[i + 1] == 1 else 0
                    s[i + 3] = p1.hand[j].divine_shield*1 if s[i + 1] == 1 else 0
                    s[i + 4] = p1.hand[j].has_deathrattle*1 if s[i + 1] == 1 else 0
                    s[i + 5] = p1.hand[j].taunt*1 if s[i + 1] == 1 else 0
                    # weapon y/n, spell y/n, cost
                    s[i + 6] = 1 if p1.hand[j].type == 7 else 0
                    s[i + 7] = 1 if p1.hand[j].type == 5 else 0
                    s[i + 8] = p1.hand[j].cost
                i += 9
            if player == self.players[0]:
                c = s
                continue
            else:
                self.state = np.append(c, s)
                return self.state
        #     self.state = s
        # return self.state

    def get_state_data(self):
        state = []
        self.get_actions()
        pass



    def check_winner(self):
        for player in self.players:
            if player.playstate == PlayState.WON:
                return True, player
            if player.playstate == PlayState.TIED:
                return True, -1
        return False, -1



class SelfPlay(object):
    def __init__(self, board):
        self.board = board

    def start_play(self, ai_player, mcts_benchmark):
        self.board.init_envi()
        game = self.board.init_game()
        p1, p2 = self.board.players
        while True:
            current_player = game.current_player
            turn_player = players[current_player]
            move = get_actions(turn_player)
            self.board.perform_action(move)
            end, winner = self.board.check_winner()
            if end:
                return winner


    def start_self_play(self, mcts_player, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_envi()
        game = self.board.init_game()

        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = mcts_player.get_action(self.board, 
                temp=temp, return_prob=1)

            #store data for training
            states.append(self.board.get_state())
            mcts_probs.append(move_probs)
            current_players.append(game.current_player)

            #perform move 
            self.board.perform_action(move, game.current_player, game)
            #evaluate game end
            end, winner = self.board.check_winner()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                mcts_player.reset_player()
                return winner, zip(states, mcts_probs, winners_z)

"""
    if __name__ == "__main__":
        initialize()
        game = setup_game()
        try:
            while True:
                if game.current_player.choice:
                    actions = get_actions(game.current_player)
                index = random.randint(0, len(actions)-1)
                perform_action(actions[index], game.current_player, game)
        except GameOver:
            print('Game ended successfully')
        except InvalidAction as err:
            print('Invalid Action: ', err)
        except:
            print('Unexcepted error!!', sys.exc_info()[0])
"""