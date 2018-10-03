import random
import numpy as np
from fireplace import cards
from fireplace.exceptions import GameOver, InvalidAction
from fireplace.game import Game
from fireplace.player import Player
from fireplace.utils import random_draft
from hearthstone.enums import CardClass
import pickle

from .exceptions import UnhandledAction


class Board:
    """
    This class interacts with Game.py to initialize the game, 
    return states, and return actions
    """
    game = None
    players = ['', '']

    def __init__(self):
        self.num_actions = 23
        self.is_basic = True

    def isolateSet(self, filename='notbasicset', set='CardSet.CORE'):
        # isolates the specified card set for exclusion in drafting
        cards.db.initialize()
        extraset = []
        for index, card in cards.db.items():
            if str(card.card_set) != set:
                    extraset.append(card.id)
        with open(f'{filename}.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(extraset, filehandle)

    def initGame(self):
        cards.db.initialize()
        if self.is_basic: #create quick simple game
            with open('notbasic.data', 'rb') as f:
                extra_set = pickle.load(f)
            p1 = 6 #priest
            p2 = 7 #rogue
            deck1 = random_draft(CardClass(p1), exclude=extra_set)
            deck2 = random_draft(CardClass(p2), exclude=extra_set)
        else:
            p1 = random.randint(1, 9)
            p2 = random.randint(1, 9)
            deck1 = random_draft(CardClass(p1))
            deck2 = random_draft(CardClass(p2))
        Board.players[0] = Player("Player1", deck1, CardClass(p1).default_hero)
        Board.players[1] = Player("Player2", deck2, CardClass(p2).default_hero)
        game = Game(players=self.players)
        game.start()

        # Skip mulligan for now
        for player in game.players:
            cards_to_mulligan = random.sample(player.choice.cards, 0)
            player.choice.choose(*cards_to_mulligan)

        # self.start_player = game.current_player
        game.player_to_start = game.current_player
        Board.game = game
        return game

    def getValidMoves(self, game_instance):
        actions = np.zeros((21,18))
        player = game_instance.current_player
        #If the player is being given a choice, return only valid choices
        if player.choice:
            for index, card in enumerate(player.choice.cards):
                actions[20, index] = 1
                #actions.append(("choose", card, None))

        else:
            # add cards in hand
            for index, card in enumerate(player.hand):
                if card.is_playable():
                    if card.requires_target():
                        for target, card in enumerate(card.targets):
                            actions[index, target] = 1
                    elif card.must_choose_one:
                        for choice, card in enumerate(card.choose_cards):
                            actions[index, choice] = 1
                    else:
                        actions[index] = 1
            # add targets available to minions that can attack
            for position, minion in enumerate(player.field):
                if minion.can_attack():
                    for target, card in enumerate(minion.attack_targets):
                        actions[position+10, target] = 1
            # add hero power and targets if applicable
            if player.hero.power.is_usable():
                if player.hero.power.requires_target():
                    for target, card in enumerate(player.hero.power.targets):
                        actions[17, target] = 1
                else:
                    actions[17] = 1
            # add hero attacking if applicable
            if player.hero.can_attack():
                for target, card in enumerate(player.hero.attack_targets):
                    actions[18, target] = 1
            # add end turn
            actions[19,1] = 1
        return actions

    def performAction(self, a, player, game_instance):
        """
        utilty to convert an action tuple
        into an action input
        Args:
            a, a tuple representing index of action
            player, 
            game,
        """
        player = game_instance.current_player
        if not game_instance.ended:
            try:
                if 0 <= a[0] <= 9:
                    if player.hand[a[0]].requires_target():
                        player.hand[a[0]].play(player.hand[a[0]].targets[a[1]])
                    elif player.hand[a[0]].must_choose_one:
                        player.hand[a[0]].play(choose=player.hand[a[0]].choose_targets[a[1]])
                    else:
                        player.hand[a[0]].play()
                elif 10 <= a[0] <= 16:
                    player.field[a[0]-10].attack(player.field[a[0]-10].attack_targets[a[1]])
                elif a[0] == 17:
                    if player.hero.power.requires_target():
                            player.hero.power.use(player.hero.power.play_targets[a[1]])
                    else:
                        player.hero.power.use()
                elif a[0] == 18:
                    player.hero.attack(player.hero.attack_targets[a[1]])
                elif a[0] == 19:
                    player.game.end_turn()
                elif a[0] == 20 and not player.choice:
                    player.game.end_turn()
                elif player.choice:
                    player.choice.choose(player.choice.cards[a[1]])
                else:
                    raise UnhandledAction
            except UnhandledAction:
                print("Attempted to take an inappropriate action!")
                print(a)
                raise
            except InvalidAction:
                print("Attempted to do something I can't!")
                player.game.end_turn()
            except IndexError:
                try:
                    player.game.end_turn()
                except GameOver:
                    pass
            except GameOver:
                pass


    def getState(self, player, game_instance):
        """
        Args:
            game, the current game object
            player, the player from whose perspective to analyze the state
        return:
            a numpy array features extracted from the
            supplied game.
        """
        s = np.zeros(263, dtype=np.int32)

        p1 = game_instance.current_player
        p2 = p1.opponent

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
        return s