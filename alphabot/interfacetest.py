import random
import numpy as np
import sys
from fireplace.game import Game
from fireplace.player import Player
from fireplace.utils import random_draft
from fireplace import cards
from fireplace.exceptions import GameOver, InvalidAction
from hearthstone.enums import CardClass, CardType
from .exceptions import UnhandledAction

class Game():
    def __init__(self, num_actions=19, is_basic=True):
        self.num_actions = num_actions
            
    def getInitBoard(self):
    	pass

    def getBoardSize(self):
        pass

    def getActionSize(self):
        return self.num_actions

    def getNextState(self, board, player, action):
        pass

    
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
        state = []
        #cards in hand
        for card in self.hand:
            state.append(card.entity_id)
        while (len(state) < 10):
            state.append(-1)
        #my hero
        state.append(self.hero.health)
        state.append(self.hero.atk)
        #my minions
        for minion in self.field:
            state.append(minion.entity_id) #eventually link entity_id to embedding
            state.append(minion.health)
            state.append(minion.atk)
            state.append(self.get_int_from_bool(minion.taunt))
            state.append(self.get_int_from_bool(minion.divine_shield))
        while(len(state) < 47):
            state.append(-1)
        #enemy hero
        state.append(self.opponent.hero.health)
        state.append(self.opponent.hero.atk)
        #enemy minions
        for minion in self.opponent.field:
            state.append(minion.health)
            state.append(minion.atk)
            state.append(self.get_int_from_bool(minion.taunt))
            state.append(self.get_int_from_bool(minion.divine_shield))
        while(len(state) < 77):
            state.append(-1)
        return state