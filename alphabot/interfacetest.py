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
    def __init__(self, is_basic=True):
        pass

    @classmethod
    def from_bool(cls, v):
        if v == True:
            return 1.0
        else:
            return -1.0
            
    def get_hero_data(self):
    	pass

    def getActionSize

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