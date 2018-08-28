import Arena
from MCTS import MCTS
from Game import YEET as Game
from NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

class HumanPlayer():
    def __init__(self, game):
        self.game = game.b.game

    def play(self):
        # display(board)
        idxid = 0
        you = self.game.b.game.current_player
        print('Hand:')
        for idx, card in enumerate(you.hand):
            print(f'Name: {card.name}, Index: {idx}, Is Playable? {card.is_playable()}')

        print('Field:')
        for idx, card in enumerate(you.field):
            print(f' Name: {card.name}, Index: {idx}, Can Attack? {card.can_attack()}')
        for idx, card in enumerate(you.opponent.field):
            print(f'Enemy: {card.name}')

        if you.hero.power.is_usable():
            print('Hero Power Available: Index: 17')
        if you.hero.can_attack():
            print('Attack with Weapon, Index: 18')
        print('End Turn, Index: 19')

        actionid = int(input('Enter action index: '))

        if actionid <= 9:
            if you.hand[actionid].requires_target():
                print('Choose a target:')
                for idx, target in enumerate(you.hand[actionid].targets):
                    print(f'Name: {target.name}, Index: {idx}')
                idxid = int(input('Enter idx:'))

        elif 10 <= actionid <= 16:
            print('Choose a target:')
            for idx, target in enumerate(you.field[actionid-10].attack_targets):
                print(f'Name: {target.name}, Index: {idx}')
            idxid = int(input('Enter idx:'))

        elif actionid == 17:
            if you.hero.power.requires_target():
                print('Choose a target:')
                for idx, target in enumerate(you.hero.power.targets):
                    print(f'Name: {target.name}, Index: {idx}')
                idxid = int(input('Enter idx:'))

        elif actionid == 18:
            print('Choose a target:')
            for idx, target in enumerate(you.hero.power.attack_targets):
                print(f'Name: {target.name}, Index: {idx}')
            idxid = int(input('Enter idx:'))

        return actionid, idxid

g = Game(is_basic=True)

# all players
hp = HumanPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts = MCTS(g, n1, args)
aip= lambda x: np.argmax(mcts.getActionProb(x, temp=0))


arena = Arena.Arena(aip, hp, g, display=display)
print(arena.playGames(2, verbose=True))
