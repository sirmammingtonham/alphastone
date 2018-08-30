import Arena
from MCTS import MCTS
from Game import YEET
from NNet import NNetWrapper as NNet
import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, game_instance):
        # display(board)
        idxid = 0
        you = game_instance.current_player

        print(f'YOUR HEALTH: {you.hero.health}')
        print(f'OPPONENT\'S HEALTH: {you.opponent.hero.health}')
        print(f'MANA: {you.mana}')
        print('\n----------Hand----------')
        for idx, card in enumerate(you.hand):
            print(f'Name: {card}, Index: {idx}, Cost: {card.cost}, Is Playable? {card.is_playable()}')
            if card.type == 4:
                print(f'Attack: {card.atk}, Health: {card.health}')

        print('\n----------Your Field----------')
        for idx, card in enumerate(you.field):
            print(f' Name: {card}, Index: {idx+10}, Can Attack? {card.can_attack()}')
        print('\n----------Enemy Field----------')
        for idx, card in enumerate(you.opponent.field):
            print(f'Enemy: {card}')

        print('\n----------Other Actions----------')
        if you.hero.power.is_usable():
            print('Hero Power Available: Index: 17')
        if you.hero.can_attack():
            print('Attack with Weapon, Index: 18')
        print('End Turn, Index: 19 \n')

        actionid = int(input('Enter action index: '))

        if actionid <= 9:
            if you.hand[actionid].requires_target():
                print('Choose a target:')
                for idx, target in enumerate(you.hand[actionid].targets):
                    print(f'Name: {target}, Index: {idx}')
                idxid = int(input('Enter idx:'))

        elif 10 <= actionid <= 16:
            print('Choose a target:')
            for idx, target in enumerate(you.field[actionid - 10].attack_targets):
                print(f'Name: {target}, Index: {idx}')
            idxid = int(input('Enter idx:'))

        elif actionid == 17:
            if you.hero.power.requires_target():
                print('Choose a target:')
                for idx, target in enumerate(you.hero.power.targets):
                    print(f'Name: {target}, Index: {idx}')
                idxid = int(input('Enter idx:'))

        elif actionid == 18:
            print('Choose a target:')
            for idx, target in enumerate(you.hero.power.attack_targets):
                print(f'Name: {target}, Index: {idx}')
            idxid = int(input('Enter idx:'))

        return actionid, idxid


g = YEET(is_basic=True)

# all players
hp = HumanPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/', 'best.pth.tar')
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts = MCTS(g, n1, args)
aip = lambda x: mcts.getActionProb(x, temp=0)

arena = Arena.Arena(aip, hp, g)

if __name__ == '__main__':
    arena.playGames(2, verbose=True)
