# Alphastone - Hearthstone Reinforcement Learning AI!
A Hearthstone AI implementation for the [fireplace simulator](https://github.com/jleclanche/fireplace/).
Uses self-play to train and Monte Carlo Tree Search + Neural Network for decision making.
Based off the alphazero algorithm (and its implementation by @suragnair [alpha zero general](https://github.com/suragnair/alpha-zero-general)). 

Hearthstone is an imperfect information game, meaning that some information is always hidden to both players. This is different from the game of go or chess where your opponent's pieces are visible at all times. As a result, we have to randomize all hidden information (opponent's hand and deck) before every search. MCTS is performed on a set of all the information available to the current player (information set mcts).

The neural network is a small-ish resnet in PyTorch defined in [`alphanet.py`](./alphabot/alphanet.py). It is called to evaluate leaf nodes in the search tree and returns a matrix of action probabilities, and the predicted outcome.

References:
1. AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
2. Information Set Monte Carlo Tree Search
3. https://hearthstone.gamepedia.com/Hearthstone_Wiki

## Experiments
Trained a few different models over the course of 2 weeks but only using basic card set (~150 cards) and priest vs rogue. Best model was trained for around 3 days. It shows decision making and is able to beat a random agent ~80% of the time, but much much more training is needed. (Although it almost beat me once when I had terrible luck)


This is my first large python project and is written by a high school student. I don't have formal coding experience so all help and critique is appreciated!

**TO-DO**
- [ ] CLEAN-UP CODE! there's a lot of comments and unnecessary bits from debugging
- [x] Change pit.py to allow for play against trained model
- [ ] Implement ideas in ideas[#1](/../../issues/1)
