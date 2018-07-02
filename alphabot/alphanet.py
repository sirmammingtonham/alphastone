import sys
sys.path.append('..')
from utils import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# def get_emb(ni,nf):
#     e = nn.Embedding(ni, nf)
#     e.weight.data.uniform_(-0.01,0.01)
#     return e

# class EmbeddingDotBias(nn.Module):
#     def __init__(self, n_users, n_movies):
#         super().__init__()
#         (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [
#             (n_users, n_factors), (n_movies, n_factors), (n_users,1), (n_movies,1)
#         ]]
        
#     def forward(self, cats, conts):
#         users,movies = cats[:,0],cats[:,1]
#         um = (self.u(users)* self.m(movies)).sum(1)
#         res = um + self.ub(users).squeeze() + self.mb(movies).squeeze()
#         res = F.sigmoid(res) * (max_rating-min_rating) + min_rating
#         return res

# class EmbeddingNet(nn.Module):
#     def __init__(self, n_users, n_movies, nh=10, p1=0.05, p2=0.5):
#         super().__init__()
#         (self.u, self.m) = [get_emb(*o) for o in [
#             (n_users, n_factors), (n_movies, n_factors)]]
#         self.lin1 = nn.Linear(n_factors*2, nh)
#         self.lin2 = nn.Linear(nh, 1)
#         self.drop1 = nn.Dropout(p1)
#         self.drop2 = nn.Dropout(p2)
        
#     def forward(self, cats, conts):
#         users,movies = cats[:,0],cats[:,1]
#         x = self.drop1(torch.cat([self.u(users),self.m(movies)], dim=1))
#         x = self.drop2(F.relu(self.lin1(x)))
#         return F.sigmoid(self.lin2(x)) * (max_rating-min_rating+1) + min_rating-0.5


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super(ResLayer, self).__init__()

        self.conv1 = nn.Conv2d(ni, nf, stride=stride, 
                kernel_size = kernel_size, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, stride=stride, 
                kernel_size = kernel_size, padding=1)

        self.bn1 = nn.BatchNorm2d(nf)
        self.bn2 = nn.BatchNorm2d(nf)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(residual + self.bn2(self.conv2(x)))
        return x


class DQN(nn.Module):
    def __init__(self, layers, board, output_size):
        """
        output_size = length of array of available actions
        board = state.board
        layers = [16, 32, 64, 128, 256]
        """
        super(DQN, self).__init__()
        self.state_size = board.state.size

        #first conv layer (input as state, feed into res layers)
        self.conv1 = nn.Conv2d(4, 16, #append last two turns onto input as 3rd dim
            kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        #residual layers (40?)
        self.layers1 = nn.ModuleList([ResBlock(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([ResBlock(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])

        #policy head (output as action probabilities (size of actions))
        self.act_conv1 = nn.Conv2d(256, 4, kernel_size=1)
        self.act_bn1 = nn.BatchNorm2d(4)
        self.act_fc1 = nn.Linear(4*self.state_size, output_size)
        
        #value head (output as state value [-1,1])
        self.val_conv1 = nn.Conv2d(256, 2, kernel_size=1)
        self.val_bn1 = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2*state.size, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        #train conv layer --> feed into res layers, 
        #pass into action and value seperately and return

        x = F.relu(self.bn1(self.conv1(state_input))) #convlayer of state into bn into relu

        #residual layers
        for l,l2 in zip(self.layers1, self.layers2):
            x = l2(l(x)) #conv into bn into relu(orig + conv into bn) into each layer batch

        #action policy head (action probabilities)
        x_act = F.relu(self.act_bn1(self.act_conv1(x))) #feed resnet into policy head
        x_act = x_act.view(-1, 4*self.state_size)
        x_act = F.log_softmax((self.act_fc1(x_act)))

        #value head (score of board state)
        x_val = F.relu(self.val_bn1(self.val_conv1(x))) #feed resnet into value head
        x_val = x_val.view(-1, 2*self.state_size)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act, x_val