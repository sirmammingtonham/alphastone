'''
neural net output as follows:
21x16 2d tensor = a
    a[0-9,] encode for playing card in hand in position 
    a[10-16,] encode for attacking with minion at position
    a[17,] encode for hero power
    a[18,] encode for hero attack
    a[19,] encode for end turn
    a[20,] encode for card index when given choice
    a[,0-15] encode for targeting available target at index (2 for heroes, 14 for board)
'''
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
def get_emb(ni,nf):
    e = nn.Embedding(ni, nf)
    e.weight.data.uniform_(-0.01,0.01)
    return e

class EmbeddingDotBias(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_movies, n_factors), (n_users,1), (n_movies,1)
        ]]
        
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        um = (self.u(users)* self.m(movies)).sum(1)
        res = um + self.ub(users).squeeze() + self.mb(movies).squeeze()
        res = F.sigmoid(res) * (max_rating-min_rating) + min_rating
        return res

class EmbeddingNet(nn.Module):
    def __init__(self, n_users, n_movies, nh=10, p1=0.05, p2=0.5):
        super().__init__()
        (self.u, self.m) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_movies, n_factors)]]
        self.lin1 = nn.Linear(n_factors*2, nh)
        self.lin2 = nn.Linear(nh, 1)
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        x = self.drop1(torch.cat([self.u(users),self.m(movies)], dim=1))
        x = self.drop2(F.relu(self.lin1(x)))
        return F.sigmoid(self.lin2(x)) * (max_rating-min_rating+1) + min_rating-0.5
'''
class BasicBlock(nn.Module):
    def __init__(self, ni, nf):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(ni, nf, kernel_size=3)
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, ni, nf, kernel_size=3):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv1d(nf, nf,
                kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(nf, nf,
                kernel_size=kernel_size, padding=1)

        self.bn1 = nn.BatchNorm1d(nf)
        self.bn2 = nn.BatchNorm1d(nf)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(residual+x)))
        return x

class DQN(nn.Module):
    def __init__(self, game, args):

        self.layers = [16, 32, 64, 128, 256]
        self.state_size = 263
        self.args = args

        super().__init__()
        #first conv layer (input as state, feed into res layers)
        self.convx = nn.Conv1d(1, 16, #append last three turns onto input as 3rd dim
            kernel_size=3, padding=1)
        self.bnx = nn.BatchNorm1d(16)

        #residual layers (10)
        self.layers1 = nn.ModuleList([BasicBlock(self.layers[i], self.layers[i+1])
            for i in range(len(self.layers) - 1)])
        self.layers2 = nn.ModuleList([ResBlock(self.layers[i+1], self.layers[i+1])
            for i in range(len(self.layers) - 1)])
        self.layers3 = nn.ModuleList([ResBlock(self.layers[i+1], self.layers[i+1])
            for i in range(len(self.layers) - 1)])

        #policy head (output as action probabilities (size of actions))
        self.pi_conv1 = nn.Conv1d(256, 2, kernel_size=1)
        self.pi_bn1 = nn.BatchNorm1d(2)
        self.pi_fc1 = nn.Linear(2*255, 21*18)
        # self.pi_fc2 = nn.Linear(18, 18)
        self.pi_conv2 = nn.Conv1d(21, 21, kernel_size=1, stride=1)

        #value head (output as state value [-1,1])
        self.v_conv1 = nn.Conv1d(256, 4, kernel_size=1)
        self.v_bn1 = nn.BatchNorm1d(4)
        self.v_fc1 = nn.Linear(4*255, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        #train conv layer --> feed into res layers, 
        #pass into action and value seperately and return
        x = F.relu(self.bnx(self.convx(state_input))) #convlayer of state into bn into relu
        for l1,l2,l3 in zip(self.layers1, self.layers2, self.layers3):
            x = l3(l2(l1(x))) #conv into bn into relu(orig + conv into bn) into each layer batch

        #action policy head (action probabilities)
        pi = F.relu(self.pi_bn1(self.pi_conv1(x))) #feed resnet into policy head
        pi = pi.view(-1, 2*255)
        pi = F.relu(self.pi_fc1(pi))
        pi = pi.view(-1, 21, 18)
        pi = F.log_softmax(self.pi_conv2(pi), dim=1)

        #value head (score of board state)
        v = F.relu(self.v_bn1(self.v_conv1(x))) #feed resnet into value head
        v = v.view(-1, 4*255)
        v = F.relu(self.v_fc1(v))
        v = F.tanh(self.v_fc2(v))   

        return pi, v