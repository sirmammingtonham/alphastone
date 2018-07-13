"""
My implementation of a deep residual network for evaluating the MCST
Need to work on input and output sizes, 
    since state and action are a 1d tensor (list)
layers = [16, 32, 64, 128, 256]

Output only as state value and have mcts evaluate actions?
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
        self.layers3 = nn.ModuleList([ResBlock(layers[i], layers[i+1])
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
        for l,l2,l3 in zip(self.layers1, self.layers2, self.layers3):
            x = l3(l2(l(x))) #conv into bn into relu(orig + conv into bn) into each layer batch

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


class PolicyEvaluator():
    def __init__(self, layers, board, model_file=None):
        self.state_size = board.state.size
        self.weight_decay = 1e-4
        self.policy_value_net = DQN(layers, board).cuda()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                            weight_decay=self.weight_decay)
        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        state_batch = Variable(torch.FloatTensor(state_batch).cuda())
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_func(self, board):
        moveset = board.actions
        current_state = np.ascontiguousarray(board.state())#.reshape(??))

        log_act_probs, value = self.policy_value_net(Variable(
                torch.from_numpy(current_state)).cuda().float)
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(moveset, act_probs[moveset])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = Variable(torch.FloatTensor(state_batch).cuda())
        mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
        winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())

        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.data[0], entropy.data[0]

    def save_model(self, model_file):
        net_params = self.policy_value_net.state_dict()  
        torch.save(net_params, model_file)