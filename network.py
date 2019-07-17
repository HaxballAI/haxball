import torch
from collections import namedtuple
import torch.nn.functional as F

class Actor(torch.nn.Module):
    def __init__(self, D_in, D_hid, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_hid)
        self.linear2 = torch.nn.Linear(D_hid, D_out - 1)
        self.linear3 = torch.nn.Linear(D_hid, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        movepred = self.linear2(h_relu)
        kickpred = torch.nn.Sigmoid()(self.linear3(h_relu))
        return movepred, kickpred

Dims = namedtuple("Dims",["input", "hidden", "out"])
# Dimensions: (in, hidden, out)
DIMS = Dims(12, 50, 10)

class Critic(torch.nn.Module):
    def __init__(self, D_in, D_hid, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_hid)
        self.linear2 = torch.nn.Linear(D_hid, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        winprob = torch.nn.Sigmoid()(self.linear2(h_relu))
        return winprob

class Policy(torch.nn.Module):
<<<<<<< HEAD
    def __init__(self, D_in = 12, D_hid = 50, D_out = 10):
=======
    def __init__(self, D_in, D_hid, D_out):
>>>>>>> 76bd3a06e92105a071958dba6ba95c2f82c3b242
        super(Policy, self).__init__()
        self.affine1 = torch.nn.Linear(D_in, D_hid)
        self.move_head = torch.nn.Linear(D_hid, 9)
        self.kick_head = torch.nn.Linear(D_hid, 1)
        self.value_head = torch.nn.Linear(D_hid, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        moveprobs = F.softmax(self.move_head(x), dim=-1)
        kickprob = torch.nn.Sigmoid()(self.kick_head(x))
        winprob = torch.nn.Sigmoid()(self.value_head(x))
        return moveprobs, kickprob, winprob
