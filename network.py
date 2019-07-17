import torch
from collections import namedtuple

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
