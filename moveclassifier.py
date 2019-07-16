#! /usr/bin/python

import torch
import torch.nn.functional as F
import numpy as np
import random
from data_handler import datahandler
from game_simulator import gameparams as gp
from torch.autograd import Variable
import math

positions = np.load('pugamedata.npy')
print(positions)
normalizedpositions = ((positions-gp.mean)/gp.stdev).tolist()
actions = np.load('pumovedata.npy').tolist()

# Shuffle normalizedpositions and actions in the same way
c = list(zip(normalizedpositions, actions))
random.shuffle(c)
normalizedpositions, actions = list(zip(*c))

print("Data normalised")

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out-1)
        self.linear3 = torch.nn.Linear(H, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        res = self.linear2(h_relu)
        movepred = self.linear2(h_relu)
        kickpred = torch.nn.Sigmoid()(self.linear3(h_relu))
        return movepred, kickpred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 12, 100, 10

# Create random Tensors to hold inputs and outputs

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

movecriterion = torch.nn.CrossEntropyLoss(reduction='sum')
kickcriterion = torch.nn.BCELoss(size_average=True)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
data_tensor = torch.FloatTensor(normalizedpositions).view(-1,32,12)
actiondata = list(map(list, zip(*actions)))
true_move = torch.tensor(actiondata[0]).view(-1,32)
true_kick = torch.FloatTensor(actiondata[1]).view(-1,32)

for t in range(3):
    runningloss = 0
    for i in range(math.floor(len(normalizedpositions)*9/320)):
        # Forward pass: Compute predicted y by passing x to the model
        movepred, kickpred = model( data_tensor[i] )
        # Compute and print loss
        loss = movecriterion(movepred , true_move[i])
        loss += kickcriterion( kickpred , true_kick[i])
        runningloss += loss
        if i % 100 == 0:
            print("Loss for iteration " + str(t)+","+str(i*32)+"/"+str(math.floor(len(normalizedpositions)*9/10)) + ":")
            print(runningloss/3200)
            runningloss = 0
        # Zero gradients, perform a backward pass, and update the weights.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    torch.save(model.state_dict(), "initialmodelweights.dat")

model = TwoLayerNet(D_in, H, D_out)
model.load_state_dict(torch.load("initialmodelweights.dat"))
model.eval()

