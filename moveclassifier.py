    #! /usr/bin/python

import torch
import torch.nn.functional as F
import numpy as np
import random
from data_handler import datahandler
from game_simulator import gameparams as gp
from torch.autograd import Variable
import math
from collections import namedtuple

positions = np.load('pugamedata.npy')
normalizedpositions = ((positions-gp.mean)/gp.stdev).tolist()
actions = np.load('pumovedata.npy').tolist()

# Shuffle normalizedpositions and actions in the same way
c = list(zip(normalizedpositions, actions))
random.shuffle(c)
normalizedpositions, actions = list(zip(*c))

print("Data normalised")

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_hid, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_hid)
        self.linear2 = torch.nn.Linear(D_hid, D_out - 1)
        self.linear3 = torch.nn.Linear(D_hid, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        res = self.linear2(h_relu)
        movepred = self.linear2(h_relu)
        kickpred = torch.nn.Sigmoid()(self.linear3(h_relu))
        return movepred, kickpred

# Batch size
N = 32
Dims = namedtuple("Dims",["input", "hidden", "out"])
# Dimensions: (in, hidden, out)
DIMS = Dims(12, 50, 10)

# Create random Tensors to hold inputs and outputs

# Construct our model by instantiating the class defined above
model = TwoLayerNet(*DIMS)

movecriterion = torch.nn.CrossEntropyLoss(reduction='mean')
kickcriterion = torch.nn.BCELoss(size_average=True)
optimiser = torch.optim.Adam(model.parameters(), lr=.001)
data_tensor = torch.FloatTensor(normalizedpositions).view(-1,N,DIMS.input)
actiondata = list(map(list, zip(*actions)))
true_move = torch.tensor(actiondata[0]).view(-1,N)
true_kick = torch.FloatTensor(actiondata[1]).view(-1,N)

for t in range(3):
    runningloss = 0
    for i in range(math.floor(len(normalizedpositions)*9/(10*N))):
        # Forward pass: Compute predicted y by passing x to the model
        movepred, kickpred = model( data_tensor[i] )
        # Compute and print loss
        loss = movecriterion(movepred , true_move[i])
        loss += kickcriterion( kickpred , true_kick[i])
        runningloss += loss
        if i % 100 == 0:
            print("Loss for iteration " + str(t)+","+str(i*N)+"/"+str(math.floor(len(normalizedpositions)*9/10)) + ":")
            print(runningloss/(100))
            runningloss = 0
        # Zero gradients, perform a backward pass, and update the weights.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    torch.save(model.state_dict(), "initialmodelweights.dat")

model = TwoLayerNet(*DIMS)
model.load_state_dict(torch.load("initialmodelweights.dat"))
model.eval()

with torch.no_grad():
    runningloss = 0
    j = 0
    for i in range(math.floor(len(normalizedpositions)*9/(10*N)), math.floor(len(normalizedpositions)/N)):
        movepred, kickpred = model( data_tensor[i] )
        j += 1
    # Compute and print loss
        loss = movecriterion(movepred , true_move[i])
        loss += kickcriterion( kickpred , true_kick[i])
        runningloss += loss
    print("validation loss: " + str(runningloss / j))
