import torch
import torch.nn.functional as F
import numpy as np
import random
from data_handler import datahandler
from game_simulator import gameparams as gp
from torch.autograd import Variable
positions = np.load('pugamedata.npy')
print(positions)
print("Divinging...")
normalizedpositions = ((positions-gp.mean)/gp.stdev).tolist()
print("divinging done")
actions = np.load('pumovedata.npy').tolist()

print("Zipping...")
c = list(zip(normalizedpositions, actions))

print("Pre shuffle")
random.shuffle(c)
print("Shuffle done")

normalizedpositions , actions = list(zip(*c))

data = [normalizedpositions, actions]

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
runningloss = 0
def optimize():
    for t in range(3):
    # Forward pass: Compute predicted y by passing x to the model

        data_tensor = torch.FloatTensor(data[0]).view(-1,1,12)
        movepred, kickpred = model( data_tensor )
        actiondata = list(map(list, zip(*data[1])))
        true_move = torch.FloatTensor(actiondata[0]).view(-1,1)
        true_kick = torch.FloatTensor(actiondata[1]).view(-1,1)
        print(true_move)
        print(movepred)
    # Compute and print loss
        loss = movecriterion(movepred , true_move)
        loss = kickcriterion( kickpred , true_kick)
        print("Loss for iteration " + str(t) + ":")
        print(loss / len(data[0]) )
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(model.state_dict(), "initialmodelweights.dat")
optimize()
model = TwoLayerNet(D_in, H, D_out)
model.load_state_dict(torch.load("initialmodelweights.dat"))
model.eval()
print(model(torch.randn(1,1,12)))
