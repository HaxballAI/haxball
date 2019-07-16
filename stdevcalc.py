import torch
import torch.nn.functional as F
import numpy as np
from data_handler import datahandler
from game_simulator import gameparams as gp
from torch.autograd import Variable

data_handler = datahandler.DataHandler("rawsaved_games.dat")
data_handler.loadFileToBuffer()
loaded_data=data_handler.buffer

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    if isinstance(S[0], type(np.array([])) ):
        return flatten(S[0].tolist()) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

redpositions = []
bluepositions = []
redactions = []
blueactions = []
for game in loaded_data:
    for frame in game:
        redpositions.append(flatten(frame[0]))
        redaction = frame[1][0]
        redactions.append(redaction)
        blueframe = []
        for object in frame[0]:
            blueframe.append([gp.rotatePos(object[0]), gp.rotateVel(object[1])])
        blueframe = flatten(blueframe)
        blueframe = blueframe[4:8] + blueframe[0:4] + blueframe[8:]
        bluepositions.append(blueframe)
        blueaction = frame[1][1]
        blueactions.append(blueaction)

positions = redpositions + bluepositions
normalizedpositions = [[x[0]/x[1] for x in zip(position, gp.normalizers)] for position in positions]
print(normalizedpositions[0])
actions = redactions + blueactions
data = [positions, actions]
positions = np.array(positions)
print("Red Positions:")
print(redpositions[10000])
print("Blue Positions:")
<<<<<<< HEAD
print(bluepositions[10000])
print(np.std(positions,axis=0))
print(np.mean(positions,axis=0))
=======
print(bluepositions[0])
print(np.std(positions, axis = 0))
>>>>>>> 5680bd16684a7f8cedabec1af90fe1d799e2681a

'''
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        movepred = self.linear2(h_relu)[0][:9]
        kickpred = torch.sigmoid(self.linear2(h_relu)[0][-1:])
        return movepred, kickpred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 12, 100, 10

# Create random Tensors to hold inputs and outputs

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)



movecriterion = torch.nn.CrossEntropyLoss(reduction='sum')
kickcriterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for t in range(1):
    for i in range(len(data[0])):
        # Forward pass: Compute predicted y by passing x to the model

        movepred, kickpred = model(torch.tensor(data[0][i]).unsqueeze(0))

        # Compute and print loss

        loss = movecriterion(movepred.unsqueeze(0),torch.tensor([data[1][i][0]]))
        loss += kickcriterion( kickpred.unsqueeze(0), torch.FloatTensor( [ data[1][i][1] ] ) )

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''
