import torch
import torch.nn.functional as F
import numpy as np
from data_handler import datahandler
from game_simulator import gameparams as gp
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
        redpositions.append(torch.FloatTensor(flatten(frame[0])))
        redonehotaction = [0]*10
        redonehotaction[frame[1][0][1]] = 1
        redonehotaction[9]=frame[1][0][0]
        redactions.append(torch.FloatTensor(redonehotaction))
        blueframe = []
        for object in frame:
            blueframe.append([gp.rotatePos(object[0]),gp.rotateVel(object[1])])
        bluepositions.append(torch.FloatTensor(flatten(blueframe)))
        blueonehotaction = [0]*10
        blueonehotaction[frame[1][1][1]] = 1
        blueonehotaction[9]=frame[1][1][0]
        blueactions.append(torch.FloatTensor(blueonehotaction))


positions = redpositions + bluepositions
actions = redactions + blueactions
data = [positions, actions]
print(data)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = torch.cat([F.softmax(self.linear2(h_relu)[:9],dim=0),torch.sigmoid(self.linear2(h_relu)[-1])], dim=0)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 12, 100, 10

# Create random Tensors to hold inputs and outputs

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for t in range(1):
    for i in range(len(data[0])):
        # Forward pass: Compute predicted y by passing x to the model
        print(data)
        y_pred = model(data[0][i])

        # Compute and print loss
        loss = criterion(y_pred, data[1])
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
