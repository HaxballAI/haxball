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
        redpositions.append(torch.FloatTensor(flatten(frame[0])))
        redaction = frame[1][0]
        redactions.append(redaction)
        blueframe = []
        for object in frame:
            blueframe.append([gp.rotatePos(object[0]),gp.rotateVel(object[1])])
        bluepositions.append(torch.FloatTensor(flatten(blueframe)))
        blueaction = frame[1][1]
        blueactions.append(blueaction)


positions = redpositions + bluepositions
actions = redactions + blueactions
data = [positions, actions]

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

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, frame in enumerate(data, 0):
        # get the inputs; data is a list of [inputs, labels]
        positions, actions = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(positions)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')'''




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
