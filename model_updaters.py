
import torch
import torch.nn.functional as F
import numpy as np
import random
from data_handler import datahandler
from game_simulator import gameparams as gp
from network import Policy, DIMS
from torch.autograd import Variable
import math

#code for loading actions and positions, normalizing positions, and shuffling
'''
positions = np.load('pugamedata.npy')
normalizedpositions = ((positions - gp.mean)/gp.stdev).tolist()
actions = np.load('pumovedata.npy').tolist()

# Shuffle normalizedpositions and actions in the same way
c = list(zip(normalizedpositions, actions))
random.shuffle(c)
normalizedpositions, actions = list(zip(*c))
'''

'''
model = TwoLayerNet(*DIMS)
data_tensor = torch.FloatTensor(normalizedpositions).view(-1,N,DIMS.input)
action_data = list(map(list, zip(*actions)))
epochs = 5
learning_rate = .001
batch_size = 32
'''
#data_tensor should be of the form torch.FloatTensor([[loserframes],[winnerframes]])
#each loserframe, winnerframe should be flattened and [loserframes], [winnerframes]
#action_data should be of the form [[[losermoves],[loserkicks]],[[winnermoves],[winnerkicks]]]
def cuttosize(x,batch):
    if len(x) % batch == 0:
        return x
    else:
        return x[:-(len(x) % batch)]


def initialize(model, data_tensor, action_data, epochs, learning_rate, batch_size):
    for t in range(epochs):
        runningloss = 0
        #Train
        movecriterion = torch.nn.CrossEntropyLoss(reduction='mean')
        kickcriterion = torch.nn.BCELoss(size_average=True)
        wincriterion = torch.nn.BCELoss(size_average=True)
        true_move = [torch.tensor(cuttosize(action_data[0][0])).view(-1,batch_size), torch.tensor(cuttosize(action_data[0][0])).view(-1,batch_size)]
        true_kick = [torch.FloatTensor(cuttosize(action_data[0][1])).view(-1,batch_size), torch.FloatTensor(cuttosize(action_data[0][1])).view(-1,batch_size)]
        for i in range(32*((len(data_tensor) * 9) // 320)):

            for k in range(2):
                # Forward pass: Compute predicted y by passing x to the model
                moveprob, kickprob, winprob = model(data_tensor[k][i])
                # Compute and print loss
                loss = movecriterion(moveprob, true_move[k][i])
                loss += kickcriterion(kickprob, true_kick[k][i])
                loss += wincriterion(winprob, torch.FloatTensor(np.repeat(k,batch_size)))
                runningloss += loss
                if i % 100 == 0:
                    print(f"Loss for iteration {t:02}, {i * N:06}/{len(normalizedpositions) * 9 // 10:06}: {float(runningloss) / 100:.5f}")
                    runningloss = 0
                # Zero gradients, perform a backward pass, and update the weights.
                optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        #Validate
        with torch.no_grad():
            runningloss = 0
            j = 0
            for i in range(32*(len(data_tensor) * 9 // 320) + 1, len(data_tensor)):
                for k in range(2):
                    # Forward pass: Compute predicted y by passing x to the model
                    moveprob, kickprob, winprob = model(data_tensor[k][i])
                    # Compute and print loss
                    loss = movecriterion(moveprob, true_move[k][i])
                    loss += kickcriterion(kickprob, true_kick[k][i])
                    loss += wincriterion(winprob, torch.FloatTensor(np.repeat(k,batch_size)))
                    runningloss += loss
            print("validation loss: " + str(runningloss / j))
'''
def selfplayupdate(model, data_tensor, action_data, epochs, learning_rate, batch_size):
    for t in range(epochs):
        runningloss = 0
        #Train
        for i in range(len(data_tensor)):
            for k in range(2):
                # Forward pass: Compute predicted y by passing x to the model
                moveprob, kickprob, winprob = model(data_tensor[k][i])
                # Compute and print loss
                movecriterion = torch.nn.CrossEntropyLoss(reduction='mean')
                kickcriterion = torch.nn.BCELoss(size_average=True)
                wincriterion = torch.nn.BCELoss(size_average=True)
                true_move = torch.tensor(action_data[k][0]).view(-1,batch_size)
                true_kick = torch.FloatTensor(action_data[k][1]).view(-1,batch_size)
                loss = movecriterion(moveprob, true_move[i])
                loss += kickcriterion(kickprob, true_kick[i])
                loss += wincriterion(winprob, torch.FloatTensor(np.repeat(k,batch_size)))
                runningloss += loss
                if i % 100 == 0:
                    print(f"Loss for iteration {t:02}, {i * N:06}/{len(normalizedpositions) * 9 // 10:06}: {float(runningloss) / 100:.5f}")
                    runningloss = 0
                # Zero gradients, perform a backward pass, and update the weights.
                optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]'''
