#! /usr/bin/python

import torch
import torch.nn.functional as F
import numpy as np
import random
from game_simulator import gameparams as gp
from network import Policy
from torch.autograd import Variable
import math
from gym_haxball.onevoneenviroment import DuelEnviroment
from torch.distributions import Categorical
import utils
import torch.nn.functional as F
from collections import namedtuple


#data_tensor should be of the form torch.FloatTensor([[loserframes],[winnerframes]])
#each loserframe, winnerframe should be flattened and [loserframes], [winnerframes]
#action_data should be of the form [[[losermoves],[loserkicks]],[[winnermoves],[winnerkicks]]]
def cuttosize(x,batch):
    if len(x) % batch == 0:
        return x
    else:
        return x[:-(len(x) % batch)]


def learnFromPlayedGames(model, data_tensor, action_data, epochs, learning_rate, batch_size):

    movecriterion = torch.nn.CrossEntropyLoss(reduction='mean')
    kickcriterion = torch.nn.BCELoss(size_average=True)
    wincriterion = torch.nn.BCELoss(size_average=True)
    loser_moves = cuttosize(action_data[0][0], batch_size)
    loser_kicks = cuttosize(action_data[0][1], batch_size)
    winner_moves = cuttosize(action_data[1][0], batch_size)
    winner_kicks = cuttosize(action_data[1][1], batch_size)
    true_move =[torch.LongTensor(loser_moves).view(-1,batch_size), torch.LongTensor(winner_moves).view(-1,batch_size)]
    true_kick = [torch.FloatTensor(loser_kicks).view(-1,batch_size), torch.FloatTensor(winner_kicks).view(-1,batch_size)]
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        runningloss = 0
        # Train

        for i in range((len(loser_moves) * 9) // (batch_size*10)):
            for k in range(2):
                # Forward pass: Compute predicted y by passing x to the model

                moveprob, kickprob, winprob = model(data_tensor[k][batch_size * i : batch_size * (i + 1)])
                # Compute and print loss

                loss = movecriterion(moveprob , true_move[k][i]) \
                     + kickcriterion(kickprob, true_kick[k][i]) \
                     #+ wincriterion(winprob, torch.FloatTensor(np.repeat(k,batch_size)))
                runningloss += loss
                # Zero gradients, perform a backward pass, and update the weights.

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            if i % 100 == 99:
                print(f"Loss for iteration {t:02}, {i * batch_size:06}/{len(winner_moves) * 9 // 10:06}: {float(runningloss) / (batch_size *100):.5f}")
                runningloss = 0
        #Validate
        with torch.no_grad():
            runningloss = 0
            j = 0
            for i in range((len(loser_moves) * 9) // (batch_size*10), len(loser_moves) // batch_size):
                for k in range(2):
                    # Forward pass: Compute predicted y by passing x to the model
                    moveprob, kickprob, winprob = model(data_tensor[k][batch_size * i : batch_size * (i + 1)])
                    # Compute and print loss
                    loss = movecriterion(moveprob, torch.LongTensor(true_move[k][i]))
                    loss += kickcriterion(kickprob, true_kick[k][i])
                    loss += wincriterion(winprob, torch.FloatTensor(np.repeat(k,batch_size)))
                    j += 1
                    runningloss += loss
            print("validation loss: " + str(2 * runningloss / (j*batch_size)))



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

def select_action(model, state, red_action_list, blue_action_list):
    # Fetches the appropriate state
    red_state = state.posToNp("red" , 0)
    blue_state = state.posToNp("blue" , 0)
    # Gets the actions
    red_move_probs, red_kick_prob, red_state_value = model( torch.FloatTensor(red_state) )
    blue_move_probs, blue_kick_prob, blue_state_value = model( torch.FloatTensor(blue_state) )
    # Make distributions for kicking and moves
    blue_m = Categorical(blue_move_probs)
    red_m = Categorical(red_move_probs)
    blue_k = Categorical(torch.cat((blue_kick_prob, 1 - blue_kick_prob)))
    red_k = Categorical(torch.cat((red_kick_prob, 1 - red_kick_prob)))
    # Samples said distributions for actual actions
    red_action = (red_m.sample(), red_k.sample())
    blue_action = (blue_m.sample(), blue_k.sample())
    # Append actions
    red_action_list.append(SavedAction((red_m.log_prob(red_action[0])
                                        , red_k.log_prob(red_action[1])),
                                        red_state_value))
    blue_action_list.append(SavedAction((blue_m.log_prob(blue_action[0])
                                        , blue_k.log_prob(blue_action[1])),
                                        blue_state_value))
    # Finally, reverse action for blue.
    return red_action, utils.reverseAction(blue_action)


def finish_episode(model, red_reward_list, blue_reward_list, red_action_list, blue_action_list, learning_rate = 3e-2, gamma = 1 - 1e-3):
    red_R = 0
    red_policy_losses = []
    red_value_losses = []
    red_returns = []
    blue_R = 0
    blue_policy_losses = []
    blue_value_losses = []
    blue_returns = []
    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate )
    # Updates the deaying rewards
    for r in red_reward_list[::-1]:
        red_R = r + gamma * red_R
        red_returns.insert(0, red_R)
    for r in blue_reward_list[::-1]:
        blue_R = r + gamma * blue_R
        blue_returns.insert(0, blue_R)
    # Makes returns into tensor and normalises
    blue_returns = torch.tensor(blue_returns)
    blue_returns = (blue_returns - blue_returns.mean()) / (blue_returns.std() + eps)
    red_returns = torch.tensor(red_returns)
    red_returns = (red_returns - red_returns.mean()) / (red_returns.std() + eps)

    # Gets losses for red and blue.
    for (log_prob, value), R in zip(red_action_list, red_returns):
        advantage = R - value.item()
        red_policy_losses.append( (-log_prob[0] - log_prob[1]) * advantage)
        red_value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    for (log_prob, value), R in zip(blue_action_list, blue_returns):
        advantage = R - value.item()
        blue_policy_losses.append( (-log_prob[0] - log_prob[1]) * advantage)
        blue_value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimiser.zero_grad()
    loss =   torch.stack(red_policy_losses).sum() + torch.stack(red_value_losses).sum() \
           + torch.stack(blue_policy_losses).sum() + torch.stack(blue_value_losses).sum()
    loss.backward()
    optimiser.step()
    del model.rewards[:]
    del model.saved_actions[:]


def actorCriticTrain(model, step_len, game_limit):
    # Makes enviroment
    env = DuelEnviroment(step_len, game_limit)
    # Initialises lists of rewards and actions

    for i_episode in range(20):

        red_running_reward = 10
        blue_running_reward = 10
        red_reward_list = []
        blue_reward_list = []
        blue_action_list = []
        red_action_list = []
        state = env.reset()
        blue_ep_reward = 0
        red_ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            # Gets an actions, performas a step
            action = select_action(model, state, red_action_list, blue_action_list)
            state, done, red_reward, blue_reward = env.step(action[0], action[1])
            # Appends the rewards got to the reward lists
            red_reward_list.append(red_reward)
            blue_reward_list.append(blue_reward)
            red_ep_reward += red_reward
            blue_ep_reward += blue_reward
            if done:
                break

        red_running_reward = 0.05 * red_ep_reward + (1 - 0.05) * red_running_reward
        blue_running_reward = 0.05 * blue_ep_reward + (1 - 0.05) * blue_running_reward
        finish_episode(model, red_reward_list, blue_reward_list, red_action_list, blue_action_list)
        if i_episode % 1 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, red_ep_reward, red_running_reward))
