import numpy as np
import torch
from torch.distributions import Categoricals
import utils


# Class design to run the games, and play out batches of them
class Game:
    def __init__(self, model, env, batch_size, gamma):
        self.model = model
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.is_norming = True
        # the action lists
        self.r_action_list = []
        self.b_action_list = []
        # the values
        self.r_values = []
        self.b_values = []
        # the running rewards (will be 0 usually, aside from list frame)
        self.r_rewards = []
        self.b_rewards = []
        # States if current game is done
        self.done = False

        self.state = env.reset()

    def getAction(self,state, is_red):
        # Gets the action based off the state, updates the value and
        # action lists accordingly.
        if is_red:
            move_probs, kick_prob, value = self.model( state.posToNp("red", 0, self.is_norming) )
        else:
            move_probs, kick_prob, value = self.model( state.posToNp("blue", 0, self.is_norming) )
        move_dist = Categorical(move_probs)
        kick_dist = Categorical(torch.cat((kick_prob, 1 - kick_prob)))
        action = (move_dist.sample(),  kick_dist.sample())
        if is_red:
            self.r_values.append(value)
            self.r_action_list.append( - move_dist.log_prob(action[0]) \
                                       - kick_dist.log_prob(action[1]) )
        else:
            self.b_values.append(value)
            self.b_action_list.append( - move_dist.log_prob(action[0]) \
                                       - kick_dist.log_prob(action[1]) )
            action = utils.reverseAction(action)
        return action



    def runSession(self):
        # Runs a session, that is gets the data up to a point.

        # Makes sure everything is in order.
        assert len(self.r_action_list) == len(self.b_action_list)
        assert len(self.b_values) == len(self.b_action_list)
        assert len(self.b_values) == len(self.r_values)

        for t in range(self.batch_size):
            # Gets actions
            r_act = self.getAction(self, state, True)
            b_act = self.getAction(self, state, False)
            # Completes a step
            self.state , rewards, self.done, _ = env.step(r_act, b_act)
            # Updates rewards
            self.r_rewards.append(rewards[0])
            self.b_rewards.append(rewards[1])
            # Breaks if done
            if self.done:
                break

    def makeDecayingReward(self, l, x):
        rew = [x]
        for R in l[::-1]:
            rew.insert(0, R + (rew[0] * self.gamma) )
        # Does not include final value, as this is usually a prediction
        # So is not accurate.
        return rew[:-1]

    def getRunningRewards(self):
        # Gets the running reward for the session,
        # bootstrapping if needed.
        if self.done:
            # if game is done, not bootstrapping is needed.
            r_final = self.r_rewards[-1]
            b_final = self.b_rewards[-1]
            r_run = self.makeDecayingReward(self.r_rewards[:-1], r_final)
            b_run = self.makeDecayingReward(self.b_rewards[:-1], b_final)
            # Appends the accurate, final rewards.
            r_run.append(r_final)
            b_run.append(b_final)
            return (r_run, b_run)
        else:
            # Otherwise, bootstraps using the predicted value of last frame.
            r_run = self.makeDecayingReward(self.r_rewards[:-1], self.r_values[-1])
            b_run = self.makeDecayingReward(self.b_rewards[:-1], self.b_values[-1])
            return (r_run, b_run)

    def collectData(self):
        # gives the data for the current session
        r_run, b_run = self.getRunningRewards()
        if self.done:
            return ((r_action_list, r_values, r_run) , (b_action_list, b_values, b_run))
        else:
            # If not done, omit the last elements.
            ((r_action_list[:-1], r_values[:-1], r_run) , (b_action_list[:-1], b_values[:-1], b_run))

    def cleanData(self):
        # Cleans the data, getting ready for a new session of collection.
        if self.done:
            # Delets all rewards, actions, and values.
            del r_action_list[:]
            del r_values[:]
            del r_rewards[:]
            del b_action_list[:]
            del b_values[:]
            del b_rewards[:]
            # Resets the enviroment.
            self.state = env.reset()
            self.done = False
        else:
            # Otherwise, keep the last element to work off of.
            del r_action_list[:-1]
            del r_values[:-1]
            del r_rewards[:-1]
            del b_action_list[:-1]
            del b_values[:-1]
            del b_rewards[:-1]
