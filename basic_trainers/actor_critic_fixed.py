import numpy as np
import torch
from torch.distributions import Categorical
import utils
import random
import queue

# Class design to run the games, and play out batches of them
class Game:
    # Takes two separate models and only trains one of them which is set to be the red player
    def __init__(self, model_training, model_fixed, env, batch_size, gamma, is_norming):
        self.model_training = model_training
        self.model_fixed = model_fixed
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.is_norming = is_norming
        # the action lists
        self.r_action_list = []
        # the values
        self.r_values = []
        # the running rewards (will be 0 usually, aside from list frame)
        self.r_rewards = []
        # Entropy values
        self.r_entropy = []

        # States if current game is done
        self.done = False

        self.state = env.reset()

    def getAction(self, state, is_red):
        # Gets the action based off the state, updates the value and
        # action lists accordingly.
        if is_red:
            move_probs, kick_prob, value = self.model_training( torch.FloatTensor(state.posToNp("red", 0, self.is_norming) ))
        else:
            move_probs, kick_prob, value = self.model_fixed( torch.FloatTensor(state.posToNp("blue", 0, self.is_norming) ))
        move_dist = Categorical(move_probs)
        kick_dist = Categorical(torch.cat((kick_prob, 1 - kick_prob)))
        action = (move_dist.sample(),  kick_dist.sample())
        if is_red:
            # Appends critic value for the state
            self.r_values.append(value)
            # Appends log prob of taking that action
            self.r_action_list.append( - move_dist.log_prob(action[0]) \
                                       - kick_dist.log_prob(action[1]) )
            # Appends entropy of that action
            self.r_entropy.append( move_dist.entropy() + kick_dist.entropy() )
        else:
            action = utils.reverseAction(action)

        return action

    def runSession(self):
        # Runs a session, that is gets the data up to a point.

        # Makes sure everything is in order.
        assert len(self.r_action_list) == len(self.r_values)

        for t in range(self.batch_size):
            # Gets actions
            r_act = self.getAction(self.state, True)
            b_act = self.getAction(self.state, False)
            # Completes a step
            self.state, rewards, self.done, _ = self.env.step(r_act, b_act)

            # Updates rewards
            self.r_rewards.append(rewards[0])
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
            r_run = self.makeDecayingReward(self.r_rewards[:-1], r_final)
            # Appends the accurate, final rewards.
            r_run.append(r_final)
            return r_run
        else:
            # Otherwise, bootstraps using the predicted value of last frame.
            r_final = torch.Tensor.detach(self.r_values[-1])
            r_run = self.makeDecayingReward(self.r_rewards[:-1], r_final)
            return r_run

    def collectData(self):
        # gives the data for the current session
        r_run = self.getRunningRewards()
        if self.done:
            return (self.r_action_list, self.r_values, self.r_entropy, r_run)
        else:
            # If not done, omit the last elements.
            return (self.r_action_list[:-1], self.r_values[:-1], self.r_entropy[:-1], r_run)

    def cleanData(self):
        # Cleans the data, getting ready for a new session of collection.
        if self.done:
            # Delets all rewards, actions, and values.
            del self.r_action_list[:]
            del self.r_values[:]
            del self.r_rewards[:]
            del self.r_entropy[:]
            # Resets the enviroment.
            self.state = self.env.reset()
            self.done = False
        else:
            # Otherwise, keep the last element to work off of.
            del self.r_action_list[:-1]
            del self.r_values[:-1]
            del self.r_rewards[:-1]
            del self.r_entropy[:-1]

class TrainSession:
    def __init__(self, model_training, model_fixed, env, worker_number, batch_size, learning_rate, gamma, entropy_rate, is_norming):
        self.model_training = model_training
        self.mode_fixed = model_fixed
        self.env = env
        self.entropy_rate = entropy_rate
        self.batch_size = batch_size
        self.lr = learning_rate
        self.gamma = gamma
        self.workers = [Game(model_training, model_fixed, env(), batch_size, gamma, is_norming) for k in range(worker_number)]
        self.opt = torch.optim.Adam(self.model_training.parameters(), lr = self.lr )

        self.seb_rolling_score_queue = queue.Queue()
        self.seb_last_score = [0, 0, 0]
        self.seb_rolling_score = [0, 0, 0]
        self.seb_total_score = [0, 0, 0]

    def getData(self):
        for w in self.workers:
            w.runSession()

    def cleanWorkers(self):
        for w in self.workers:
            w.cleanData()

    def trainFromData(self, actions, values, entropy, rewards):
        advantage = [rewards[i] - torch.Tensor.detach(values[i]) for i in range(len(rewards))]
        losses = [actions[i] * advantage[i] for i in range(len(actions))]
        loss = torch.stack(losses).sum() \
             + torch.nn.MSELoss(reduction = "sum")(torch.FloatTensor(values) , torch.FloatTensor( rewards) ) \
             - (self.entropy_rate * torch.stack(entropy).sum())
        loss.backward()

    def trainFromWorkerData(self, worker):
        r_data = worker.collectData()
        self.trainFromData(*r_data)

    def printInfo(self):
        self.seb_last_score = [0, 0, 0]

        for w in self.workers:
            if w.done:
                goals = w.env.goalScored()

                self.seb_rolling_score[(goals + 2) % 3] += 1
                self.seb_rolling_score_queue.put((goals + 2) % 3)
                if self.seb_rolling_score_queue.qsize() > 100:
                    self.seb_rolling_score[self.seb_rolling_score_queue.get()] -= 1

                self.seb_last_score[(goals + 2) % 3] += 1
                self.seb_total_score[(goals + 2) % 3] += 1

        print(f"Last L-F-T: {self.seb_last_score[0]}-{self.seb_last_score[1]}-{self.seb_last_score[2]}")
        print("Running L-F-T: {}-{}-{}".format(self.seb_rolling_score[0], self.seb_rolling_score[1], self.seb_rolling_score[2]) )
        if sum(self.seb_total_score) != 0:
            print("Overall L-F-T percentage: {:.3f}% - {:.3f}% - {:.3f}%, # of games = {}".format(self.seb_total_score[0]*100/sum(self.seb_total_score), self.seb_total_score[1]*100/sum(self.seb_total_score), self.seb_total_score[2]*100/sum(self.seb_total_score), sum(self.seb_total_score)))
        else:
            print("# of games = {}".format(sum(self.seb_total_score)))
        print("\n", end="")

    def runStep(self):
        self.getData()
        self.opt.zero_grad()
        self.printInfo()
        for w in self.workers:
            self.trainFromWorkerData(w)
        self.opt.step()
        self.cleanWorkers()
