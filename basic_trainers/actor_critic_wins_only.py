import numpy as np
import torch
from torch.distributions import Categorical
import utils
import random
import queue

# Class design to run the games, and play out full games of them
class Game:
    def __init__(self, red_model, blue_model, env, gamma, is_norming):
        self.red_model = red_model
        self.blue_model = blue_model
        self.env = env
        self.gamma = gamma
        self.is_norming = is_norming
        # the action lists
        self.r_action_list = []
        self.b_action_list = []
        # the values
        self.r_values = []
        self.b_values = []
        # the running rewards (will be 0 usually, aside from list frame)
        self.r_rewards = []
        self.b_rewards = []
        # Entropy values
        self.r_entropy = []
        self.b_entropy = []

        # States if current game is done
        self.done = False

        self.state = env.reset()

    def getAction(self, state, is_red):
        # Gets the action based off the state, updates the value and
        # action lists accordingly.
        if is_red:
            move_probs, kick_prob, value = self.red_model(torch.FloatTensor(state.posToNp("red", 0, self.is_norming)))
        else:
            move_probs, kick_prob, value = self.blue_model(torch.FloatTensor(state.posToNp("blue", 0, self.is_norming)))
        move_dist = Categorical(move_probs)
        kick_dist = Categorical(torch.cat((kick_prob, 1 - kick_prob)))
        action = (move_dist.sample(),  kick_dist.sample())
        if is_red:
            # Appends critic value for the state
            self.r_values.append(value)
            # Appends log prob of taking that action
            self.r_action_list.append( - move_dist.log_prob(action[0]) \
                                       - kick_dist.log_prob(action[1]))
            # Appends entropy of that action
            self.r_entropy.append( move_dist.entropy() + kick_dist.entropy() )
        else:
            # Does same for blue.
            self.b_values.append(value)
            self.b_action_list.append( - move_dist.log_prob(action[0]) \
                                       - kick_dist.log_prob(action[1]))
            self.b_entropy.append(move_dist.entropy() + kick_dist.entropy())
            action = utils.reverseAction(action)

        return action



    def runSession(self):
        # Runs a session, that is gets the data up to a point.

        # Makes sure everything is in order.
        assert len(self.r_action_list) == len(self.b_action_list)
        assert len(self.b_values) == len(self.b_action_list)
        assert len(self.b_values) == len(self.r_values)

        for t in range(self.env.max_steps + 1):
            # Gets actions
            r_act = self.getAction(self.state, True)
            b_act = self.getAction(self.state, False)
            # Completes a step
            self.state , rewards, self.done, _ = self.env.step(r_act, b_act)
            # Updates rewards
            self.r_rewards.append(rewards[0])
            self.b_rewards.append(rewards[1])
            # Breaks if done
            if self.done:
                break

        # The game should be guaranteed to end by this point
        assert(self.done)

    def makeDecayingReward(self, l, x):
        rew = [x]
        for R in l[::-1]:
            rew.insert(0, R + (rew[0] * self.gamma) )
        # Does not include final value, as this is usually a prediction
        # So is not accurate.
        return rew[:-1]

    def getRunningRewards(self):
        # The game should be guaranteed to end by this point
        assert(self.done)

        r_final = self.r_rewards[-1]
        b_final = self.b_rewards[-1]
        r_run = self.makeDecayingReward(self.r_rewards[:-1], r_final)
        b_run = self.makeDecayingReward(self.b_rewards[:-1], b_final)
        # Appends the accurate, final rewards.
        r_run.append(r_final)
        b_run.append(b_final)
        return (r_run, b_run)

    def collectData(self):
        # The game should be guaranteed to end by this point
        assert(self.done)

        # gives the data for the current session
        r_run, b_run = self.getRunningRewards()
        if self.done:
            return ((self.r_action_list, self.r_values, self.r_entropy, r_run) , \
                    (self.b_action_list, self.b_values, self.b_entropy, b_run))

    def cleanData(self):
        # Cleans the data, getting ready for a new session of collection.
        if self.done:
            # Delets all rewards, actions, and values.
            del self.r_action_list[:]
            del self.r_values[:]
            del self.r_rewards[:]
            del self.b_action_list[:]
            del self.b_values[:]
            del self.b_rewards[:]
            del self.r_entropy[:]
            del self.b_entropy[:]
            # Resets the enviroment.
            self.state = self.env.reset()
            self.done = False
        else:
            # Otherwise, keep the last element to work off of.
            del self.r_action_list[:-1]
            del self.r_values[:-1]
            del self.r_rewards[:-1]
            del self.b_action_list[:-1]
            del self.b_values[:-1]
            del self.b_rewards[:-1]
            del self.r_entropy[:-1]
            del self.b_entropy[:-1]

class TrainSession:
    def __init__(self, model_red, model_blue, env, worker_number, learning_rate, gamma, entropy_rate, is_norming, print_debug = False):
        self.model_red = model_red
        self.model_blue = model_blue
        self.env = env
        self.entropy_rate = entropy_rate
        self.lr = learning_rate
        self.gamma = gamma
        self.workers = [Game(model_red, model_blue, env(), gamma, is_norming) for k in range(worker_number)]
        self.opt_red = torch.optim.Adam(self.model_red.parameters(), lr = self.lr )
        self.opt_blue = torch.optim.Adam(self.model_blue.parameters(), lr = self.lr )
        self.critic_loss_bias = 10

        self.seb_rolling_score_queue = queue.Queue()
        self.seb_last_score = [0, 0, 0]
        self.seb_rolling_score = [0, 0, 0]
        self.seb_total_score = [0, 0, 0]

        self.print_debug = print_debug

    def getData(self):
        for w in self.workers:
            w.runSession()

    def cleanWorkers(self):
        for w in self.workers:
            w.cleanData()

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

    def trainFromData(self, actions, values, entropy, rewards):
        advantage = [rewards[i] - torch.Tensor.detach(values[i]) for i in range(len(rewards))]
        losses = [actions[i] * advantage[i] for i in range(len(actions))]
        value_tensor = torch.stack(values).reshape(-1)
        reward_tensor = torch.FloatTensor(rewards).reshape(-1)
        value_loss = torch.nn.MSELoss(reduction = "sum")( value_tensor , reward_tensor)
        policy_loss = torch.stack(losses).sum()
        loss = policy_loss \
             + value_loss \
             - (self.entropy_rate * torch.stack(entropy).sum())

        loss.backward()


    def trainFromWorkerData(self, worker):
        r_data, b_data = worker.collectData()
        # Only train if the agent won, i.e. if the reward at the final frame is positive
        if self.print_debug:
            if r_data[3][-1] > 0:
                print("Red", end="")
                self.trainFromData(*r_data)
            if b_data[3][-1] > 0:
                print("Blue", end="")
                self.trainFromData(*b_data)
            print("")

    def runStep(self):
        self.getData()
        self.opt_red.zero_grad()
        self.opt_blue.zero_grad()
        if self.print_debug:
            self.printInfo()
        for w in self.workers:
            self.trainFromWorkerData(w)
        self.opt_red.step()
        self.opt_blue.step()
        self.cleanWorkers()
