from basic_trainers.actor_critic_wins_only import TrainSession as WinTrainSession
from basic_trainers.actor_critic_fixed import TrainSession as FixedTrainSession
from game_simulator import gamesim
import gym_haxball.onevoneenviroment
import utils
from utils import global_timer

from torch.distributions import Categorical
import numpy as np
import nashpy as nash
import torch

import copy


class GameTester:
    def __init__(self, env):
        self.env = env
        self.done = False
        self.state = self.env.reset()
        self.is_norming = False

    def getAction(self, model, is_red):
        # Gets the action based off the state, updates the value and
        # action lists accordingly.
        if is_red:
            move_probs, kick_prob, value = model(torch.FloatTensor(self.state.posToNp("red", 0, self.is_norming)))
        else:
            move_probs, kick_prob, value = model(torch.FloatTensor(self.state.posToNp("blue", 0, self.is_norming)))
        move_dist = Categorical(move_probs)
        kick_dist = Categorical(torch.cat((kick_prob, 1 - kick_prob)))
        action = (move_dist.sample(),  kick_dist.sample())

        if not is_red:
            action = utils.reverseAction(action)

        return action

    def getAverageReward(self, red_model, blue_model, number_of_games = 1):
        red_reward, blue_reward = 0, 0
        for g in range(number_of_games):
            while True:
                # Gets actions
                red_action = self.getAction(red_model, True)
                blue_action = self.getAction(blue_model, False)
                # Completes a step
                self.state , rewards, self.done, _ = self.env.step(red_action, blue_action)
                # Updates rewards
                red_reward += rewards[0]
                blue_reward += rewards[1]
                # Breaks if done
                if self.done:
                    self.done = 0
                    self.state = self.env.reset()
                    break
        return red_reward / number_of_games, blue_reward / number_of_games


class PopulationHandler:
    def __init__(self, initial_agents, env, batch_size, learning_rate, gamma, entropy_rate, is_norming):
        self.agents = initial_agents
        self.number_of_agents = len(self.agents)

        self.env = env
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_rate = entropy_rate
        self.is_norming = is_norming

        # Initialise the champion agent, take it as the first agent for now
        self.champion = copy.deepcopy(initial_agents[0])

        self.nash_distribution = torch.FloatTensor([0 for i in range(self.number_of_agents)])
        self.nash_categorical = Categorical(self.nash_distribution)

        self.game_tester = GameTester(env())

        self.reference_model = torch.load("models/classifier.model")

    def updateNashEquilibrium(self, number_of_games):
        reward_matrix = np.zeros((self.number_of_agents, self.number_of_agents))

        for i in range(self.number_of_agents):
            for j in range(i + 1, self.number_of_agents):
                average_reward, _ = self.game_tester.getAverageReward(self.agents[i], self.agents[j], number_of_games)
                reward_matrix[i][j] = average_reward
                reward_matrix[j][i] = -average_reward

        if True:
            for i in range(self.number_of_agents):
                print(reward_matrix[i])

        x = nash.Game(reward_matrix).support_enumeration()
        self.nash_distribution = torch.FloatTensor(list(x)[0][0]) # Don't ask

        # Add epsilon to the nash equilibrium and normalise
        eps = 0.2
        for i in range(self.number_of_agents):
            self.nash_distribution[i] = eps / self.number_of_agents + self.nash_distribution[i] / (1 + eps)
        print(self.nash_distribution)


        self.nash_categorical = Categorical(self.nash_distribution)


    def updateChampion(self, number_of_iterations):
        for t in range(number_of_iterations):
            i = self.nash_categorical.sample()
            print(f"{t}: {i}")

            trainer = FixedTrainSession(model_training=self.champion, model_fixed=self.agents[i], env=self.env, worker_number=5,\
                                          batch_size=self.batch_size, learning_rate=self.learning_rate, gamma=self.gamma, entropy_rate=self.entropy_rate, is_norming=self.is_norming)

            for k in range(5):
                trainer.runStep()

    def runSession(self, number_of_iterations):
        self.updateNashEquilibrium(10)
        print(f"Nash equilibrium updated! \t{global_timer.getElapsedTime():.3f}s")

        new_agents = [copy.deepcopy(self.agents[i]) for i in range(self.number_of_agents)]
        for t in range(number_of_iterations):
            i, j = int(self.nash_categorical.sample()), int(self.nash_categorical.sample())
            print(f"{t}: {i}, {j}")

            trainer = WinTrainSession(model_red=new_agents[i], model_blue=new_agents[j], env=self.env, worker_number=5,\
                                          learning_rate=self.learning_rate, gamma=self.gamma, entropy_rate=self.entropy_rate, is_norming=self.is_norming)

            for k in range(3):
                trainer.runStep()

        self.agents = new_agents

    def getChampion(self):
        return copy.deepcopy(self.champion)

    def referenceTestChampion(self):
        average_reward, _ = self.game_tester.getAverageReward(self.champion, self.reference_model, 20)
        print(f"average champion reward against the reference is: {average_reward:.3f} \t{global_timer.getElapsedTime():.3f}s")


def makeEnv(step_len, reward_shape = False):
    return gym_haxball.onevoneenviroment.DuelEnviroment(step_len, 3000 / step_len, True, reward_shape = reward_shape)

def main():
    starter_agents = [torch.load("models/classifier.model") for i in range(6)]

    pop = PopulationHandler(initial_agents=starter_agents, env=lambda: makeEnv(5, False), \
                            batch_size=256, learning_rate=1e-4, gamma=1-3e-3, entropy_rate=0.01, is_norming=False)

    for i in range(100):
        pop.runSession(number_of_iterations=20)
        print(f"agent update iteration nr. {i} finished! \t{global_timer.getElapsedTime():.3f}s")
        pop.updateChampion(number_of_iterations=20)
        pop.referenceTestChampion()

        champion = pop.getChampion()
        torch.save(champion, "models/champion_v" + str(i) + ".model")
        print(f"champion v{i} saved! \t{global_timer.getElapsedTime():.3f}s\n\n")

if __name__ == "__main__":
    main()
