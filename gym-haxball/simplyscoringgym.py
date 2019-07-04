import gym
from gym import error, spaces, utils
from gym.utils import seeding

from game_simulator import gamesim




class SingleGoal(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game_sim = gamesim.GameSim(1,0,1)
        self.ticks = 0
        self.stopped = False
    def step(self, action):
            ...
    def reset(self):
                ...
    def render(self, mode='human', close=False):
            ...
