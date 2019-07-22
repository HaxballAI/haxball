from agents import ACagent
from game_simulator import gameparams
from gym_haxball.duelenvironment import DuelEnviroment
from utils import flatten

from gym import core, spaces
import numpy as np
import torch

class DuelFixedGym(core.Env):
    def __init__(self):
        self.envo = DuelEnviroment(15, 400)

        win_w = gameparams.windowwidth
        win_h = gameparams.windowheight

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
           low = np.array([0.0, 0.0, -15.0, -15.0, 0.0, 0.0,
                          -15.0, -15.0, 0.0, 0.0, -15.0, -15.0]),
           high = np.array([win_w, win_h, 15.0, 15.0, win_w, win_h,
                            15.0, 15.0, win_w, win_h, 15.0, 15.0]),
            dtype = np.float32
           )

        opponent_model = torch.load("models/train2.model")
        self.opponent = ACagent.ACAgent(opponent_model, "blue",  "random")

    def getState(self):
        return self.envo.getState()

    def getOpponentAction(self):
        return self.opponent.getAction(self.envo.game_sim.log())

    def step(self, single_action):
        # advances the simulator by step_len number of steps. Returns a list of
        # [observation (object), reward (float), done (bool), info (dict)]
        # Actions must be integeres in the range [0, 18)
        opponent_single_action = self.getOpponentAction().singleAction()
        step_data = self.envo.step(single_action, opponent_single_action)
        return step_data

    def render(self, mode='human'):
        # Only the human consumptiom mode is implemented
        self.envo.render(mode)

    def reset(self):
        self.envo.reset()
        return self.getState()
