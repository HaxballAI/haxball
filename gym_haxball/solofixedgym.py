from game_simulator import gameparams
from gym_haxball.duelenvironment import DuelEnviroment
from agents import ACagent

from gym import core, spaces
import numpy as np
import torch

def env_constructor(opponent, step_length = 10, max_steps = 400):
    def toRet():
        return DuelFixedGym({"step_length": step_length, "max_steps" : max_steps ,\
                             "model object": opponent})
    return toRet

class DuelFixedGym(core.Env):
    def __init__(self, config):
        if "step_length" in config:
            step_len = config["step_length"]
        else:
            step_len = 7

        if "max_steps" in config:
            max_steps = config["max_steps"]
        else:
            max_steps = 400

        if "norming" in config:
            norming = config["norming"]
        else:
            norming = True

        if "model name" in config:
            model_name = config["model name"]
        else:
            model_name = "arun_v6"

        if "model object" in config:
            model_obj = config["model object"]
        else:
            model_obj = None

        self.envo = DuelEnviroment(step_len, max_steps, norming)

        win_w = gameparams.windowwidth
        win_h = gameparams.windowheight

        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(
           low = np.array([0.0, 0.0, -15.0, -15.0, 0.0, 0.0,
                          -15.0, -15.0, 0.0, 0.0, -15.0, -15.0]),
           high = np.array([win_w, win_h, 15.0, 15.0, win_w, win_h,
                            15.0, 15.0, win_w, win_h, 15.0, 15.0]),
            dtype = np.float32
           )

        if model_obj = None:
            opponent_model = torch.load(f"models/{model}.model").to("cpu")
            self.opponent = ACagent.ACAgent(opponent_model, "blue",  "random")
        else:
            self.opponent = ACagent.ACAgent(model_obj, "blue",  "random")

    def getState(self):
        return self.envo.getState()

    def getOpponentAction(self):
        return self.opponent.getAction(self.envo.game_sim.log())

    def step(self, action_single):
        # advances the simulator by step_len number of steps. Returns a list of
        # [observation (object), reward (float), done (bool), info (dict)]
        # Actions must be integeres in the range [0, 18)
        opponent_action_single = self.getOpponentAction().singleAction()
        step_data = self.envo.step(action_single, opponent_action_single)
        return step_data

    def render(self, mode='human'):
        # Only the human consumptiom mode is implemented
        self.envo.render(mode)

    def reset(self):
        return self.envo.reset()
