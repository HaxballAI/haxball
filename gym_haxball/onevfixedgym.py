from gym_haxball.onevoneenviroment import DuelEnviroment
from gym import core, spaces
from game_simulator import gameparams
import numpy as np

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    if isinstance(S[0], type(np.array([])) ):
        return flatten(S[0].tolist()) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

class DuelFixedGym(core.Env):

    def __init__(self, config):
        self.envo = DuelEnviroment()
        self.opponent = config["opponent"]
        self.action_space = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(2)))
        win_w = gameparams.windowwidth
        win_h = gameparams.windowheight

        self.observation_space = spaces.Box(
           low = np.array([0.0, 0.0, -15.0, -15.0, 0.0, 0.0,
                          -15.0, -15.0, 0.0, 0.0, -15.0, -15.0]),
           high = np.array([win_w, win_h, 15.0, 15.0, win_w, win_h,
                            15.0, 15.0, win_w, win_h, 15.0, 15.0]),
            dtype = np.float32
           )



        # self.reward_range = (-1,1)

    def getState(self):
        raw_state = self.envo.getState()[0]

        return flatten(raw_state)

    def getRotatedState(self):
        raw_state = self.envo.getState()[0]
        # Flips the states
        raw_state[0], raw_state[1] = raw_state[1], raw_state[0]

        for elem in raw_state:
            elem[0] = gameparams.rotatePos(elem[0])
            elem[1] = gameparams.rotateVel(elem[1])

        return flatten(raw_state)

    def getOpAction(self):
        return self.opponent( self.getRotatedState() )

    def step(self, action):
        opAction = self.getOpAction()
        step_data = self.envo.step(action, opAction)

        return [self.getState(), step_data[2], step_data[1], {}]

    def reset(self):
        self.envo.reset()
        return self.getState()