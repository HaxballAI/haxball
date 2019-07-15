from 1v1enviroment import DuelEnviroment
from ray.rllib.env import multi_agent_env
from gym import core , spaces
from gamesim import gameparams

class DuelFixedGym(multi_agent_env.MultiAgentEnv):
    def __init__(self):
        self.envo = DuelEnviroment()

        self.action_space = spaces.MultiDiscrete([9, 2])
        win_w = gameparams.windowwidth
        win_h = gameparams.windowheight

        self.observation_space = spaces.Box(
            low = np.array[0.0, 0.0, -15.0, -15.0, 0.0, 0.0,
                           -15.0, -15.0, 0.0, 0.0, -15.0, -15.0],
            high = np.array[win_w, win_h, 15.0, 15.0, win_w, win_h,
                            15.0, 15.0, win_w, win_h, 15.0, 15.0,],
            dtype = np.float32
            )

        self.reward_range = (-1,1)

    def getState(self):
        raw_state = self.envo.getState("raw sa pairs")[0]
        raw_state.flatten()
        raw_state.flatten()
        return raw_state

    def getRotatedState(self):
        raw_state = self.envo.getState("raw sa pairs")[0]
        # Flips the states
        raw_state[0], raw_state[1] = raw_state[1], raw_state[0]

        for elem in raw_state:
            elem[0] = gameparams.rotatePos(elem[0])
            elem[1] = gameparams.rotateVel(elem[1])

        raw_state.flatten()
        raw_state.flatten()

        return raw_state

    def getMultiState(self):
        rState = self.getState()
        bState = self.getRotatedState()

        return {"red" : rState, "blue" : bState}

    def step(self, actionDict):
        step_data = self.envo.step(actionDict["red"], actionDict["blue"])

        rewards = {"red" : step_data[2], "blue" : step_data[3]}
        dones = {"red" : step_data[1], "blue" : step_data[1]}
        infos = {"red" : {}, "blue" : {}}

        return [self.getMultiState(), rewards, dones, infos]

    def reset(self):
        self.envo.reset()
        return self.getMultiState()

