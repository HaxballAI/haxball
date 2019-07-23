from gym_haxball.solofixedgym import DuelFixedGym
from gym_haxball.singleplayergym import SingleplayerGym
from game_simulator import playeraction

import random

def runEnv(env):
    env.reset()
    running = True
    while running == True:
        env.render()
        results = env.step(random.randint(0, 17))
        if results[2] == True:
            env.reset()

if __name__ == "__main__":
    # env = DuelFixedGym(10, 400)
    env = SingleplayerGym(15, 50)

    runEnv(env)
