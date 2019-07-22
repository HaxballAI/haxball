from gym_haxball import solofixedgym
from game_simulator import playeraction

if __name__ == "__main__":
    env = solofixedgym.DuelFixedGym()

    env.reset()
    running = True
    while running == True:
        env.render()
        results = env.step(7)
        if results[2] == True:
            env.reset()
