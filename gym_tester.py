from gym_haxball import solofixedgym
from game_simulator import playeraction

if __name__ == "__main__":
    env = solofixedgym.DuelFixedGym()

    env.reset()
    running = True
    while running == True:
        env.render()
        results = env.step(playeraction.Action(5, 1))
        if results[2] == True:
            env.reset()
