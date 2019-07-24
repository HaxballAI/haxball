import gym_haxball.onevoneenviroment
import torch

def playGames(model_red, model_blue, games, step_len = 15, max_steps = 200, randStart = False, normalising = False):
    env = gym_haxball.onevoneenviroment.DuelEnviroment(step_len = step_len, max_steps = max_steps, rand_reset = randStart)
    score = [0,0]
    for g in range(games):
        print(g)
        done = False
        state = env.reset()
        rew = [0,0]
        while not done:
            red_a = model_red.getAction(state)
            blue_a = model_red.getAction(state)
            state, rew, done, _ = env.step(red_a, blue_a)
        score[0] += rew[0]
        score[1] += rew[1]
    print("Final score:")
    print("Red:" + str(score[0]))
    print("Blue:" + str(score[1]))
    return(score)
