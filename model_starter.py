#! /usr/bin/python

import game_log.log as l
import numpy as np
import torch
from model_updaters import learnFromPlayedGames
import model_updaters
import network
import gym_haxball.onevoneenviroment
from basic_trainers.actor_critic_symmetric import TrainSession as SymmetricTrainSession
from basic_trainers.actor_critic_fixed import TrainSession as FixedTrainSession
import random
from utils import global_timer

def noiseArray():
    to_ret = np.zeros(12)
    for i in [0,1,4,5,8,9]:
        to_ret[i] = random.normalvariate(0, 0.001)
    for i in [2,3,6,7,10,11]:
        to_ret[i] = random.normalvariate(0, 1)
    return to_ret

def makeVelIntoNoise(l):
    for i in [2,3,6,7,10,11]:
        l[i] = random.normalvariate(0, 0.1)


def getData(data_dir, game_number, normalise = False, add_noise = False):
    loser_frames = []
    winner_frames = []
    loser_actions = []
    winner_actions = []
    for g in range(game_number):
        game = l.Game.load(data_dir + "/" + str(g))
        if game.blue_goals == 0:
            assert game.red_goals == 1
            winf, wina = game.toNp("red" , 0, normalise)
            winner_frames.append(winf)
            winner_actions.append(wina)

            losf, losa = game.toNp("blue" , 0, normalise)
            loser_frames.append(losf)
            loser_actions.append(losa)




        elif game.red_goals == 0:
            assert game.blue_goals == 1
            winf, wina = game.toNp("blue" , 0, normalise)
            winner_frames.append(winf)
            winner_actions.append(wina)

            losf, losa = game.toNp("red" , 0, normalise)
            loser_frames.append(losf)
            loser_actions.append(losa)

        else:
            raise ValueError

        if add_noise:
            makeVelIntoNoise(winner_frames[-1])
            makeVelIntoNoise(loser_frames[-1])


    print("Data loaded.")
    loser_frames = np.concatenate(loser_frames)
    winner_frames = np.concatenate(winner_frames)
    loser_actions = np.concatenate(loser_actions)
    winner_actions = np.concatenate(winner_actions)


    assert len(loser_frames) == len(loser_actions)
    p = np.random.permutation(len(loser_frames))
    loser_frames = loser_frames[p]
    loser_actions = loser_actions[p]

    assert len(winner_frames) == len(winner_actions)
    p = np.random.permutation(len(winner_frames))
    winner_frames = winner_frames[p]
    winner_actions = winner_actions[p]

    print("Data shuffled.")



    loser_actions = np.transpose(loser_actions)
    winner_actions = np.transpose(winner_actions)

    return torch.FloatTensor([loser_frames, winner_frames]) , (loser_actions, winner_actions)



# MAKE AND TRAINS A NEW NETWORK BASED ON DATA GIVEN BY DATA_DIR WHICH HAS TO
# BE IN FORMAT OF A FILE OF GAME LOGS INDEXED BY NUMBER
def newNet(net_name, data_dir, game_number, epochs, learning_rate, batch_size, normalise = True, add_noise = False):
    data_tensor, action_data = getData(data_dir, game_number, normalise, add_noise)
    model = network.GregPolicy2()
    learnFromPlayedGames(model, data_tensor, action_data, epochs, learning_rate, batch_size)
    torch.save(model, "models/" + net_name + ".model")

# IMPROVED THE NETWORK GIVEN BY NET_NAME
def improveNet(net_name, data_dir, game_number, epochs, learning_rate, batch_size, normalise = True, add_noise = False):
    data_tensor, action_data = getData(data_dir, game_number, normalise, add_noise)
    model = torch.load(f"models/{net_name}.model")
    learnFromPlayedGames(model, data_tensor, action_data, epochs, learning_rate, batch_size)
    torch.save(model, f"models/{net_name}.model")

def makeEnv(step_len, reward_shape = False):
    return gym_haxball.onevoneenviroment.DuelEnviroment(step_len, 3000 / step_len, True, reward_shape = reward_shape)


def actorTrain(primary_model, env, save_dir, number_of_steps, batch_size, learning_rate, gamma, entropy_rate, is_norming, worker_number, save_frequency=2147483648, secondary_model = None):
    if secondary_model == None:
        trainer = SymmetricTrainSession(model=primary_model, env=env, worker_number=worker_number,\
                                      batch_size=batch_size, learning_rate=learning_rate, gamma=gamma, entropy_rate=entropy_rate, is_norming=is_norming)
    else:
        trainer = FixedTrainSession(model_training=primary_model, model_fixed=secondary_model, env=env, worker_number=worker_number,\
                                      batch_size=batch_size, learning_rate=learning_rate, gamma=gamma, entropy_rate=entropy_rate, is_norming=is_norming)

    cnt = 0
    for i in range(number_of_steps):
        print("Step {}, {:.3f}s".format(str(i), global_timer.getElapsedTime()))
        trainer.runStep()

        if i % save_frequency == 0 and cnt > 0:
            torch.save(primary_model, save_dir + "_v" + str(cnt) + ".model")
            cnt += 1
    torch.save(primary_model, save_dir + ".model")

# v2, 3e-4, v3 1e-3, v4 arun_v6 vs psychov1
if __name__ == "__main__":

    model = torch.load("models/psycho_v5.model")
    model_fixed_opponent = torch.load("models/psycho_v1.model")

    actorTrain(primary_model=model, secondary_model=None, env=lambda: makeEnv(5, False), worker_number=15,\
                batch_size=256, learning_rate=1e-4, gamma=1-3e-3, entropy_rate=0.004, is_norming=False,\
                save_dir="models/psycho_v5_imp", save_frequency=10, number_of_steps=1000)
