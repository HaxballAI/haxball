#! /usr/bin/python

import game_log.log as l
import numpy as np
import torch
from model_updaters import learnFromPlayedGames
import model_updaters
import network
import gym_haxball.onevoneenviroment
import basic_trainers.actor_critic as ac_t

def getData(data_dir, game_number, normalise = False):
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
def newNet(net_name, data_dir, game_number, epochs, learning_rate, batch_size, normalise = True):
    data_tensor, action_data = getData(data_dir, game_number, normalise)
    model = network.GregPolicy()
    learnFromPlayedGames(model, data_tensor, action_data, epochs, learning_rate, batch_size)
    torch.save(model, "models/" + net_name + ".model")

# IMPROVED THE NETWORK GIVEN BY NET_NAME
def improveNet(net_name, data_dir, game_number, epochs, learning_rate, batch_size, normalise = True):
    data_tensor, action_data = getData(data_dir, game_number, normalise)
    model = torch.load(f"models/{net_name}.model")
    learnFromPlayedGames(model, data_tensor, action_data, epochs, learning_rate, batch_size)
    torch.save(model, f"models/{net_name}.model")

def envMaker():
    return gym_haxball.onevoneenviroment.DuelEnviroment()

if __name__ == "__main__":
    newNet("gregNet","sebgames",100,3,1e-3,32)
    if False:
        mod = torch.load("models/gregNet.model")
        trainer = ac_t.TrainSession(mod, envMaker, 5, 100, 1e-3, 1- 1e-2)
        for i in range(10):
            print("Step " + str(i))
            trainer.runStep()
        torch.save(mod, "models/gregNet2.model")
