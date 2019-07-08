from game_simulator import gamesim
from game_displayer import basicdisplayer
from human_agent import humanagent
from retarded_agent import retardedagent
import os
import pickle

import random

import numpy as np
from random import randrange

'''critic = torch.nn.Sequential(
    torch.nn.Linear(4, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.functional.sigmoid()
    )
actor = torch.nn.Sequential(
    torch.nn.Linear(, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.functional.sigmoid()'''

def main():
    red_player_count = 1
    blue_player_count = 1
    player_count = red_player_count + blue_player_count
    ball_count = 1 # Doesn't work with >1 yet as balls reset in the exact center

    # Intialise the agents in the order of all reds sequentially, then blues
    agents = []
    # Red agents
    agents.append(humanagent.HumanAgent(('w', 'd', 's', 'a', 'x')))
        for i in range(red_player_count - 1):
        agents.append(retardedagent.RetardedAgent())
    # Blue agents
    agents.append(humanagent.HumanAgent(('UP', 'RIGHT', 'DOWN', 'LEFT', 'RCTRL')))
    for i in range(blue_player_count - 1):
        agents.append(retardedagent.RetardedAgent())


    disp = basicdisplayer.GameWindow(840 , 400)

    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count ,
                        {"printDebug" : True , "auto score" : True})
    game.getFeedback()

    gamedata = []

    running = True

    while(running):
        # Need to update what keys are being pressed down for the human agents
        disp.updateKeys()
        commands = [agents[i].getRawAction(disp) for i in range(player_count)]
        game.giveCommands(commands, "raw")
        # if random.random() < 0.01:
        #     game.giveCommands([[randrange(9), 1] for i in range(red_player_count + blue_player_count)] , "raw")

        disp.drawThings( game.getState( "full info" ) )

        gamedata.append([game.getState( "state-action pairs" )])

        if game.was_point_scored:
            print("goal!")
            saved_games_filename = 'save_game_1v1.dat'
            games = []
            if os.path.exists(saved_games_filename) and os.path.getsize(saved_games_filename) > 0:
                with open(saved_games_filename,'rb') as rfp:
                    games = pickle.load(rfp)
            games.append(gamedata)
            with open(saved_games_filename,'wb') as wfp:
                pickle.dump(games, wfp)
                print("dumped!")


            for frame in gamedata:
                if game.red_last_goal:
                    classify(critic, frame, 1, loss_f, learning_rate = 1e-4, steps = 500)
                else:
                    classify(critic, frame, 0, loss_f, learning_rate = 1e-4, steps = 500)'''

            gamedata = []
            game.was_point_scored = False


        game.step()


        if game.frames % 10000 == 0:
            game.getFeedback()

        disp.getInput()
        if disp.rip:
            disp.shutdown()
            running = False


if __name__ == "__main__":
    main()
