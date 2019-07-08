from game_simulator import gamesim
from game_displayer import basicdisplayer
from human_agent import humanagent
from retarded_agent import retardedagent
from data_handler import datahandler

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


    # Intialise the graphical interface of the game
    disp = basicdisplayer.GameWindow(840 , 400)

    # Initialise the game simulator
    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count ,
                        {"printDebug" : True, "auto score" : True})

    # Initialise the data handler (saving data, loading it, etc)
    data_handler = datahandler.DataHandler("rawsaved_games.dat")



    running = True

    while(running):
        # Need to update what keys are being pressed down for the human agents
        disp.updateKeys()
        # Query each agent on what commands should be sent to the game simulator
        commands = [agents[i].getRawAction(disp) for i in range(player_count)]
        game.giveCommands(commands, "raw")

        # Update the graphical interface canvas
        disp.drawThings( game.getState( "full info" ) )

        # Load the last game state to the data handler
        data_handler.loadIntoBuffer(game.getState( "raw sa pairs" ))

        # At some arbitrary point, store the buffered game states into the
        # destination file. In this case it's after a goal has been scored
        if game.was_point_scored:

            data_handler.dumpBufferToFile()
            game.was_point_scored = False


        game.step()


        # Get Debugging data from the game
        if game.frames % 10000 == 0:
            game.getFeedback()

        disp.getInput()
        if disp.rip:
            disp.shutdown()
            running = False


if __name__ == "__main__":
    main()
