from game_simulator import gamesim
from game_displayer import basicdisplayer
from human_agent import humanagent
from retarded_agent import retardedagent
from data_handler import datahandler
#from model_tuner import tuner

import game_simulator.gameparams as gp


import random

import numpy as np
from random import randrange

import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out-1)
        self.linear3 = torch.nn.Linear(H, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        res = self.linear2(h_relu)
        movepred = self.linear2(h_relu)
        kickpred = torch.nn.Sigmoid()(self.linear3(h_relu))
        return movepred, kickpred

model = TwoLayerNet(12, 100, 10)
model.load_state_dict(torch.load("initialmodelweights.dat"))
model.eval()

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

    #tuner.tuner()

    # FUNCTION THAT DEFINES OPPONENT REPLACE RHS WITH THIS

    def opponent(x):
        movepred, kickpred  = model(x)
        ran_move = np.random.choice( len( movepred ) , p = torch.nn.Softmax()( movepred ).detach().numpy() )
        p_kick = float(kickpred[0])
        ran_kick = np.random.choice( [False, True] , p = [ 1 - p_kick , p_kick] )
        return [ran_move , ran_kick]

    while(running):
        # Need to update what keys are being pressed down for the human agents
        disp.updateKeys()
        # Query each agent on what commands should be sent to the game simulator
        commands = [agents[i].getRawAction(disp) for i in range(player_count)]
        commands[0] = opponent( torch.tensor(gp.flatten( game.getState("raw state") )) )
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
