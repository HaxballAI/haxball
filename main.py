#! /usr/bin/python

from game_simulator import gamesim
from game_displayer import basicdisplayer
from human_agent import humanagent
from retarded_agent import retardedagent
from data_handler import datahandler
from move_displayer import movedisplayer
from network import TwoLayerNet, DIMS
#from model_tuner import tuner
from game_simulator import gameparams as gp

from utils import flatten

import pygame

import random

import numpy as np
from random import randrange
from moveclassifier import DIMS

import torch

def main():
    model = TwoLayerNet(*DIMS)
    model.load_state_dict(torch.load("initialmodelweights.dat"))
    model.eval()

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
    #disp = basicdisplayer.GameWindow(840, 400)
    disp = basicdisplayer.GameWindow(1096, 400)

    # Initialise the game simulator
    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count,
                           {"printDebug" : True, "auto score" : True})

    # Initialise the data handler (saving data, loading it, etc)
    data_handler = datahandler.DataHandler("rawsaved_games.dat")

    running = True

    #tuner.tuner()

    # FUNCTION THAT DEFINES OPPONENT REPLACE RHS WITH THIS

    def opponent(x):
        movepred, kickpred = model(x)
        ran_move = np.random.choice(len(movepred), p = torch.nn.Softmax(dim = 0)(movepred).detach().numpy())
        p_kick = float(kickpred[0])
        ran_kick = np.random.choice([False, True], p = [1 - p_kick, p_kick])
        debug_surf = movedisplayer.drawMove(torch.nn.Softmax(dim = 0)(movepred).detach().numpy(), ran_move)
        return [ran_move, ran_kick], debug_surf

    while(running):
        # Need to update what keys are being pressed down for the human agents
        disp.updateKeys()
        # Query each agent on what commands should be sent to the game simulator
        commands = [agents[i].getRawAction(disp) for i in range(player_count)]

        c_state = flatten( game.getState(   "raw state" ) )
        c_state[4] -= c_state[0]
        c_state[5] -= c_state[1]
        c_state[8] -= c_state[0]
        c_state[9] -= c_state[1]
        c_state = np.array( c_state )
        #c_state -= gp.mean
        #c_state /= gp.stdev

        commands[0], debug_surf = opponent( torch.FloatTensor( c_state) )
        game.giveCommands(commands, "raw")

        # Update the graphical interface canvas
        disp.drawThings(game.getState("full info"))

        # Add the debug thing
        disp.win.blit(debug_surf, (840,0))

        # Display
        disp.clock.tick(disp.fps)
        pygame.display.update()

        # Load the last game state to the data handler
        data_handler.loadIntoBuffer(game.getState("raw sa pairs"))

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
