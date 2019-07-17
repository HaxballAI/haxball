#! /usr/bin/python

from game_simulator import gamesim
from game_displayer import basicdisplayer
from agents import humanagent
from agents import retardedagent
from data_handler import datahandler
from move_displayer import movedisplayer
from network import Policy, DIMS
#from model_tuner import tuner
from game_simulator import gameparams as gp

from utils import flatten

import pygame

import random

import numpy as np
from random import randrange

import torch

def main():
    #model = Policy(*flatten(DIMS))
    #model.load_state_dict(torch.load("initialmodelweights.dat"))
    #model.eval()

    # Intialise the graphical interface of the game
    #disp = basicdisplayer.GameWindow(840, 400)
    disp = basicdisplayer.GameWindow(1096, 400)

    red_player_count = 1
    blue_player_count = 1
    player_count = red_player_count + blue_player_count
    ball_count = 1 # Doesn't work with >1 yet as balls reset in the exact center

    # Intialise the agents in the order of all reds sequentially, then blues
    agents = []
    # Red agents
    agents.append(humanagent.HumanAgent(('w', 'd', 's', 'a', 'x'), disp))
    for i in range(red_player_count - 1):
        agents.append(retardedagent.RetardedAgent())
    # Blue agents
    agents.append(humanagent.HumanAgent(('UP', 'RIGHT', 'DOWN', 'LEFT', 'RCTRL'), disp))
    for i in range(blue_player_count - 1):
        agents.append(retardedagent.RetardedAgent())

    # Initialise the game simulator
    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count,
                           {"printDebug" : True, "auto score" : True})

    # Initialise the data handler (saving data, loading it, etc)
    data_handler = datahandler.DataHandler("rawsaved_games.dat")

    running = True

    #tuner.tuner()



    while(running):
        # Need to update what keys are being pressed down for the human agents
        disp.updateKeys()
        # Query each agent on what commands should be sent to the game simulator
        commands = [agents[i].getRawAction(game.getState("raw state")) for i in range(player_count)]


        game.giveCommands(commands, "raw")

        # Update the graphical interface canvas
        disp.drawThings(game.getState("full info"))


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


        if game.frames % 1000 == 0:
            print("c_state:")
            print(c_state)
            print("c_state_norm:")
            print(c_state_norm)
            game.getFeedback()

        disp.getInput()
        if disp.rip:
            disp.shutdown()
            running = False

if __name__ == "__main__":
    main()
