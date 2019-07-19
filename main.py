#! /usr/bin/python

from game_simulator import gamesim
from game_displayer import basicdisplayer
from move_displayer import movedisplayer
from game_simulator import gameparams as gp
from agents import ACagent
from agents import humanagent
from agents import randomagent

import pygame

import numpy as np

import torch

def main():
    model = torch.load("models/train2.model")

    # Intialise the graphical interface of the game
    #disp = basicdisplayer.GameWindow(gp.windowwidth, gp.windowheight)
    disp = basicdisplayer.GameWindow(gp.windowwidth + 2 * 256, gp.windowheight)

    red_player_count = 1
    blue_player_count = 1
    ball_count = 1 # Doesn't work with >1 yet as balls reset in the exact center

    # Intialise the agents in the order of all reds sequentially, then blues
    agents = []
    # Red agents
    #agents.append(humanagent.HumanAgent(('w', 'd', 's', 'a', 'LSHIFT'), disp))
    red_debug_surf = movedisplayer.DebugSurf()
    agents.append(ACagent.ACAgent(model, "red",  "random", red_debug_surf))
    for i in range(red_player_count - 1):
        agents.append(randomagent.RandomAgent())
    # Blue agents
    #agents.append(humanagent.HumanAgent(('UP', 'RIGHT', 'DOWN', 'LEFT', 'u'), disp))
    blue_debug_surf = movedisplayer.DebugSurf()
    agents.append(ACagent.ACAgent(model, "blue", "random", blue_debug_surf))
    for i in range(blue_player_count - 1):
        agents.append(randomagent.RandomAgent())


    # Initialise the game simulator
    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count,
                           {"printDebug" : True, "auto score" : True})

    while True:
        # Need to update what keys are being pressed down for the human agents
        disp.updateKeys()

        f_data = game.log()

        # Query each agent on what commands should be sent to the game simulator
        game.giveCommands([a.getAction(f_data) for a in agents])

        # Update the graphical interface canvas
        disp.drawFrame(game.log(), [red_debug_surf.surf, blue_debug_surf.surf])

        game.step()

        disp.getInput()

        if disp.rip:
            disp.shutdown()
            break

if __name__ == "__main__":
    main()
