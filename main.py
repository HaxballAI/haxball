#! /usr/bin/python

from game_simulator import gamesim
from game_displayer import basicdisplayer
from move_displayer import movedisplayer
from game_simulator import gameparams as gp
from agents import ACagent
from agents import humanACagent
from agents import randomagent

import pygame

import numpy as np

import torch

def main():
    model_1 = torch.load("models/trained_nonorm_v9.model")
    model_2 = torch.load("models/trained_nonorm_v11_2.model")

    # Intialise the graphical interface of the game
    red_debug_surf = movedisplayer.DebugSurf()
    blue_debug_surf = movedisplayer.DebugSurf()
    #disp = basicdisplayer.GameWindow(gp.windowwidth, gp.windowheight)
    disp = basicdisplayer.GameWindow(gp.windowwidth + 2 * 256, gp.windowheight,\
                                     debug_surfs = [red_debug_surf.surf, blue_debug_surf.surf])

    red_player_count = 1
    blue_player_count = 1
    ball_count = 1 # Doesn't work with >1 yet as balls reset in the exact center

    # Intialise the agents in the order of all reds sequentially, then blues
    agents = []
    # Red agents
    #agents.append(humanagent.HumanAgent(('w', 'd', 's', 'a', 'LSHIFT'), disp))
    agents.append(ACagent.ACAgent(model_2, "red",  "random", red_debug_surf, False))
    # agents.append(randomagent.RandomAgent())

    # Blue agents

    blueA = ACagent.ACAgent(model_1, "blue", "random", blue_debug_surf, False)
    agents.append(humanACagent.HumanACAgent(('UP', 'RIGHT', 'DOWN', 'LEFT', 'u'), disp, blueA))
    #agents.append(blueA)
    # agents.append(randomagent.RandomAgent())


    # Initialise the game simulator
    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count,
                           printDebug = True, auto_score = True)
    game.run(disp, agents)

if __name__ == "__main__":
    main()
