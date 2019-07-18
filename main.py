#! /usr/bin/python

from game_simulator import gamesim
from game_displayer import basicdisplayer
from move_displayer import movedisplayer
from game_simulator import gameparams as gp
from agents import ACagent
from agents import humanagent

import pygame

import numpy as np

import torch

def main():
    model = torch.load("newSebNet.model")

    # Intialise the graphical interface of the game
    #disp = basicdisplayer.GameWindow(840, 400)
    disp = basicdisplayer.GameWindow(840 + 2 * 256, 400)

    red_player_count = 1
    blue_player_count = 1
    ball_count = 1 # Doesn't work with >1 yet as balls reset in the exact center

    # Intialise the agents in the order of all reds sequentially, then blues
    agents = []
    # Red agents
    #agents.append(humanagent.HumanAgent(('w', 'd', 's', 'a', 'x'), disp))
    red_debug_surf = movedisplayer.DebugSurf()
    agents.append(ACagent.ACAgent(model, "red",  "random", red_debug_surf))
    for i in range(red_player_count - 1):
        agents.append(retardedagent.RetardedAgent())
    # Blue agents
    #agents.append(humanagent.HumanAgent(('UP', 'RIGHT', 'DOWN', 'LEFT', 'RCTRL'), disp))
    blue_debug_surf = movedisplayer.DebugSurf()
    agents.append(ACagent.ACAgent(model, "blue", "random", blue_debug_surf))
    for i in range(blue_player_count - 1):
        agents.append(retardedagent.RetardedAgent())


    # Initialise the game simulator
    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count,
                           {"printDebug" : True, "auto score" : True})

    running = True

    while(running):
        # Need to update what keys are being pressed down for the human agents
        disp.updateKeys()

        f_data = game.log()

        # Query each agent on what commands should be sent to the game simulator
        game.giveCommands([a.getAction(f_data) for a in agents])

        # Update the graphical interface canvas
        disp.drawFrame(game.log())

        # Add the debug thing
        disp.win.blit(red_debug_surf.surf,  (840, 0))
        disp.win.blit(blue_debug_surf.surf, (840 + 256, 0))

        # Display
        disp.clock.tick(disp.fps)
        pygame.display.update()

        # Load the last game state to the data handler

        # At some arbitrary point, store the buffered game states into the
        # destination file. In this case it's after a goal has been scored
        if game.was_point_scored:
            game.was_point_scored = False

        game.step()

        # Get Debugging data from the game


        disp.getInput()
        if disp.rip:
            disp.shutdown()
            running = False

if __name__ == "__main__":
    main()
