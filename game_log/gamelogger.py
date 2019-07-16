import game_simulator.gamesim as gs
from game_displayer import basicdisplayer
from human_agent import humanagent
import pygame


def recordPlayerGames(session_name, game_to_play = 10):
    sim = gs.GameSim(1,1,1)
    disp = basicdisplayer.GameWindow(840, 400)
    blue_agent = humanagent.HumanAgent(('w', 'd', 's', 'a', 'f'))
    red_agent = humanagent.HumanAgent(('UP', 'RIGHT', 'DOWN', 'LEFT', 'RCTRL'))
    pygame.init()

    for game_number in range(game_to_play):
        game_done = False
        while not game_done:
            disp.updateKeys()
            red_move = red_agent.getRawAction(disp)
            blue_move = blue_agent.getRawAction(disp)

            sim.giveCommands([red_move, blue_move], "raw")
            sim.step()

            disp.drawThings(sim.getState("full info"))
            pygame.display.update()
            disp.clock.tick(disp.fps)

            if sum(sim.checkGoals()) > 0:
                game_done = True

            disp.getInput()
            if disp.rip:
                game_done = True
        #Save games here
