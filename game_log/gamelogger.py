import game_simulator.gamesim as gs
from game_displayer import basicdisplayer
from human_agent import humanagent
from game_log import log
import pygame

def recordPlayerGames(dest, games_to_play = 10):
    disp = basicdisplayer.GameWindow(840, 400)
    blue_agent = humanagent.HumanAgent(('w', 'd', 's', 'a', 'f'))
    red_agent = humanagent.HumanAgent(('UP', 'RIGHT', 'DOWN', 'LEFT', 'RCTRL'))
    pygame.init()

    for game_number in range(games_to_play):
        sim = gs.GameSim(1,1,1)
        game_done = False
        game_log = log.Game()
        while not game_done:
            disp.updateKeys()
            red_move = red_agent.getRawAction(disp)
            blue_move = blue_agent.getRawAction(disp)

            sim.giveCommands([red_move, blue_move], "raw")
            sim.step()

            game_log.append(sim.log())

            disp.drawThings(sim.getState("full info"))
            pygame.display.update()
            disp.clock.tick(disp.fps)

            goals = sim.checkGoals()
            if sum(goals) > 0:
                game_done = True
                game_log.red_goals = goals[0]
                game_log.blue_goals = goals[1]


            disp.getInput()
            if disp.rip:
                return
        game_log.save(f"{dest}/{game_number}")
