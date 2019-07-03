from game_simulator import gamesim
from game_displayer import basicdisplayer

import numpy as np
from random import randrange


def main():
    red_player_count = 4
    blue_player_count = 4
    ball_count = 1 # Doesn't work with >1 yet as balls reset in the exact center

    disp = basicdisplayer.GameWindow(840 , 400)

    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count ,
                        {"printDebug" : True})
    game.getFeedback()

    running = True

    while(running):
        game.giveCommands([[randrange(9), 1] for i in range(red_player_count + blue_player_count)] , "raw")
        game.step()
        disp.drawThings( game.getState( "positions" ) )

        if game.frames % 10000 == 0:
            game.getFeedback()

        disp.getInput()
        if disp.rip:
            disp.shutdown()
            running = False


if __name__ == "__main__":
    main()
