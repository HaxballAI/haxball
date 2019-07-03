from game_simulator import gamesim

import numpy as np
from random import randrange


def main():
    red_player_count = 4
    blue_player_count = 4
    ball_count = 1 # Doesn't work with >1 yet as balls reset in the exact center

    game = gamesim.GameSim(red_player_count, blue_player_count, ball_count)
    game.getFeedback()

    while(True):
        game.giveCommands([[randrange(9), 1] for i in range(red_player_count + blue_player_count)])
        game.step()
        game.getFeedback()


if __name__ == "__main__":
    main()
