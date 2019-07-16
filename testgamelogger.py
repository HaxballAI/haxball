#! /usr/bin/python

from game_log import gamelogger
import sys

gamelogger.recordPlayerGames(sys.argv[1], games_to_play = 10)

