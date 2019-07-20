#! /usr/bin/python

from game_log import log, log_old
from game_simulator import playeraction
import sys

def convertBall(b): return b
def convertPlayer(p): return log.PlayerState(playeraction.Action(p.move, p.kick))
def convertFrame(f): return log.Frame(map(convertPlayer, f.blues), map(convertPlayer, f.reds), map(convertBall, f.balls))
def convertGame(g): return log.Game(g.red_goals, g.blue_goals, map(convertFrame, g.frames))

for f in sys.argv[1:]:
    print(f"Converting {f}")
    convertGame(log_old.Game.load(f)).save(f)
