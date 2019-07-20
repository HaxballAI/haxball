#! /usr/bin/python

from game_log import log, log_old
from game_simulator import playeraction
import sys

def convertBall(b): return b
def convertPlayer(p): return log.PlayerState(p.x, p.y, p.vx, p.vy, playeraction.Action(p.move, p.kick))
def convertFrame(f): return log.Frame([convertPlayer(p) for p in f.blues], [convertPlayer(p) for p in f.reds], [convertBall(b) for b in f.balls])
def convertGame(g): return log.Game(g.red_goals, g.blue_goals, [convertFrame(f) for f in g.frames])

for f in sys.argv[1:]:
    print(f"Converting {f}")
    convertGame(log_old.Game.load(f)).save(f)
