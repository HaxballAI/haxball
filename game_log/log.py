from game_simulator import playeraction
from game_simulator import gameparams as gp
from dataclasses import dataclass, field
from typing import List
import pickle
import numpy as np

@dataclass
class BallState:
    x: float
    y: float
    vx: float
    vy: float

    def posToList(self, myTeam, normalise = True):
        if myTeam == "red":
            l = [self.x, self.y, self.vx, self.vy]
        elif myTeam == "blue":
            l = [gp.windowwidth - self.x, gp.windowheight - self.y, -self.vx, -self.vy]
        else:
            raise ValueError
        if normalise:
            v_max = gp.accel * gp.playerdamping / (1 - gp.playerdamping)
            l = [l[0] / gp.windowwidth, l[1] / gp.windowheight, l[2] / v_max, l[3] / v_max]
        return l

@dataclass
class PlayerState(BallState):
    action: playeraction.Action

    def actToList(self, myTeam):
        if myTeam == "red":
            return list(self.action.singleAction())
        elif myTeam == "blue":
            return list(self.action.flipped().singleAction())
        else:
            raise ValueError

@dataclass
class Frame:
    blues: List[PlayerState]
    reds: List[PlayerState]
    balls: List[BallState]

    def posToNp(self, myTeam = "red", me = 0, normalise = True):
        if myTeam == "blue":
            return np.array(
                    self.blues[me].posToList(myTeam, normalise)
                    + [x for p in self.blues[me+1:] for x in p.posToList(myTeam, normalise)]
                    + [x for p in self.blues[:me]   for x in p.posToList(myTeam, normalise)]
                    + [x for p in self.reds         for x in p.posToList(myTeam, normalise)]
                    + [x for b in self.balls        for x in b.posToList(myTeam, normalise)]
                    )
        elif myTeam == "red":
            return np.array(
                    self.reds[me].posToList(myTeam, normalise)
                    + [x for p in self.reds[:me]   for x in p.posToList(myTeam, normalise)]
                    + [x for p in self.reds[me+1:] for x in p.posToList(myTeam, normalise)]
                    + [x for p in self.blues       for x in p.posToList(myTeam, normalise)]
                    + [x for b in self.balls       for x in b.posToList(myTeam, normalise)]
                    )
        else:
            raise ValueError

    def singleActToNp(self, myTeam, me):
        if myTeam == "blue":
            return np.array(self.blues[me].actToList(myTeam))
        elif myTeam == "red":
            return np.array(self.reds[me].actToList(myTeam))
        else:
            raise ValueError

@dataclass
class Game:
    red_goals: int = 0
    blue_goals: int = 0
    frames: List[Frame] = field(default_factory = list)

    def append(self, frame):
        self.frames.append(frame)

    def toNp(self, myTeam, me, normalise = True):
        return np.array([f.posToNp(myTeam, me, normalise) for f in self.frames]), np.array([f.singleActToNp(myTeam, me) for f in self.frames])

    @staticmethod
    def load(filename):
        f = open(filename, "rb")
        return pickle.load(f)

    def save(self, filename):
        f = open(filename, "w+b")
        pickle.dump(self, f)
