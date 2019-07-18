from game_simulator import playeraction
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

    def posToList(self, myTeam):
        if myTeam == "red":
            return [self.x, self.y, self.vx, self.vy]
        elif myTeam == "blue":
            return [840 - self.x, 400 - self.y, -self.vx, -self.vy]
        else:
            raise ValueError

@dataclass
class PlayerState(BallState):
    action: playeraction.Action

    def actToList(self, myTeam):
        if myTeam == "red":
            return list(self.action.rawAction())
        elif myTeam == "blue":
            return list(self.action.flipped().rawAction())
        else:
            raise ValueError

@dataclass
class Frame:
    blues: List[PlayerState]
    reds: List[PlayerState]
    balls: List[BallState]

    def posToNp(self, myTeam = "red", me = 0):
        if myTeam == "blue":
            return np.array(
                    self.blues[me].posToList(myTeam)
                    + [x for p in self.blues[me+1:] for x in p.posToList(myTeam)]
                    + [x for p in self.blues[:me]   for x in p.posToList(myTeam)]
                    + [x for p in self.reds         for x in p.posToList(myTeam)]
                    + [x for b in self.balls        for x in b.posToList(myTeam)]
                    )
        elif myTeam == "red":
            return np.array(
                    self.reds[me].posToList(myTeam)
                    + [x for p in self.reds[:me]   for x in p.posToList(myTeam)]
                    + [x for p in self.reds[me+1:] for x in p.posToList(myTeam)]
                    + [x for p in self.blues       for x in p.posToList(myTeam)]
                    + [x for b in self.balls       for x in b.posToList(myTeam)]
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

    def toNp(self, myTeam, me):
        return np.array([f.posToNp(myTeam, me) for f in self.frames]), np.array([f.singleActToNp(myTeam, me) for f in self.frames])

    @staticmethod
    def load(filename):
        f = open(filename, "rb")
        return pickle.load(f)

    def save(self, filename):
        f = open(filename, "w+b")
        pickle.dump(self, f)
