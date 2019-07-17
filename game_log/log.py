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
            return [840 - self.x,400 - self.y, -self.vx, -self.vy]
        else:
            raise ValueError

@dataclass
class PlayerState(BallState):
    move: int
    kick: int

    def actToList(self, myTeam):
        if myTeam == "red":
            return [self.move, self.kick]
        elif myTeam == "blue":
            if self.move == 0:
                conv_move = 0
            else:
                conv_move = ((self.move + 3) % 8) + 1
            return [conv_move, self.kick]
        else:
            raise ValueError

@dataclass
class Frame:
    blues: List[PlayerState]
    reds: List[PlayerState]
    balls: List[BallState]

    def posToNp(self, myTeam, me):
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

    def actToNp(self, myTeam, me):
        if myTeam == "blue":
            return np.array(
                    self.blues[me].actToList(myTeam)
                    + [x for p in self.blues[:me]   for x in p.actToList(myTeam)]
                    + [x for p in self.blues[me+1:] for x in p.actToList(myTeam)]
                    + [x for p in self.reds         for x in p.actToList(myTeam)]
                    )
        elif myTeam == "red":
            return np.array(
                    self.reds[me].actToList(myTeam)
                    + [x for p in self.reds[:me]   for x in p.actToList(myTeam)]
                    + [x for p in self.reds[me+1:] for x in p.actToList(myTeam)]
                    + [x for p in self.blues       for x in p.actToList(myTeam)]
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
    frames: List[Frame] = field(default_factory = list)
    red_goals: int
    blue_goals: int

    def append(self, frame):
        self.frames.append(frame)

    def toNpAll(self, myTeam, me):
        return np.array([f.posToNp(myTeam, me) for f in self.frames]), np.array([f.actToNp(myTeam, me) for f in self.frames])

    def toNp(self, myTeam, me):
        return np.array([f.posToNp(myTeam, me) for f in self.frames]), np.array([f.singleActToNp(myTeam, me) for f in self.frames])

    @staticmethod
    def load(filename):
        f = open(filename, "rb")
        return pickle.load(f)

    def save(self, filename):
        f = open(filename, "w+b")
        pickle.dump(self, f)
