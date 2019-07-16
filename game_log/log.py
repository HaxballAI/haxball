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
        if myTeam == "blue":
            return [x, y, vx, vy]
        elif myTeam == "red":
            return [840 - x, y, vx, vy]
        else:
            raise ValueError

@dataclass
class PlayerState(BallState):
    move: int
    kick: int

    def actToList(self):
        return [move, kick]

@dataclass
class Frame:
    blues: List[PlayerState]
    reds: List[PlayerState]
    balls: List[BallState]

    def posToNp(myTeam, me):
        if myTeam == "blue":
            return np.array(
                    blues[me].posToList(myTeam)
                    + [x for p in blues[:me]   for x in p.posToList(myTeam)]
                    + [x for p in blues[me+1:] for x in p.posToList(myTeam)]
                    + [x for p in reds         for x in p.posToList(myTeam)]
                    + [x for b in balls        for x in b.posToList(myTeam)]
                    )
        elif myTeam == "red":
            return np.array(
                    reds[me].posToList(myTeam)
                    + [x for p in reds[:me]   for x in p.posToList(myTeam)]
                    + [x for p in reds[me+1:] for x in p.posToList(myTeam)]
                    + [x for p in blues       for x in p.posToList(myTeam)]
                    + [x for b in balls       for x in b.posToList(myTeam)]
                    )
        else:
            raise ValueError

    def actToNp(myTeam, me):
        if myTeam == "blue":
            return np.array(
                    blues[me].actToList
                    + [x for p in blues[:me]   for x in p.actToList()]
                    + [x for p in blues[me+1:] for x in p.actToList()]
                    + [x for p in reds         for x in p.actToList()]
                    )
        elif myTeam == "red":
            return np.array(
                    reds[me].actToList()
                    + [x for p in reds[:me]   for x in p.actToList()]
                    + [x for p in reds[me+1:] for x in p.actToList()]
                    + [x for p in blues       for x in p.actToList()]
                    )
        else:
            raise ValueError

@dataclass
class Game:
    frames: List[Frame] = field(default_factory = list)

    def append(self, frame):
        self.frames.append(frame)
        print(frame.blues[0].x, frame.blues[0].y)

    def toNp(self, myTeam, me):
        return np.array([f.posToNp(myTeam, me) for f in frames]), np.array([f.actToNp(myTeam, me) for f in self.frames])

    @staticmethod
    def load(filename):
        f = open(filename, "rb")
        return pickle.load(f)

    def save(self, filename):
        f = open(filename, "wb")
        pickle.dump(self, f)
