from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class BallState:
    x: float
    y: float
    vx: float
    vy: float

    def posToList():
        return [x, y, vx, vy]

class PlayerState(BallState):
    move: int
    kick: int

    def actToList():
        return [move, kick]

@dataclass
class Frame:
    blues: List[PlayerState]
    reds: List[PlayerState]
    balls: List[BallState]

    def posToNp(myTeam, me):
        if myTeam == "blue":
            return np.array(
                    flatMap(PlayerState.posToList, blues[me])
                    + flatMap(PlayerState.posToList, blues[:me])
                    + flatMap(PlayerState.posToList, blues[me+1:])
                    + flatMap(PlayerState.posToList, reds)
                    + flatMap(BallState.posToList, balls)
                    )
        elif myTeam == "red":
            return np.array(
                    flatMap(PlayerState.posToList, reds[me])
                    + flatMap(PlayerState.posToList, reds[:me])
                    + flatMap(PlayerState.posToList, reds[me+1:])
                    + flatMap(PlayerState.posToList, blues)
                    + flatMap(BallState.posToList, balls)
                    )
        else:
            raise ValueError

    def actToNp(myTeam, me):
        if myTeam == "blue":
            return np.array(
                    flatMap(PlayerState.actToList, blues[me])
                    + flatMap(PlayerState.actToList, blues[:me])
                    + flatMap(PlayerState.actToList, blues[me+1:])
                    + flatMap(PlayerState.actToList, reds)
                    )
        elif myTeam == "red":
            return np.array(
                    flatMap(PlayerState.actToList, reds[me])
                    + flatMap(PlayerState.actToList, reds[:me])
                    + flatMap(PlayerState.actToList, reds[me+1:])
                    + flatMap(PlayerState.actToList, blues)
                    )
        else:
            raise ValueError

@dataclass
class Game:
    frames: List[Frame]

    def toNp(myTeam, me):
        return np.array([f.posToNp(myTeam, me) for f in frames]), np.array([f.actToNp(myTeam, me) for f in frames])

    @staticmethod
    def load(filename):
        pass

    @staticmethod
    def save(filename):
        pass
