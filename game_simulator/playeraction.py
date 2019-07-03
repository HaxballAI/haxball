import numpy as np
import random

dirx = [0, 0, 1, 1, 1, 0, -1, -1, -1]
diry = [0, 1, 1, 0, -1, -1 ,-1, 0, 1]




class Action:
    def __init__(self, directionNumber = 0, isKicking = 0, canKick = 1):
        self.kicking = isKicking
        self.can_kick = canKick
        self.dir_idx = directionNumber

        self.direction = np.array((dirx[self.dir_idx], diry[self.dir_idx])).astype("float")
        if directionNumber != 0:
            self.direction /= np.linalg.norm(self.direction)


    def isKicking(self):
        return self.kicking

    def canKick(self):
        return self.can_kick

    def isMovingDir(self, direction):
        # Return true if directions is the current one
        # Directions should be a string like "ul" for up-left for
        # ease of use
        stringToNumber = {
            "still": 0,
            "u": 1,
            "ur": 2,
            "ru": 2,
            "r": 3,
            "dl": 4,
            "ld": 4,
            "d": 5,
            "dl": 6,
            "ld": 6,
            "l": 7,
            "ul": 8,
            "lu": 8
        }

        if (type(direction) == "str"):
            direction = stringToNumber[direction]

        return self.dir_idx == direction

    def getDirection(self):
        # Returns the movement direction as a normalised vector
        return self.direction

    def rawAction(self):
        # Returns raw action for use in networks. A tuple of the kicking state (0 or 1)
        # and movement direction (from 0 to 8)
        return (self.kicking, self.dir_idx)

def getRandomAction():
    return Action(random.randint(0,8), random.randint(0,1))
