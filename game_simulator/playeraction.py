import numpy as np
import random

dirx = [0, 0, 1, 1, 1, 0, -1, -1, -1]
diry = [0, 1, 1, 0, -1, -1 ,-1, 0, 1]

class Action:
    def __init__(self, directionNumber = 0, isKicking = 0):
        self.kicking = isKicking
        self.dir_idx = directionNumber

        self.direction = np.array((dirx[self.dir_idx], diry[self.dir_idx])).astype("float")
        if directionNumber != 0:
            self.direction /= np.linalg.norm(self.direction)

    def isKicking(self):
        return self.kicking == 1

    def getDirection(self):
        # Returns the movement direction as a normalised vector
        return self.direction

    def rawAction(self):
        # Returns raw action for use in networks. A tuple of the kicking state (0 or 1)
        # and movement direction (from 0 to 8)
        return self.kicking, self.dir_idx

    def flipped(self):
        if self.dir_idx == 0:
            return Action(self.dir_idx, self.kicking)
        else:
            return Action(((self.dir_idx + 3) % 8) + 1, self.kicking)

def getRandomAction():
    return Action(random.randint(8), random.randint(1))

