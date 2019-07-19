import numpy as np
import random

dirx = [0, 0, 1, 1, 1, 0, -1, -1, -1]
diry = [0, 1, 1, 0, -1, -1 ,-1, 0, 1]

def rawToBinary(*x):
    if len(x) != 2:
        raise TypeError("Raw action should be a tuple of length 2")
    if x[0] < 0 or x[0] > 8 or x[1] < 0 or x[1] > 1:
        raise ValuError("Raw action is not of the correct format")

    ret = (0, 0, 0, 0, x[1])
    if x == 8 or x == 1 or x == 2:
        ret[0] = 1
    if 2 <= x[0] and x[0] <= 4:
        ret[1] = 1
    if 4 <= x[0] and x[0] <= 6:
        ret[2] = 1
    if 6 <= x[0] and x[0] <= 8:
        ret[3] = 1
    return ret

def binaryToRaw(*x):
    if len(x) != 5:
        raise TypeError("Binary action should be a tuple of length 5")
    for i in range(5):
        if x[i] < 0 or x[i] > 1:
            raise ValueError("Binary action should only have booleans as its members")

    a, b, dir = 0, 0, 0
    if x[0] + x[2] == 1:
        a = 1 * x[0] + 5 * x[2]
    if x[1] + x[3] == 1:
        b = 3 * x[1] + 7 * x[3]

    if a == 1 and b == 7:
        dir = 8
    elif a != 0 and b != 0:
        dir = (a + b) // 2
    else:
        dir = max(a, b)
    return (dir, x[4])

class Action:
    # Action stores all the useful info about an action that the player can have
    # Possibility to convert to the "raw" form of (movement_direction, kicking_state) or
    # binary form of (UP, RIGHT, DOWN, LEFT, IS_KICKING).
    # Also supports returning a vector of the movement direction
    def __init__(self, *action):
        if len(action) == 0:
            action = (0, 0)

        if len(action) == 1:
            self.dir_idx = action[0] >> 1
            self.kicking = action[0] & 1
        elif len(action) == 2:
            # Handle the case of (kicking_state, movement_direction)
            self.dir_idx = action[0]
            self.kicking = action[1]
        elif len(action) == 5:
            # Handle the case of a binary action tuple
            self.dir_idx, self.kicking = binaryToRaw(*action)
        else:
            raise ValueError

        self.direction = np.array((dirx[self.dir_idx], diry[self.dir_idx])).astype("float")
        if self.dir_idx != 0:
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

    def binaryAction(self):
        return rawToBinary(self.dir_idx, self.kicking)

    def singleAction(self):
        return (self.dir_idx << 1) & self.kicking

    def flipped(self):
        if self.dir_idx == 0:
            return Action(self.dir_idx, self.kicking)
        else:
            return Action(((self.dir_idx + 3) % 8) + 1, self.kicking)

    @staticmethod
    def randomAction():
        return Action(random.randint(9), random.randint(1))
