

class Action:
    def __init__(self, directionNumber = 0, isKicking = 0):
        self.kicking = isKicking
        self.dir = directionNumber


    def isKicking(self):
        return self.kicking

    def isMovingDir(self,direction):
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

        return self.dir == direction

    def getDirection(self):
        # Returns the movement direction (from 0 to 8)
        return self.dir

    def rawAction(self):
        # Returns raw action for use in networks. A tuple of the kicking state (0 or 1)
        # and movement direction (from 0 to 8)
        return (self.isKicking(), self.getDirection())
