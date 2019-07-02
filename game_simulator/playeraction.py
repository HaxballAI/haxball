

class Action:
    def __init__(self, direction, isKicking):
        self.kick = isKicking
        self.dir = direction

    def isKicking(self):
        return self.kick

    def isMovingDir(self,direction):
        #Return true if directions is current one
        #Directions should be a string like "ul" for up-left for
        #ease of use
        continue

    def rawAction(self):
        #Returns raw action for use in networks.
        continue
