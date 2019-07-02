

class Action:
    def __init__(self, direction, isKicking):
        self.kick = isKicking
        self.dir = direction

        def isKicking(self):
        return self.kick

    def isMovingDir(self,direction):
        if (self.dir[0,0]==0 and direction == none)
            or (self.dir[0,0]==1 and direction == "r")
            or (self.dir[0,0]==2 and direction in ["ur","ru"])
            or (self.dir[0,0]==3 and direction == "u")
            or (self.dir[0,0]==4 and direction in ["ul","lu"])
            or (self.dir[0,0]==5 and direction == "l")
            or (self.dir[0,0]==6 and direction in ["dl","ld"])
            or (self.dir[0,0]==7 and direction == "d")
            or (self.dir[0,0]==8 and direction in ["dr","rd"]):
            return True
        else:
            return False



        #Return true if directions is current one
        #Directions should be a string like "ul" for up-left for
        #ease of use
        continue

    def rawAction(self):
        #Returns raw action for use in networks.
        continue
