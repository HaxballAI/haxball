from game_simulator import playeraction

from random import randrange

class RetardedAgent():
    # A really clever agent that only returns random commands
    def __init__(self):
        self.is_learning = 0

    def getRawAction(self, gui):
        # Returns raw action of the agent based on the key presses queried from
        # the gui. Returns (dir_idx, kicking_state)
        return randrange(9), randrange(2)

    def getAction(self, gui):
        return playeraction.Action(*self.getRawAction(gui))
