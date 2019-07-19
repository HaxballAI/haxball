from game_simulator import playeraction

from random import randrange

class RandomAgent():
    # A really clever agent that only returns random commands
    def __init__(self):
        pass

    def getAction(self, frame = None):
        # Ignore frame
        _ = frame
        # Returns raw action of the agent based on the key presses queried from
        # the gui. Returns (dir_idx, kicking_state)
        return playeraction.Action.randomAction()
