from game_simulator import playeraction

from random import randrange

class ACAgent():
    # A really clever agent that only returns random commands
    def __init__(self, network):
        self.network = network

    def getRawAction(self, state = []):
        # Returns raw action of the agent based on the key presses queried from
        # the gui. Returns (dir_idx, kicking_state)
        return self.network.getRawAction(state)

    def getAction(self, state = []):
        return playeraction.Action(*self.network.getRawAction(state))
