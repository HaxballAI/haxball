from game_simulator import playeraction

class HumanAgent():
    def __init__(self, keybindings, gui):
        # Keybindings is a list containing the strings of the keybindings

        # Movement keys of the agent in the following order: up, right, down, left
        self.movement_keys = keybindings[0:4]

        # Kicking key for the agent
        self.kick = keybindings[4]

        self.gui = gui

    def getAction(self, frame = None):
        # Ignore frame
        _ = frame
        # Returns raw action of the agent based on the key presses queried from
        # the gui. Returns (dir_idx, kicking_state)
        movements = [self.gui.isKeyPressed(key) for key in self.movement_keys]
        movements[0], movements[2] = movements[2], movements[0]

        raw_action = playeraction.binaryToRaw(*movements, self.gui.isKeyPressed(self.kick))
        action = playeraction.Action(*raw_action)

        return action
