from game_simulator import playeraction


class HumanAgent():
    def __init__(self, keybindings):
        # Keybindings is a list containing the strings of the keybindings

        # Movement keys of the agent in the following order: up, right, down, left
        self.movement_keys = keybindings[0:4]

        # Kicking key for the agent
        self.kick = keybindings[4]

        self.is_human = 1

    def getRawAction(self, gui):
        # Returns raw action of the agent based on the key presses queried from
        # the gui. Returns (dir_idx, kicking_state)
        movements = [gui.isKeyPressed(key) for key in self.movement_keys]

        a, b, dir = 0, 0, 0
        if movements[0] + movements[2] == 1:
            a = 1 * movements[0] + 5 * movements[2]
        if movements[1] + movements[3] == 1:
            b = 3 * movements[1] + 7 * movements[3]

        if a == 1 and b == 7:
            dir = 8
        elif a != 0 and b != 0:
            dir = (a + b) // 2
        else:
            dir = max(a, b)

        return (dir, gui.isKeyPressed(self.kick))

    def getAction(self, gui):
        raw_action = self.getRawAction(gui)
        return playeraction.Action(raw_action[0], raw_action[1])
