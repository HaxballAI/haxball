from  game_simulator import gamsesim

class DuelEnviroment:

    def __init__(self, step_len = 15, max_steps = 200):
        self.step_len = step_len
        self.max_steps = max_steps

        self.game_sim = gamesim.GameSim(1,1,1)
        self.game_sim.resetMap()

        self.steps_since_reset = 0

    def step(self, red_action, blue_action):
        self.steps_since_reset += 1

        self.game_sim.giveCommands( [red_action , blue_action ] )

        state_action_pairs = self.game_sim.getState("raw sa pairs")

        for i in range(self.step_len):
            self.game_sim.step()
            goal = self.goalScorded()
            # If a goal is scored return instantly
            if goal == 1:
                return [state_action_pairs , True, 1 , -1]
            elif goal == -1:
                return [state_action_pairs, True, -1 , 1]

        # If no goal consider it a tie.
        if self.steps_since_reset >= self.max_steps:
            return [state_action_pairs , True, 0 , 0]
        else:
            return [state_action_pairs , False, 0 , 0]

    def reset(self):
        self.steps_since_reset = 0
        self.game_sim.resetMap()


    def goalScored(self):
        # Checks goals. Returns 1 for red, 0 for none, -1 for blue.
        goals = self.game_sim.checkGoals()
        if goals[0] > 0:
            return 1
        elif goals[1] > 0:
            return -1
        else return 0
