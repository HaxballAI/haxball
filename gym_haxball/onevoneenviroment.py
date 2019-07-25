from game_simulator import gamesim
from game_simulator.playeraction import Action

class DuelEnviroment:

    def __init__(self, step_len = 15, max_steps = 200, rand_reset = True, reward_shape = False):
        self.step_len = step_len
        self.max_steps = max_steps
        self.rand_rest = rand_reset
        self.reward_shape = reward_shape

        self.game_sim = gamesim.GameSim(1,1,1, rand_reset = rand_reset)
        self.game_sim.resetMap()

        self.steps_since_reset = 0

    def getState(self):
        return self.game_sim.log()

    def step(self, red_action, blue_action):
        self.steps_since_reset += 1

        if  not isinstance(red_action, Action):
            red_action = Action(*red_action)
            blue_action = Action(*blue_action)

        self.game_sim.giveCommands( [red_action , blue_action ] )

        if self.reward_shape:
            red_bonus = -0.01 * red_action.isKicking()
            blue_bonus = -0.01 * blue_action.isKicking()
        else:
            red_bonus = 0
            blue_bonus = 0
        tie_penalty = 0
        if self.reward_shape:
            tie_penalty = 1

        state_action_pairs = self.game_sim.log()

        for i in range(self.step_len):
            self.game_sim.step()
            goal = self.goalScored()
            # If a goal is scored return instantly
            if goal == 1:
                return [state_action_pairs ,  (1 + red_bonus , -1 + blue_bonus) , True, {}]
            elif goal == -1:
                return [state_action_pairs,  (-1 + red_bonus , 1 + blue_bonus), True, {}]

        # If no goal consider it a tie.
        if self.steps_since_reset >= self.max_steps:
            return [state_action_pairs , (red_bonus - tie_penalty , blue_bonus - tie_penalty), True, {}]
        else:
            return [state_action_pairs ,  (red_bonus, blue_bonus) , False, {}]

    def reset(self):
        self.steps_since_reset = 0
        self.game_sim.resetMap()
        return self.game_sim.log()


    def goalScored(self):
        # Checks goals. Returns 1 for red, 0 for none, -1 for blue.
        goals = self.game_sim.checkGoals()
        if goals[0] > 0:
            return 1
        elif goals[1] > 0:
            return -1
        else:
            return 0
