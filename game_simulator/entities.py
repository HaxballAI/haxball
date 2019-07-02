import gameparameters

# Base class for any entity, stores position, velocity, acceleration.
class Entity:
    def __init__(self, initial_position, initial_velocity, initial_acceleration):
        self.pos = initial_position
        self.vel = initial_velocity
        self.acc = initial_acceleration
        self.radius = 0


class Player(Entity):
    def __init__(self, initial_position, initial_velocity, initial_acceleration = np.zeros((2, 1))):
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration)
        #Action is weird, not nice. Maybe make into a class?
        self.current_action = 0


class Ball(Entity):
    def __init__(self, initial_position, initial_velocity, initial_acceleration = np.zeros((2, 1))):
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration)


class GoalPost(Entity):
    def __init__(self, initial_position, initial_velocity = np.zeros((2, 1)), initial_acceleration = np.zeros((2, 1))):
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration)
