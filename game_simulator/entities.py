import gameparams
import playeraction

# handles player indexing
curr_idx = -1

def get_idx():
    global curr_idx
    curr_idx += 1
    return curr_idx

# Base class for any entity, stores position, velocity, acceleration.
class Entity:
    def __init__(self, initial_position, initial_velocity, initial_acceleration):
        self.pos = initial_position
        self.vel = initial_velocity
        self.acc = initial_acceleration


class Player(Entity):
    def __init__(self, initial_position, initial_velocity, initial_acceleration = np.zeros((2, 1))):
        # TODO: NO NEWKICK

        # Initialise positional parameters
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration)

        # Set the reset position, TODO: Doesn't make any sense, isn't called later
        self.default_position = initial_position
        self.idx = get_idx()

        #Action is weird, not nice. Maybe make into a class?
        self.current_action = playeraction.Action()

        # player properties
        self.colour = colour
        self.bouncingquotient = gameparams.playerbouncing
        self.radius = gameparams.playerradius
        self.mass = 1 / gameparams.playerinvmass

    def reset(self):
        # position vectors
        self.pos = np.array([pitchcornerx + (np.random.random_sample())*580, pitchcornery + (np.random.random_sample())*200]).astype(float)

        # velocity and speed
        self.velocity = np.array([0, 0])
        self.speed = 0

        # acceleration
        self.acc = np.array([0, 0])

        # player properties
        self.current_action = playeraction.Action()


class Ball(Entity):
    def __init__(self, initial_position, initial_velocity, initial_acceleration = np.zeros((2, 1))):
        # Initialise positional parameters
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration)

        # sets default positions
        self.default_position = initial_position

        # ball properties
        self.bouncingquotient = gameparams.ballbouncing
        self.radius = gameparams.ballradius
        self.mass = 1 / gameparams.ballinvmass


class GoalPost(Entity):
    def __init__(self, initial_position, initial_velocity = 0, initial_acceleration = np.zeros((2, 1))):
        # Initialise positional parameters
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration)

        self.bouncingquotient = gameparams.goalpostbouncingquotient
        self.radius = gameparams.goalpostradius


class CentreCircleBlock(Entity):
    def __init__(self, initial_position, initial_velocity = 0, initial_acceleration = np.zeros((2, 1))):
        # Initialise positional parameters
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration)

        self.bouncingquotient = 0
        self.radius = gameparams.centrecircleradius
