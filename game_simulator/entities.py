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
    def __init__(self, initial_position, initial_velocity, initial_acceleration, radius, bouncingquotient):
        self.pos = initial_position
        self.vel = initial_velocity
        self.acc = initial_acceleration

        self.radius = radius
        self.bouncingquotient = bouncingquotient

    def updatePosition(self):
        # Updates the position of the entity. Doesn't include any step duration for
        # whatever reason. God help us all
        self.velocity *= gameparams.balldamping
        self.pos += self.velocity


class Player(Entity):
    def __init__(self, initial_position, initial_velocity, initial_acceleration = np.zeros((2, 1))):
        # TODO: NO NEWKICK

        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration,
                        gameparams.playerradius, gameparams.playerbouncing)

        # Set the reset position, TODO: Doesn't make any sense, isn't called later
        self.default_position = initial_position
        self.idx = get_idx()

        #Action is weird, not nice. Maybe make into a class?
        self.current_action = playeraction.Action()

        # player properties
        self.colour = colour
        self.mass = 1 / gameparams.playerinvmass

    def updatePosition(self):
        self.velocity *= gameparams.balldamping
        self.pos += self.velocity

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
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration, gameparams.ballradius, gameparams.ballbouncing)

        # sets default positions
        self.default_position = initial_position

        # ball properties
        self.mass = 1 / gameparams.ballinvmass


class GoalPost(Entity):
    def __init__(self, initial_position, initial_velocity = 0, initial_acceleration = np.zeros((2, 1))):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration, gameparams.goalpostradius, gameparams.goalpostbouncingquotient)


class CentreCircleBlock(Entity):
    def __init__(self, initial_position, initial_velocity = 0, initial_acceleration = np.zeros((2, 1))):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration, gameparams.centrecircleradius, 0)
