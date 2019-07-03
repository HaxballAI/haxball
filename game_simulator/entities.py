from game_simulator import gameparams
from game_simulator import playeraction

import numpy as np


# handles player indexing
curr_idx = -1

def get_idx():
    global curr_idx
    curr_idx += 1
    return curr_idx

# Base class for any entity, stores position, velocity, acceleration.
class Entity:
    def __init__(self, initial_position, initial_velocity, initial_acceleration, radius, bouncingquotient):
        self.pos = np.array(initial_position)
        self.vel = np.array(initial_velocity)
        self.acc = np.array(initial_acceleration)

        self.radius = radius
        self.bouncingquotient = bouncingquotient

    # Get the Euclidian distance from self to obj
    def getDistanceTo(self, obj):
        return np.linalg.norm(obj.pos - self.pos)

    # Get the normalised vector pointing from self to obj
    def getDirectionTo(self, obj):
        return (obj.pos - self.pos) / self.getDistanceTo(obj)


class Player(Entity):
    def __init__(self, team, initial_position, initial_velocity = np.zeros(2), initial_acceleration = np.zeros(2)):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration,
                        gameparams.playerradius, gameparams.playerbouncing)

        # Set the reset position, TODO: Doesn't make any sense, isn't called later
        self.default_position = initial_position
        self.idx = get_idx()

        # Initialise current action + can_kick which presents kick-spamming
        self.current_action = playeraction.Action()
        self.can_kick = True

        # player properties
        self.team = team
        self.mass = 1 / gameparams.playerinvmass

    def updatePosition(self):
        # Updates the position of the player while taking the player input into account
        # Damping effect when trying to kick the ball
        if self.current_action.isKicking() == True and self.can_kick == True:
            self.vel += self.current_action.getDirection() * gameparams.kickaccel
        else:
            self.vel += self.current_action.getDirection() * gameparams.accel

        self.vel *= gameparams.playerdamping
        self.pos += self.vel

    def reset(self):
        # TODO: Make the positional initialisation better...
        # positional parameters
        self.pos = np.array([gameparams.pitchcornerx + (np.random.random_sample())*580, gameparams.pitchcornery + (np.random.random_sample())*200]).astype(float)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)

        # Set the action to default action state
        self.current_action = playeraction.Action()


class Ball(Entity):
    def __init__(self, initial_position, initial_velocity = np.zeros(2), initial_acceleration = np.zeros(2)):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration, gameparams.ballradius, gameparams.ballbouncing)

        # ball properties
        self.mass = 1 / gameparams.ballinvmass
        self.inv_mass = gameparams.ballinvmass

    def updatePosition(self):
        # Updates the position of the entity. Doesn't include any step duration for
        # whatever reason. God help us all
        self.vel *= gameparams.balldamping
        self.pos += self.vel

    def reset(self):
        # positional parameters
        self.pos = np.array([gameparams.pitchcornerx + (np.random.random_sample())*580, gameparams.pitchcornery + (np.random.random_sample())*200]).astype(float)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)


class GoalPost(Entity):
    def __init__(self, initial_position, initial_velocity = np.zeros(2), initial_acceleration = np.zeros(2)):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration, gameparams.goalpostradius, gameparams.goalpostbouncingquotient)


class CentreCircleBlock(Entity):
    def __init__(self, initial_position, initial_velocity = np.zeros(2), initial_acceleration = np.zeros(2)):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration, gameparams.centrecircleradius, 0)
