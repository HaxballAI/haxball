from game_simulator import gameparams
from game_simulator import playeraction
from game_log import log

import numpy as np

# Base class for any entity, stores position, velocity, acceleration.
class Entity:
    def __init__(self, initial_position, initial_velocity, initial_acceleration, radius, bouncingquotient):
        self.pos = np.array(initial_position)
        self.vel = np.array(initial_velocity)

        self.radius = radius
        self.bouncingquotient = bouncingquotient

    # Get the Euclidian distance from self to obj
    def getDistanceTo(self, obj):
        return np.linalg.norm(obj.pos - self.pos)

    # Get the normalised vector pointing from self to obj
    def getDirectionTo(self, obj):
        return (obj.pos - self.pos) / self.getDistanceTo(obj)

class Player(Entity):
    def __init__(self, team, initial_position, initial_velocity = np.zeros(2), initial_acceleration = np.zeros(2), ):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration,
                        gameparams.playerradius, gameparams.playerbouncing)

        # Set the not random reset position
        self.default_position = initial_position

        # Initialise current action + can_kick which presents kick-spamming
        self.current_action = playeraction.Action()
        self.can_kick = True

        # Records the number of kicks the ball has made, used for reward shaping
        self.kick_count = 0

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

    def reset(self, reset_type):
        if reset_type == "random":
			# positional parameters
            self.pos = np.array([gameparams.pitchcornerx + (np.random.random_sample())*580, gameparams.pitchcornery + (np.random.random_sample())*200]).astype(float)
        elif reset_type == "default":
            self.pos = self.default_position
        else:
            raise ValueError("Passed a wrong reset type to a player")

        self.vel = np.zeros(2)

        self.kick_count = 0

        # Set the action to default action state
        self.current_action = playeraction.Action()

    def log(self):
        return log.PlayerState(*self.pos, *self.vel, self.current_action)

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

    def reset(self, reset_type):
        if reset_type == "random":
			# positional parameters
            self.pos = np.array([gameparams.pitchcornerx + (np.random.random_sample())*580, gameparams.pitchcornery + (np.random.random_sample())*200]).astype(float)
        elif reset_type == "default":
            self.pos = np.array([gameparams.pitchcornerx + gameparams.pitchwidth / 2, gameparams.pitchcornery + gameparams.pitchheight / 2])
        else:
            raise ValueError("Passed a wrong reset type to a ball")
        self.vel = np.zeros(2)

    def log(self):
        return log.BallState(self.pos[0], self.pos[1], self.vel[0], self.vel[1])

class GoalPost(Entity):
    def __init__(self, initial_position, initial_velocity = np.zeros(2), initial_acceleration = np.zeros(2)):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration, gameparams.goalpostradius, gameparams.goalpostbouncingquotient)

class CentreCircleBlock(Entity):
    def __init__(self, initial_position, initial_velocity = np.zeros(2), initial_acceleration = np.zeros(2)):
        # Initialise positional parameters, basic properties of the object
        Entity.__init__(self, initial_position, initial_velocity, initial_acceleration, gameparams.centrecircleradius, 0)
