import playeraction

class Player:
    def __init__(self,i_position,i_velocity,i_acceleration):
        self.pos = i_position
        self.vel = i_velocity
        self.acc = i_acceleration
        #Action is weird, not nice. Maybe make into a class?
        self.current_action = 0


class Ball:
    def __init__(self,initial_position,initial_velocity):
        self.pos = initial_position
        self.vel = initial_velocity



class GoalPost:
    def __init__(self,initial_position,initial_velocity):
        self.pos = initial_position
        self.vel = initial_velocity

class GameSim:
    def __init__(self,players,balls,goalposts):
        self.players = players

    def step(self):
        #DoStuff
