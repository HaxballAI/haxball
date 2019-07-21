from game_simulator import playeraction
from game_simulator.gamesimengine import GameSimEngine
from game_log import log

import numpy as np
import time

class GameSim(GameSimEngine):
    def __init__(self, red_player_count, blue_player_count, ball_count, extraParams = {}, seed = -1):
        GameSimEngine.__init__(self, red_player_count, blue_player_count, ball_count, extraParams, seed)

        # Sets extra information to do with. Probably a convention that I am
        # not following here.
        if "printDebug" in extraParams:
            self.printDebug = extraParams["printDebug"]
        else:
            self.printDebug = False

        if "printDebugFreq" in extraParams:
            self.printDebugFreq = extraParams["printDebugFreq"]
        else:
            self.printDebugFreq = 600

    def getFeedback(self):
        # Gives feedback about the state of the game
        if self.printDebug:
            # Print some stuff
            print("Frame {}, score R-B: {}-{}".format(self.frames, self.red_score, self.blue_score))
            if self.was_point_scored:
                print("    A point was scored, nice!")
            for obj in self.reds:
                print("    red player at: {:.3f}; {:.3f} with velocity {:.3f}; {:.3f}".format(obj.pos[0], obj.pos[1], obj.vel[0], obj.vel[1]))
            for obj in self.blues:
                print("    blue player at: {:.3f}; {:.3f} with velocity {:.3f}; {:.3f}".format(obj.pos[0], obj.pos[1], obj.vel[0], obj.vel[1]))
            for obj in self.balls:
                print("    ball at: {:.3f}; {:.3f} with velocity {:.3f}; {:.3f}\n".format(obj.pos[0], obj.pos[1], obj.vel[0], obj.vel[1]))
        return

    def log(self):
        return log.Frame(
                blues = [p.log() for p in self.blues],
                reds  = [p.log() for p in self.reds ],
                balls = [b.log() for b in self.balls],
                )

    def giveCommands(self, actions):
        # Gives commands to all the controllable entities in the game in the form of a list pf commands.
        # Each command is a tuple of size 2 specifying direction (18 possible states) and then the kick state.
        # The position of the command in the list determines which entity the command is sent to.
        # TODO: Pls complete this function

        # NOTE: reds come before blues.
        for i in range(len(self.players)):
            self.players[i].current_action = actions[i]

    def step(self):
        self.frames += 1
        self.was_point_scored = 0

        # Update positions
        self.updatePositions()
        # Handle collisions
        self.detectAndResolveCollisions()

        # Update the score of the game
        if self.auto_score:
            self.updateScore("random")

        if self.printDebug and self.frames % self.printDebugFreq == 0:
            self.getFeedback()

        return

    def run(self, disp, agents):
        while True:
            # Query each agent on what commands should be sent to the game simulator
            self.giveCommands([a.getAction(self.log()) for a in agents])

            self.step()

            # Update the graphical interface canvas
            disp.drawFrame(self.log())

            disp.getInput()

            if disp.rip:
                disp.shutdown()
                break
