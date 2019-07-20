from game_simulator import playeraction
from move_displayer import movedisplayer
from random import randrange
import random
import numpy as np
import torch

class ACAgent():
    # Agent that works off of a actor-critic model
    def __init__(self, network, team, method = "random", debug_surf = None):
        self.network = network
        self.team = team
        self.method = method
        self.debug_surf = debug_surf

    def getAction(self, frame):
        actionpred, win_prob = self.network(torch.FloatTensor(frame.posToNp(self.team)))
        if self.method == "random":
            action = playeraction.Action(np.random.choice(len(actionpred), p = actionpred.detach().numpy()))
        elif self.method == "max":
            action = playeraction.Action(int(np.argmax(actionpred.detach().numpy())))
        else:
            raise ValueError
        if self.team == "red":
            pass
        elif self.team == "blue":
            action = action.flipped()
        else:
            raise ValueError
        if self.debug_surf:
            if self.team == "red":
                action_probs = actionpred.detach().numpy()
            elif self.team == "blue":
                action_probs = actionpred.detach().numpy()[[0,5,6,7,8,1,2,3,4]]
            else:
                raise ValueError
            self.debug_surf.drawMove([(x + y) / 2 for x, y in zip(*([action_probs] * 2))], action.dir_idx, self.team, float(win_prob))
        return action
