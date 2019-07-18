from game_simulator import playeraction
from move_displayer import movedisplayer
from random import randrange
import random
import numpy as np
import torch

class ACAgent():
    # Agent that works off of a actor-critic model
    def __init__(self, network, team):
        self.network = network
        self.team = team

    def getRawAction(self, frame, method = "random", give_debug_surf = False):
        movepred, kickpred , _ = self.network(torch.FloatTensor(frame.posToNp(self.team)))
        if method == "random":
            move = np.random.choice(len(movepred), p = torch.nn.Softmax(dim = 0)(movepred).detach().numpy())
        elif method == "max":
            move = int(np.argmax(movepred.detach().numpy()))
        else:
            raise ValueError
        if self.team == "red":
            pass
        elif self.team == "blue":
            if move != 0:
                move = ((move + 3) % 8) + 1
        else:
            raise ValueError
        p_kick = float(kickpred[0])
        kick = np.random.choice([False, True], p = [1 - p_kick, p_kick])
        if give_debug_surf:
            debug_surf = movedisplayer.drawMove(torch.nn.Softmax(dim = 0)(movepred).detach().numpy(), move)
            return (move, kick), debug_surf
        else:
            return (move, kick)
