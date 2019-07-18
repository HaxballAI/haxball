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

    def getAction(self, frame, method = "random", give_debug_surf = False):
        movepred, kickpred , _ = self.network(torch.FloatTensor(frame.posToNp(self.team)))
        if method == "random":
            move = np.random.choice(len(movepred), p = movepred.detach().numpy() )
        elif method == "max":
            move = int(np.argmax(movepred.detach().numpy()))
        else:
            raise ValueError
        p_kick = float(kickpred[0])
        kick = np.random.choice([False, True], p = [1 - p_kick, p_kick])
        action = playeraction.Action(move, kick)
        if self.team == "red":
            pass
        elif self.team == "blue":
            action = action.flipped()
        else:
            raise ValueError
        if give_debug_surf:
            if self.team == "red":
                move_probs = movepred.detach().numpy()
            elif self.team == "blue":
                move_probs = movepred.detach().numpy()[[0,5,6,7,8,1,2,3,4]]
            else:
                raise ValueError
            debug_surf = movedisplayer.drawMove(move_probs, action.dir_idx, self.team) #TODO: Pass win_prob in here
            return action, debug_surf
        else:
            return action
