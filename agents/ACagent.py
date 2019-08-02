from game_simulator import playeraction
from move_displayer import movedisplayer
from random import randrange
import random
import numpy as np
import torch

class ACAgent():
    # Agent that works off of a actor-critic model
    def __init__(self, network, team, method = "random", debug_surf = None, accepts_normalised = False, value_is_prob = False):
        self.network = network
        self.team = team
        self.method = method
        self.debug_surf = debug_surf
        self.accepts_normalised = accepts_normalised
        self.value_is_prob = value_is_prob

    def getAction(self, frame):
        frame_tensor = torch.FloatTensor(frame.posToNp(self.team, 0, self.accepts_normalised))

        #    frame_tensor = frame_tensor.cuda()
        #if torch.cuda.is_available():

        output = self.network(frame_tensor)
        win_prob = output[-1]
        action_pred_data = output[0:-1]

        if not self.value_is_prob:
            #win_prob = torch.nn.Sigmoid()(win_prob)
            win_prob = (win_prob + 1.0)/2

        #if torch.cuda.is_available():
        #    movepred = movepred.cpu()


        action_data = []
        if self.method == "random":
            for i in range(len(action_pred_data)):
                if len(action_pred_data[i]) == 1:
                    p = action_pred_data[i].detach().numpy()[0]
                    action_data.append(np.random.choice([False, True], p = [1 - p, p]))
                else:
                    action_data.append(np.random.choice(len(action_pred_data[i]), p = action_pred_data[i].detach().numpy()))
        elif self.method == "max":
            for i in range(len(action_pred_data)):
                action_data.append(np.argmax(action_pred_data[i].detach().numpy()))
        else:
            raise ValueError
        action = playeraction.Action(*action_data)
        if self.team == "red":
            pass
        elif self.team == "blue":
            action = action.flipped()
        else:
            raise ValueError


        if self.debug_surf:
            move_probs = []
            if len(action_pred_data) == 1:
                temp = action_pred_data[0].detach().numpy()
                for i in range(len(temp)):
                    if i % 2 == 0:
                        move_probs.append(temp[i])
                    else:
                        move_probs[i // 2] = (move_probs[i // 2] + temp[i]) / 2
                move_probs = np.array(move_probs)
            elif len(action_pred_data) == 2:
                move_probs = action_pred_data[0].detach().numpy()
            else:
                raise ValueError

            if self.team == "blue":
                move_probs = move_probs[[0,5,6,7,8,1,2,3,4]]
            self.debug_surf.drawMove(move_probs, action.dir_idx, self.team, float(win_prob))
        return action
