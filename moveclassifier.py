import torch
import itertools
import numpy as np
from data_handler import datahandler
from game_simulator import gameparams as gp
data_handler = datahandler.DataHandler("rawsaved_games.dat")
data_handler.loadFileToBuffer()
loaded_data=data_handler.buffer

#print(a[0][0])
# N is batch size; D_in is input dimension;
# H is hidden dimension: D_out is output dimension.
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    if isinstance(S[0], type(np.array([])) ):
        return flatten(S[0].tolist()) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


redpositions = []
bluepositions = []
blueframe = []
redactions = []
blueactions = []
for game in loaded_data:
    for frame in game:
        redpositions.append(flatten(frame[0]))
        redactions.append(frame[1][0])
        for object in frame:
            blueframe.append([gp.rotatePos(object[0]),gp.rotateVel(object[1])])
        bluepositions.append(flatten(blueframe))
        blueactions.append(frame[1][1])
        blueframe = []

positions = redpositions + bluepositions
actions = redactions + blueactions
data = [positions,actions]
print(data)
