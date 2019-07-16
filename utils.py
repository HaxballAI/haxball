import numpy as np

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    if isinstance(S[0], type(np.array([]))):
        return flatten(S[0].tolist()) + flatten(S[1:])
    return S[:1] + flatten(S[1:])
