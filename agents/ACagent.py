from game_simulator import playeraction
from move_displayer import movedisplayer
from random import randrange

class ACAgent():
    # Agent that works off of a actor-critic model
    def __init__(self, network):
        self.network = network

    def getRawAction(self, state, give_debug_surf = False)
        movepred, kickpred , _ = model(state)
        ran_move = np.random.choice(len(movepred), p = torch.nn.Softmax(dim = 0)(movepred).detach().numpy())
        p_kick = float(kickpred[0])
        ran_kick = np.random.choice([False, True], p = [1 - p_kick, p_kick])
        if give_debug_surf:
            debug_surf = movedisplayer.drawMove(torch.nn.Softmax(dim = 0)(movepred).detach().numpy(), ran_move)
            return (ran_move, ran_kick), debug_surf
        else:
            return (ran_move, ran_kick)

    def getAction(self, state = []):
        raise NotImplementedError
