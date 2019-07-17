from game_simulator import playeraction
from move_displayer import movedisplayer
from random import randrange

class ACAgent():
    # A really clever agent that only returns random commands
    def __init__(self, network):
        self.network = network

    def getRawAction(self, state, give_debug_surf = False):
        # Returns raw action of the agent based on the key presses queried from
        # the gui. Returns (dir_idx, kicking_state)
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
        return playeraction.Action(*self.network.getRawAction(state))
