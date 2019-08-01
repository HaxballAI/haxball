import torch
import torch.nn.functional as F

class Policy(torch.nn.Module):
    def __init__(self, D_in = 12, D_hid = 50, D_out = 10):
        super(Policy, self).__init__()
        self.affine1 = torch.nn.Linear(D_in, D_hid)
        self.move_head = torch.nn.Linear(D_hid, D_out - 1)
        self.kick_head = torch.nn.Linear(D_hid, 1)
        self.value_head = torch.nn.Linear(D_hid, 1)

    def forward(self, x):
        y = F.relu(self.affine1(x))
        moveprobs = F.softmax(self.move_head(y), dim=-1)
        kickprob = torch.nn.Sigmoid()(self.kick_head(y))
        winprob = torch.nn.Sigmoid()(self.value_head(y))
        return moveprobs, kickprob, winprob

class multiplayer_actor(torch.nn.Module):
    def __init__(self, num_players = 4, , D_hid = 50):
        super(GregPolicy2, self).__init__()
        #inputs are 4 for the ball, 4 for each player.
        D_in = 4 + 4 * num_players

        self.hidden_1 = torch.nn.Linear(D_in, D_hid)
        self.hidden_2 = torch.nn.Linear(D_hid, D_hid)
        self.action_signal = torch.nn.Linear(D_hid, 18)

    def forward(self, x):
        hidden_1 = F.relu(self.hidden_1(x))
        hidden_2 = F.relu(self.hidden_2(hidden_1))
        policy = F.softmax(self.action_signal(hidden_2))

        return policy

class multiplayer_critic(torch.nn.Module):
    def __init__(self, num_players = 4, , D_hid = 50):
        super(GregPolicy2, self).__init__()
        # inputs are 4 for the ball, 4 for each player, and 10 to one-hot encode the action of each player in that frame.
        D_in = 4 + 4 * num_players + 10 * num_players

        self.critic_hid_1 = torch.nn.Linear(D_in, D_hid)
        self.critic_hid_2 = torch.nn.Linear(D_in, D_hid)
        self.critic_Q = torch.nn.Linear(D_hid, 18)

    def forward(self, x):
        critic_hidden_1 = F.relu(self.critic_hid_1(x))
        critic_hidden_2 = F.relu(self.critic_hid_2(critic_hidden_1))
        critic_Q = self.critic_Q(critic_hidden_2)
        # critic_Q are the estimated Q values for each possible action for first player encoded, given the state and the actions of its teammates
        return critic_Q


class GregPolicy(torch.nn.Module):
    def __init__(self, D_hid = 80):
        super(GregPolicy, self).__init__()
        self.affine_actor_1 = torch.nn.Linear(12, D_hid)
        self.affine_actor_2 = torch.nn.Linear(D_hid, D_hid)
        self.affine_critic = torch.nn.Linear(12, D_hid)
        self.move_head = torch.nn.Linear(D_hid, 9)
        self.kick_head = torch.nn.Linear(D_hid, 1)
        self.value_head = torch.nn.Linear(D_hid, 1)

    def forward(self, x):
        y_actor_1 = F.relu(self.affine_actor_1(x))
        y_actor_2 = F.relu(self.affine_actor_2(y_actor_1))
        y_critic = F.relu(self.affine_critic(x))
        moveprobs = F.softmax(self.move_head(y_actor_2), dim=-1)
        kickprob = torch.nn.Sigmoid()(self.kick_head(y_actor_2))
        winprob = self.value_head(y_critic)
        return moveprobs, kickprob, winprob

class GregPolicy2(torch.nn.Module):
    def __init__(self, D_hid = 50):
        super(GregPolicy2, self).__init__()
        self.affine_actor_1 = torch.nn.Linear(12, D_hid)
        self.affine_actor_2 = torch.nn.Linear(D_hid, D_hid)
        self.affine_critic_1 = torch.nn.Linear(12, D_hid)
        self.affine_critic_2 = torch.nn.Linear(D_hid, D_hid)
        self.move_head = torch.nn.Linear(D_hid, 9)
        self.kick_head = torch.nn.Linear(D_hid, 1)
        self.value_head = torch.nn.Linear(D_hid, 1)

    def forward(self, x):
        y_actor_1 = F.relu(self.affine_actor_1(x))
        y_actor_2 = F.relu(self.affine_actor_2(y_actor_1))
        y_critic_1 = F.relu(self.affine_critic_1(x))
        y_critic_2 = F.relu(self.affine_critic_2(y_critic_1))
        moveprobs = F.softmax(self.move_head(y_actor_2), dim=-1)
        kickprob = torch.nn.Sigmoid()(self.kick_head(y_actor_2))
        winprob = self.value_head(y_critic_2)
        return moveprobs, kickprob, winprob
