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
