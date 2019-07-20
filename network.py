import torch
import torch.nn.functional as F

class Policy(torch.nn.Module):
    def __init__(self, D_in = 12, D_hid = 50, D_out = 19):
        super(Policy, self).__init__()
        self.affine1 = torch.nn.Linear(D_in, D_hid)
        self.move_head = torch.nn.Linear(D_hid, D_out - 1)
        self.value_head = torch.nn.Linear(D_hid, 1)

    def forward(self, x):
        y = F.relu(self.affine1(x))
        actionprobs = F.softmax(self.action_head(y), dim=-1)
        winprob = torch.nn.Sigmoid()(self.value_head(y))
        return actionprobs, winprob


class GregPolicy(torch.nn.Module):
    def __init__(self, D_hid = 80):
        super(GregPolicy, self).__init__()
        self.affine_actor = torch.nn.Linear(12, D_hid)
        self.affine_critic = torch.nn.Linear(12, D_hid)
        self.action_head = torch.nn.Linear(D_hid, 18)
        self.value_head = torch.nn.Linear(D_hid, 1)

    def forward(self, x):
        y_actor = F.relu(self.affine_actor(x))
        y_critic = F.relu(self.affine_critic(x))
        actionprobs = F.softmax(self.action_head(y_actor), dim=-1)
        winprob = self.value_head(y_critic)
        return actionprobs, winprob
