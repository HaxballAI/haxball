import numpy as np
from torch.distributions import Categorical
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from game_log import log
import torch

class GameDataset(Dataset):
    def __init__(self, data_dir, game_indicies, norming = False):
        self.root_dir = data_dir
        games = [log.Game.load(data_dir + "/" + str(i)) for i in game_indicies]
        act_list = []
        state_list = []
        win_list = []
        for i in range(len(games)):
            if games[i].blue_goals == 0:
                assert games[i].red_goals == 1
                red_win = 1.0
            elif games[i].red_goals == 0:
                assert games[i].blue_goals == 1
                red_win = 0.0
            else:
                raise ValueError
            r_states, r_acts = games[i].toNp("red", 0, norming)
            act_list.append(r_acts)
            state_list.append(r_states)
            win_list.append( np.full(len(r_acts), red_win) )
            b_states, b_acts = games[i].toNp("blue", 0, norming)
            act_list.append(b_acts)
            state_list.append(b_states)
            win_list.append( np.full(len(r_acts), 1.0 - red_win) )
        self.actions = torch.LongTensor(np.concatenate(act_list))
        self.states = torch.FloatTensor(np.concatenate(state_list))
        self.wins = torch.FloatTensor(np.concatenate(win_list))
        self.data_len = len(self.wins)

    def __len__(self):
        return self.data_len

    def __getitem__(self,idx):
        return {'state': self.states[idx], 'action': self.actions[idx], 'won': self.wins[idx]}




def learnFromGames(model, data_dir, game_indicies, epoch_num, learning_rate, batch_size, norming = False):
    game_data = GameDataset(data_dir, game_indicies, norming)
    game_loader = DataLoader(game_data, batch_size=batch_size, shuffle = True)
    actioncriterion = torch.nn.CrossEntropyLoss(reduction = 'mean')
    wincriterion = torch.nn.MSELoss(reduction = 'mean')
    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i, data in enumerate(game_loader):
            act_pred, value_pred = model(data["state"])
            actor_loss = actioncriterion(act_pred, data["action"])
            critic_loss = wincriterion(value_pred.squeeze(), data["won"])
            loss = actor_loss + critic_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if i % 400 == 399:
                print(f"E: {epoch} Iter: {i} Loss : {running_loss}")
                running_loss = 0.0
    print("Training finished!")
