import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gym_haxball.parallel_env as vecenv

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")



class PPOTrainer:
    def __init__(self, model, env_constructor, worker_num,
                 learning_rate =3e-4, gamma = 1 - 3e-3, tau = 0.95, critic_param = 0.5,
                 temperature = 1e-3, clip_param = 0.1, ppo_epochs = 4,
                 mini_batch_size = 5, parallelise = True):
        self.model = model.to(device)
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.critic_param = critic_param
        self.temperature = temperature
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_param = clip_param
        if parallelise:
            self.envs = vecenv.SubprocVecEnv([env_constructor for i in range(worker_num)])
        else:
            self.envs = vecenv.BadVecEnv([env_constructor for i in range(worker_num)])
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        self.state = torch.FloatTensor(self.envs.reset()).to(device)

        print("Done.")




    def train(self, steps):
        frame_idx = 0
        early_stop = False
        while frame_idx < steps and not early_stop:

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy = 0

            for _ in range(32):
                a_probs, value = self.model(self.state)
                dist = Categorical(a_probs)

                action = dist.sample()
                next_state, reward, done, _ = self.envs.step(action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy = entropy + dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

                states.append(self.state)
                actions.append(action)

                state = torch.FloatTensor(next_state).to(device)
                frame_idx += 1

                # Might be nice to have this sort of visauliser.
                '''
                if frame_idx % 1000 == 0:
                    test_reward = np.mean([test_env() for _ in range(10)])
                    test_rewards.append(test_reward)
                    plot(frame_idx, test_rewards)
                    if test_reward > threshold_reward: early_stop = True
                '''


            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values

            self.ppo_update(self.ppo_epochs, self.mini_batch_size, states,
                            actions, log_probs, returns, advantage)

    def compute_gae(self,next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self,mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            # Need to be concerned probably about : only being used on states.
            yield states[rand_ids, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]

    def ppo_update(self,ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages):
        for _ in range(ppo_epochs):
            # Splits data into minibatches.
            for state, action, old_log_probs, return_, advantage \
             in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                a_prob, value = self.model(state)
                dist = Categorical(a_prob)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                # Computes actor loss.
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = (self.critic_param * critic_loss) \
                     + actor_loss \
                     - (self.temperature * entropy)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

#    def __del__():
        #self.envs.close()
