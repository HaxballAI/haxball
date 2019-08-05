import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from gym_haxball.parallel_env import SubprocVecEnv

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class PPOTrainer:
    def __init__(self, model, env_constructor, worker_num,
                 learning_rate = 0.001, gamma = 0.99, tau = 0.95, critic_param = 0.5,
                 temperature = 0.001, clip_param = 0.1):
        self.model = model
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.critic_param = critic_param
        self.temperature = temperature
        self.envs = SubprocVecEnv([env_constructor() for i in range(workers)])
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        self.state = self.envs.reset()


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

            for _ in range(num_steps):
                self.state = torch.FloatTensor(self.state).to(device)
                a_probs, value = model(self.state)
                dist = Categorical(a_probs)

                action = dist.sample()
                next_state, reward, done, _ = self.envs.step(action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

                states.append(state)
                actions.append(action)

                state = next_state
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
            _, next_value = model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values

            self.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

    def compute_gae(next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages):
        for _ in range(ppo_epochs):
            # Splits data into minibatches.
            for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
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
