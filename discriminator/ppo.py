import gym
import gym_waf
import time
import const

import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.categorical import Categorical

class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        self._init_hyperparameters(hyperparameters)

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = 2 # real or fake

        self.device = env.device

        self.actor = policy_class(self.obs_dim, self.act_dim, self.device)
        self.actor.to(self.device)

        self.critic = policy_class(self.obs_dim, 1, self.device)
        self.critic.to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr}
        ])
        
        self._reset_ep_memory()
        self._reset_batch()

        self.logger = {
            'total_loss': [],
            'entropy': [],
        }

        self.current_episode = 1
        self.time_steps = 0
        self.log = {
            'episode': [],
            'time_steps': [],
            'payload': [],
            'real_or_fake': [],
            'label': [],
            'reward': [],
            'predict': [],
            'output_fake': [],
            'output_real': [],
        }

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _reset_ep_memory(self):
        self._reset_ep_memory_fake()
        self._reset_ep_memory_real()

    def _reset_ep_memory_fake(self):
        self.ep_fake_reward = []

    def _reset_ep_memory_real(self):
        self.ep_real_reward = []

    def _reset_batch(self):
        self.batch_fake = {
            'obs': [],
            'acts': [],
            'output': [],
            'log_probs': [],
            'rews': [],
            'rtgs': [],
        }
        
        self.batch_real = {
            'obs': [],
            'acts': [],
            'output': [],
            'log_probs': [],
            'rews': [],
            'rtgs': [],
        }

    def update(self):
        batch_real_obs, batch_real_acts, batch_real_log_probs = self.batch_real['obs'], self.batch_real['acts'], self.batch_real['log_probs']
        batch_real_rtgs = self.compute_rtgs(self.batch_real['rews'])
        batch_fake_obs, batch_fake_acts, batch_fake_log_probs = self.batch_fake['obs'], self.batch_fake['acts'], self.batch_fake['log_probs']
        batch_fake_rtgs = self.compute_rtgs(self.batch_real['rews'])

        batch_obs = batch_real_obs + batch_fake_obs
        batch_acts = batch_real_acts + batch_fake_acts
        batch_log_probs = torch.FloatTensor(batch_real_log_probs + batch_fake_log_probs).to(self.device).clone()
        batch_rtgs = torch.cat((batch_real_rtgs, batch_fake_rtgs), 0)

        V, _, _ = self.evaluate(batch_obs, batch_acts)
        A_k = batch_rtgs - V.detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(self.n_updates_per_iteration):
            V, curr_log_probs, dist_entropy = self.evaluate(batch_obs, batch_acts)

            importance_ratios = torch.exp(curr_log_probs - batch_log_probs)

            surr1 = importance_ratios * A_k
            surr2 = torch.clamp(importance_ratios, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            total_loss = actor_loss + self.vf_coefficient*critic_loss + self.en_coefficient*dist_entropy.mean()
            
            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optimizer.step()

            self.logger['total_loss'].append(total_loss.cpu().detach())
            self.logger['entropy'].append(dist_entropy.mean().cpu().detach())

        self._reset_ep_memory()
        self._reset_batch()

    def score_real(self, obs, done):
        action, output, log_prob = self.get_action(obs)
        rew = self.step(action, label=1)

        self.batch_real['obs'].append(obs)
        self.batch_real['acts'].append(action)
        self.batch_real['output'].append(action)
        self.batch_real['log_probs'].append(log_prob)
        self.ep_real_reward.append(rew)

        self.log['reward'].append(rew)
        self.log['predict'].append(action.item())
        self.log['output_fake'].append(output[0].item())
        self.log['output_real'].append(output[1].item())

        if done:
            self.current_episode += 1
            self.batch_real['rews'].append(self.ep_real_reward)

            self._reset_ep_memory_real()

        return output

    def score_fake(self, obs, done):
        action, output, log_prob = self.get_action(obs)
        rew = self.step(action, label=0)

        self.batch_fake['obs'].append(obs)
        self.batch_fake['acts'].append(action)
        self.batch_fake['output'].append(action)
        self.batch_fake['log_probs'].append(log_prob)
        self.ep_fake_reward.append(rew)

        self.log['reward'].append(rew)
        self.log['predict'].append(action.item())
        self.log['output_fake'].append(output[0].item())
        self.log['output_real'].append(output[1].item())

        if done:
            self.batch_fake['rews'].append(self.ep_fake_reward)
            
            self._reset_ep_memory_fake()

        return output

    def step(self, action, label):
        if action == label:
            return const.DIS_POSITIVE
        else:
            return const.DIS_NEGATIVE

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

        return batch_rtgs

    def get_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device).clone()

        mean = F.softmax(self.actor(obs), dim=-1)

        dist = Categorical(mean)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.cpu().detach(), mean.detach(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        with torch.no_grad():
            batch_obs = torch.FloatTensor(batch_obs).to(self.device).clone()
            batch_acts = torch.FloatTensor(batch_acts).to(self.device).clone()

        V = self.critic(batch_obs).squeeze()

        mean = F.softmax(self.actor(batch_obs), dim=-1)

        dist = Categorical(mean)
        log_probs = dist.log_prob(batch_acts)

        dist_entropy = dist.entropy()

        return V, log_probs, dist_entropy
