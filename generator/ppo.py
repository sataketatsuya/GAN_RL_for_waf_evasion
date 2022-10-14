"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import re
import os
import gym
import gym_waf
import time

import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.categorical import Categorical

class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, policy_class, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # Make sure the environment is compatible with our code
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Discrete)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Log setting
        self.directory_name = 'FFN'
        os.makedirs(f'./models/generator/{self.directory_name}', exist_ok=True)
        os.makedirs(f'./models/discriminator/{self.directory_name}', exist_ok=True)
        os.makedirs(f'./logs/generator/', exist_ok=True)
        os.makedirs(f'./logs/discriminator/', exist_ok=True)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0] # 
        self.act_dim = env.action_space.n # 9

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

         # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim, self.device)                                                   # ALG STEP 1
        self.actor.to(self.device)

        self.critic = policy_class(self.obs_dim, 1, self.device)
        self.critic.to(self.device)

        # Initialize optimizers for actor and critic
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr}
        ])

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'total_loss': [],       # losses of actor network in current iteration
            'entropy': [],          # entropy mean
        }

        self.current_episode = 0
        self.log = {
            'episode': [],
            'time_steps': [],
            'steps': [],
            'win': [],
            'mean_reward': [],
            'original_payload': [],
            'payload': [],
        }

    def learn(self, total_episodes):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_episodes - the total number of episodes to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.episode_per_batch} episodes per batch for a total of {total_episodes} episodes")
        t_so_far = 0 # Episodes simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_episodes:                                                                       # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

            # Calculate how many episodes we collected this batch
            t_so_far += len(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging episodes so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _, _ = self.evaluate(batch_obs, batch_acts) # Critic evaluatation
            A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of 
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                # Calculate V_pi and pi_theta(a_t | s_t)
                V, curr_log_probs, dist_entropy = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation: 
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                importance_ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = importance_ratios * A_k
                surr2 = torch.clamp(importance_ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                total_loss = actor_loss + self.vf_coefficient*critic_loss + self.en_coefficient*dist_entropy.mean()

                # Calculate gradients and perform backward propagation for actor network and critic network
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                # Log actor loss
                self.logger['total_loss'].append(total_loss.cpu().detach())
                self.logger['entropy'].append(dist_entropy.mean().cpu().detach())

            if self.env.check_discriminator:
                # Update Discriminator Network
                self.env.update_discriminator()

            # Print a summary of our training so far
            
            self._log_summary(list(batch_acts.cpu().detach().numpy().copy()))

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), f'./models/generator/{self.directory_name}/ppo_actor.pth')
                torch.save(self.critic.state_dict(), f'./models/generator/{self.directory_name}/ppo_critic.pth')

                if self.env.check_discriminator:
                    torch.save(self.env.discriminator.actor.state_dict(), f'./models/discriminator/{self.directory_name}/ppo_actor.pth')
                    torch.save(self.env.discriminator.critic.state_dict(), f'./models/discriminator/{self.directory_name}/ppo_critic.pth')

                self.save_log_to_csv()

        self.save_log_to_csv(end=True)

    def rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.

            Parameters:
                None

            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0 # Keeps track of how many episodes we've run so far this batch
        time_steps = self.logger['t_so_far']

        # Run an episode for a maximum of max_timesteps_per_episode timesteps
        for ep_t in range(self.episode_per_batch):
            # Keep simulating until we've run more than or equal to specified timesteps per batch
            ep_rews = [] # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation. 
            obs = self.env.reset()
            done = False
            episode_steps = 0
            self.current_episode += 1

            while not done:
                # If render is specified, render the environment
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                t += 1 # Increment timesteps ran this batch so far
                episode_steps += 1
                time_steps += 1

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                obs, rew, done, infos = self.env.step(action)

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    # Log the episode info
                    self.log['episode'].append(self.current_episode)
                    self.log['time_steps'].append(time_steps)
                    self.log['steps'].append(episode_steps)
                    self.log['mean_reward'].append(np.mean(ep_rews))
                    self.log['win'].append(infos['win'])
                    self.log['original_payload'].append(infos['original'])
                    self.log['payload'].append(infos['payload'])
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
        batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

        return batch_rtgs

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        with torch.no_grad():
            # Convert observation to tensor if it's a numpy array
            obs = torch.FloatTensor(obs).to(self.device).clone()

        # Query the actor network for a mean action
        mean = self.actor(obs)

        # Create a distribution with the mean action
        # Note that this is equivalent to what used to be called multinomial
        dist = Categorical(F.softmax(mean, dim=-1))
        # print(F.softmax(mean, dim=-1))
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.item(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        with torch.no_grad():
            # Convert observation to tensor if it's a numpy array
            batch_obs = batch_obs.to(self.device).clone()
            batch_acts = batch_acts.to(self.device).clone()

        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)

        dist = Categorical(F.softmax(mean, dim=-1))
        log_probs = dist.log_prob(batch_acts)

        dist_entropy = dist.entropy()

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist_entropy

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
        self.episode_per_batch = 4800                 # Number of timesteps to run per batch
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

    def _log_summary(self, batch_acts):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['total_loss']])
        avg_entropy = np.mean([losses.float().mean() for losses in self.logger['entropy']])
        avg_discriminator_loss = np.mean([losses.float().mean() for losses in self.env.discriminator.logger['total_loss']])
        avg_discriminator_entropy = np.mean([losses.float().mean() for losses in self.env.discriminator.logger['entropy']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_entropy = str(round(avg_entropy, 5))
        avg_discriminator_loss = str(round(avg_discriminator_loss, 5))
        avg_discriminator_entropy = str(round(avg_discriminator_entropy, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Generator Loss: {avg_actor_loss}", flush=True)
        print(f"Average Generator Entropy: {avg_entropy}", flush=True)
        print(f"Average Discriminator Loss: {avg_discriminator_loss}", flush=True)
        print(f"Average Discriminator Entropy: {avg_discriminator_entropy}", flush=True)
        print(f"Episodes So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Generator action counter", Counter(batch_acts), flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['total_loss'] = []
        self.logger['entropy']    = []
        self.env.discriminator.logger['total_loss'] = []
        self.env.discriminator.logger['entropy'] = []

    def save_log_to_csv(self, end=False):
        # Save Generator's Logs
        df = pd.DataFrame()
        df['episode'] = self.log['episode']
        df['time_steps'] = self.log['time_steps']
        df['steps'] = self.log['steps']
        df['win'] = self.log['win']
        df['mean_reward'] = self.log['mean_reward']
        df['original_payload'] = self.log['original_payload']
        df['payload'] = self.log['payload']
        df.to_csv(f'./logs/generator/{self.directory_name}.csv')

        # Save Discriminator's Logs
        df = pd.DataFrame()
        df['episode'] = self.env.discriminator.log['episode']
        df['time_steps'] = self.env.discriminator.log['time_steps']
        df['payload'] = self.env.discriminator.log['payload']
        df['real_or_fake'] = self.env.discriminator.log['real_or_fake']
        df['label'] = self.env.discriminator.log['label']
        df['reward'] = self.env.discriminator.log['reward']
        df['predict'] = self.env.discriminator.log['predict']
        df['output_fake'] = self.env.discriminator.log['output_fake']
        df['output_real'] = self.env.discriminator.log['output_real']
        df.to_csv(f'./logs/discriminator/{self.directory_name}.csv')

        if end:
            self.trim_csv(f'./logs/generator/{self.directory_name}.csv')
            self.trim_csv(f'./logs/discriminator/{self.directory_name}.csv')

            print(f'Saved to ./logs/generator/{self.directory_name}.csv')
            print(f'Saved to ./logs/discriminator/{self.directory_name}.csv')

    def trim_csv(file_path):
        with open(file_path) as f:
            count = 0
            data = []
            index = 0
            for line in f:
                if count == 0:
                    data.append(line[:-1])
                    count = 1
                else:
                    split = line.split(',')
                    if re.fullmatch('[0-9]+', split[0]):
                        data.append(line[:-1])
                    else:
                        line = data[-1] + line
                        data[-1] = line[:-1]

        with open(file_path, mode='w') as f:
            f.write('\n'.join(data))
