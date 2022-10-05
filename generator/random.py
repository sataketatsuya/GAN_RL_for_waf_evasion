import gym
import gym_waf
import time

import numpy as np
import pandas as pd


class Random:
    def __init__(self, env, **hyperparameters):
        # Make sure the environment is compatible with our code
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Discrete)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)
        
        self.seed = 1234
        self.rng = np.random.RandomState(self.seed)

        # Extract environment information
        self.env = env
        self.act_dim = env.action_space.n # 9

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
        }
        
        self.current_episode = 0
        self.log = {
            'episode': [],
            'steps': [],
            'win': [],
            'mean_reward': [],
            'original_payload': [],
            'payload': [],
        }

    def run(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:
            batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Update Discriminator Network
            self.env.update_discriminator()

            # Print a summary of our training so far
            self._log_summary()
            
            if i_so_far % self.save_freq == 0:
                # Save Logs
                df = pd.DataFrame()
                df['episode'] = self.log['episode']
                df['steps'] = self.log['steps']
                df['win'] = self.log['win']
                df['mean_reward'] = self.log['mean_reward']
                df['original_payload'] = self.log['original_payload']
                df['payload'] = self.log['payload']

                df.to_csv('./logs/random.csv')

    def rollout(self):
        # Batch data. For more details, check function header.
        batch_acts = []
        batch_rews = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        ep_rews = [] # rewards collected per episode

        # Reset the environment. sNote that obs is short for observation. 
        obs = self.env.reset()
        done = False
        episode_steps = 0
        self.current_episode += 1

        # Run an episode for a maximum of max_timesteps_per_episode timesteps
        for ep_t in range(self.max_timesteps_per_episode):
            # If render is specified, render the environment
            if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                self.env.render()

            episode_steps += 1

            # Calculate action and make a step in the env. 
            # Note that rew is short for reward.
            action = self.rng.randint(0, self.act_dim)
            obs, rew, done, infos = self.env.step(action)

            # Track recent reward, action, and action log probability
            ep_rews.append(rew)

            # If the environment tells us the episode is terminated, break
            if done:
                # Log the episode info
                self.log['episode'].append(self.current_episode)
                self.log['steps'].append(episode_steps)
                self.log['mean_reward'].append(np.mean(ep_rews))
                self.log['win'].append(infos['win'])
                self.log['original_payload'].append(infos['original'])
                self.log['payload'].append(infos['payload'])
                break

        # Track episodic lengths and rewards
        batch_lens.append(ep_t + 1)
        batch_rews.append(ep_rews)
            
        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_lens

    def _init_hyperparameters(self, hyperparameters):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration

        # Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 100                            # How often we save in number of iterations

    def _log_summary(self):
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.env.logger['discriminator_losses'] = []
