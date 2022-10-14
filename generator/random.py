import os
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

        # Log setting
        self.directory_name = 'random'
        os.makedirs(f'./models/generator/{self.directory_name}', exist_ok=True)
        os.makedirs(f'./models/discriminator/{self.directory_name}', exist_ok=True)
        os.makedirs(f'./logs/generator/', exist_ok=True)
        os.makedirs(f'./logs/discriminator/', exist_ok=True)

        
        self.seed = 1234
        self.rng = np.random.RandomState(self.seed)

        # Extract environment information
        self.env = env
        self.act_dim = env.action_space.n # 9

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
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

    def run(self, total_episodes):
        t_so_far = 0 # Episodes simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_episodes:
            batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += len(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Update Discriminator Network
            self.env.update_discriminator()

            # Print a summary of our training so far
            self._log_summary()
                
        self.save_log_to_csv(end=True)

    def rollout(self):
        # Batch data. For more details, check function header.
        batch_rews = []
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
                episode_steps += 1
                time_steps += 1

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
            
        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_lens

    def _init_hyperparameters(self, hyperparameters):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.episode_per_batch = 4                 # Number of timesteps to run per batch
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

    def _log_summary(self):
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
        avg_discriminator_loss = np.mean([losses.float().mean() for losses in self.env.discriminator.logger['total_loss']])
        avg_discriminator_entropy = np.mean([losses.float().mean() for losses in self.env.discriminator.logger['entropy']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_discriminator_loss = str(round(avg_discriminator_loss, 5))
        avg_discriminator_entropy = str(round(avg_discriminator_entropy, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Discriminator Loss: {avg_discriminator_loss}", flush=True)
        print(f"Average Discriminator Entropy: {avg_discriminator_entropy}", flush=True)
        print(f"Episodes So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
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

        if end:
            with open(f'./logs/generator/{self.directory_name}.csv') as f:
                count = 0
                data = []
                previous_i = 0
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

            with open(f'./logs/generator/new_{self.directory_name}.csv', mode='w') as f:
                f.write('\n'.join(data))

            print(f'Saved to ./logs/generator/new_{self.directory_name}.csv')
