"""
    This file is the executable for running PPO. It is based on this medium article: 
    https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gym
import gym_waf
import sys
import torch

import const

from generator.random import Random
from generator.arguments import get_args
from generator.ppo import PPO
from generator.network import FeedForwardNN
from generator.eval_policy import eval_policy

def train(env, hyperparameters, actor_model, critic_model):
    """
    Trains the model.

    Parameters:
    env - the environment to train on
    hyperparameters - a dict of hyperparameters to use, defined in main
    actor_model - the actor model to load in if we want to continue training
    critic_model - the critic model to load in if we want to continue training

    Return:
    None
    """
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=const.TOTAL_TIMESTEPS)
    
def random_run(env, **hyperparameters):
    print(f"Random Agent Run", flush=True)
    
    # Create a model for Rondom
    model = Random(env=env, **hyperparameters)
    
    model.run(total_timesteps=const.TOTAL_TIMESTEPS)

def main(args):
    """
    The main function to run.

    Parameters:
    args - the arguments parsed from command line

    Return:
    None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
        'timesteps_per_batch': const.TIMESTEP_PER_BATCH, 
        'max_timesteps_per_episode': const.MAX_TIMESTEPS_PER_EPISODE, 
        'gamma': const.GAMMA, 
        'n_updates_per_iteration': const.N_UPDATES_PER_ITERATION,
        'lr': const.G_LR, 
        'clip': const.CLIP,
        'render': False,
        'render_every_i': const.RENDER_EVERY_I
    }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    env = gym.make('WafLibinj-v0')

    # Train or test, depending on the mode specified
    if args.mode == 'train':
        if args.model == 'ppo':
            train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
        elif args.model == 'random':
            random_run(env=env, hyperparameters=hyperparameters)
        else:
            raise NotImplementedError("unsupported model was selected")
            
    else:
        test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)
