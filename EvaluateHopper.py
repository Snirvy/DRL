import gym
from stable_baselines3 import SAC
import os
import gymnasium as gym ### watch out
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import os 

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
from stable_baselines3 import SAC
from stable_baselines3 import PPO

import pandas as pd


class ChangeMassWrapper(gym.Wrapper):
    def __init__ (self, env, torso_mass =9):
        super().__init__ (env)

        self.torso_mass = torso_mass
        self.env.model.body_mass[1] = self.torso_mass


if __name__ == '__main__':

    # Define the number of episodes for evaluation
    num_eval_episodes = 100

    # Define the range of torso masses
    torso_masses = range(3, 10)

    # Initialize a DataFrame to hold all results
    all_rewards_df = pd.DataFrame()

    for torso_mass in torso_masses:

        # Make environment with the given torso mass
        env = make_vec_env('Hopper-v4', n_envs=1, seed=999, vec_env_cls=SubprocVecEnv, wrapper_class=ChangeMassWrapper, wrapper_kwargs=dict(torso_mass=torso_mass))
        env = VecNormalize.load('Hopper_w6.pkl', env )
        env.training = False
        env.norm_reward = False


        # Load the trained model
        model = SAC.load("Hopper_W6.zip", env=env)

        # Initialize variables
        episode_rewards = []
        total_timesteps = 0

        for episode in range(num_eval_episodes):
            episode_reward = 0
            done = False
            obs = env.reset()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)

                episode_reward += reward
                total_timesteps += 1

            episode_rewards.append(episode_reward.flatten())
            print(f"Torso Mass {torso_mass}, Episode {episode + 1}: Reward = {episode_reward.flatten()}")

        # Extract values from array and create a list
        rewards = [arr[0] for arr in episode_rewards]

        # Convert list to pandas DataFrame
        df = pd.DataFrame(rewards, columns=[f'w{torso_mass}'])

        # Concatenate the new results to the existing DataFrame
        all_rewards_df = pd.concat([all_rewards_df, df], axis=1)

    # Save the DataFrame with all results to a CSV file
    all_rewards_df.to_csv('Hopper_W6.csv', index=False)