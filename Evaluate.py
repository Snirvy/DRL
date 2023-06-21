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



if __name__ == '__main__':

    num_eval_episodes = 100
    os.chdir('C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant')
    # Create the evaluation environment
    env = make_vec_env('Ant-v4', n_envs=1, seed=999, vec_env_cls=SubprocVecEnv,
    )
    env = VecNormalize.load('Ant_Shadow_128.pkl', env )
    env.training = False
    env.norm_reward = False


    # Load the trained model
    model = SAC.load("Ant_Shadow_128.zip", env=env)

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
        print(f"Episode {episode + 1}: Reward = {episode_reward.flatten()}")
        
    rewards = [arr[0] for arr in episode_rewards]

    # Convert list to pandas DataFrame
    df = pd.DataFrame([rewards])

    # Concatenate the new results to the existing DataFrame
    

# Save the DataFrame with all results to a CSV file
    df.to_csv('Ant_Shadow_128_Rewards.csv', index=False)

