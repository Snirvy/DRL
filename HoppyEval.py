#Evaluate Ant

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



if __name__ == '__main__':


# Create the evaluation environment
eval_env = make_vec_env('Ant-v4', seed=0, vec_env_cls=SubprocVecEnv
                   )

eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

# Load the trained SAC model
model = PPO.load('C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant_Expert.zip', env = env)

# Define variables to store the rewards
episode_rewards = []
total_timesteps = 0

# Set the number of episodes for evaluation
num_eval_episodes = 100

# Evaluate the model
for episode in range(num_eval_episodes):
    episode_reward = 0
    done = False
    obs = eval_env.reset()

    while not done:
        # Use the model to predict actions
        action, _ = model.predict(obs, deterministic=True)

        # Take the predicted action in the environment
        obs, reward, done, _ = eval_env.step(action)

        # Update the episode reward
        episode_reward += reward

        # Render the environment if desired
        #eval_env.render()

        # Increment the total timesteps
        total_timesteps += 1

    # Store the episode reward
    episode_rewards.append(episode_reward.flatten())
    print(f"Episode {episode + 1}: Reward = {episode_reward.flatten()}")

# Calculate the mean and standard deviation of rewards
mean_reward = sum(episode_rewards) / num_eval_episodes
std_reward = np.std(episode_rewards)

print(f"Mean Reward: {mean_reward:.2f}")
print(f"Std Reward: {std_reward:.2f}")