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

rng = np.random.default_rng(999)


def train_expert(env, mass):
    print("Training a expert.")
    expert = SAC(
        policy=MlpPolicy,
        env=env,
        
    )
    

    for i in range(300):
        expert.learn(10_000)
        print('Checkpoint reached! epoch {}'.format(i))
        print('Saving the model!')
        expert.save("C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Hopper_W{}.zip".format(mass)        # Note: change this to 100000 to train a decent expert.

    return expert

def sample_expert_transitions(expert, env_id):
    print("Sampling expert transitions.")

    wrapped_venv = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_id))])
    rollouts = rollout.rollout(
        expert,
        wrapped_venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)
    
    
from stable_baselines3.common.callbacks import BaseCallback


if __name__ == '__main__':

    env_id = 'Hopper-v4'
    
    masses = [6, 9]
    for mass in massses:
    
        env = make_vec_env(env_id, n_envs=6, seed=999, vec_env_cls=SubprocVecEnv)
       
        expert = train_expert(env, mass)
    
        print('Finished! Saving the model.')
        expert.save("C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Hopper_W{}.zip".format(mass))
