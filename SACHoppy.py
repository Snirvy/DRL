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

rng = np.random.default_rng(999)


class ChangeMassWrapper(gym.Wrapper):
    def __init__ (self, env, torso_mass =3):
        super().__init__ (env)

        self.torso_mass = torso_mass
        self.env.model.body_mass[1] = self.torso_mass



def train_expert(env, mass):
    print("Training an expert.")
    
    expert = SAC(
        policy=MlpPolicy,
        env=env,
        batch_size=64,
        ent_coef=0.001,
        learning_rate=0.0003
        
    )
    

    for i in range(300):
        expert.learn(10_000)
        print('Checkpoint reached! epoch {}'.format(i))
        print('Saving the model!')
        expert.save("C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Hopper_W{}.zip".format(mass))       # Note: change this to 100000 to train a decent expert.

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
    
    masses = [3, 6, 9]
    for mass in masses:
    
    
        env = make_vec_env('Hopper-v4', seed=0, vec_env_cls=SubprocVecEnv,
                           wrapper_class=ChangeMassWrapper,
                           wrapper_kwargs=dict(torso_mass=mass),
                           n_envs=6
                           
                           )

        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        

    
       # env = make_vec_env(env_id, n_envs=6, seed=999, vec_env_cls=SubprocVecEnv)
       
        expert = train_expert(env, mass)
        env.save( 'Hopper_w{}.pkl'.format(mass))
    
        print('Finished! ')
        expert.save("C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Hopper_W{}.zip".format(mass))
