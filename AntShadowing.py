import gymnasium as gym ### watch out
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import torch as th
import csv


rng = np.random.default_rng(0)

def train_expert(env):
    print("Training a expert.")
    expert = SAC(
        policy=MlpPolicy,
        env=env,
        seed=999,
        batch_size=64,
        ent_coef=0.001,
        learning_rate=0.0003,
        
      
    )
    for i in range(300):
        expert.learn(10_000)
        print('Checkpoint reached! epoch {}'.format(i))
        print('Saving the model!')
        expert.save("C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant_Expert_SAC.zip")        # Note: change this to 100000 to train a decent expert.
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


if __name__ == '__main__':
    env_id = 'Ant-v4'
    env = make_vec_env(env_id, n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    env = make_vec_env ( 'Ant-v4' , n_envs =6 , seed =999 , vec_env_cls =
    SubprocVecEnv)
    
    env = VecNormalize ( env , norm_obs = True , norm_reward = True )

    #expert = train_expert(env)
    #env.save( 'C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant-v4_vecnormalize.pkl')
    expert = SAC.load('C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant_Expert_SAC')
    env = VecNormalize.load( 'C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant-v4_vecnormalize.pkl',env )
    env.training = False
    env.norm_reward = False
    
    transitions = sample_expert_transitions(expert, env_id='Ant-v4')
    
    policy_kwargs = dict ( activation_fn = th.nn.ReLU,net_arch =[128 , 128 , dict ( pi =[32] , vf =[32]) ])
    
    rewards_post = []
    rewards_pre = []
    
    bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
   # policy_kwargs=policy_kwargs  # Pass the modified policy_kwargs)

    )

    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=10,
        render=False,
    )
    
    rewards_pre.append(reward)
    print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=128)

    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=100,
        render=False,
    )
    
    rewards_post.append(reward)
    print(f"Reward after training: {reward}")

    before = "C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant_Shadow_Reward_Before_128.csv"
    after = "C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant_Shadow_Reward_After_128.csv"

    # Open the CSV file in write mode
    with open(before, mode='w') as file:
        writer = csv.writer(file)

        # Write the list as a row in the CSV file
        writer.writerow(rewards_pre)

    print("List saved as CSV successfully!")
    
    with open(after, mode='w') as file:
        writer = csv.writer(file)

        # Write the list as a row in the CSV file
        writer.writerow(rewards_post)

    print("List saved as CSV successfully!")
    
    bc_trainer.policy.save('C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant_Shadow_128.zip')
    print('Model saved!')
    
    env.save( 'C:\\Users\\Gebruiker\\OneDrive\\Bureaublad\\Anita\'s stuff II\\Tilburg university\\DRL\\Ant\\Ant_Shadow_128.pkl')
    print('Environment saved!')