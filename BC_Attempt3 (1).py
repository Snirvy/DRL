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
import torch as th

rng = np.random.default_rng(999)



def train_expert(env, netsize):
    print("Training a expert.")
    expert = SAC(
        policy=MlpPolicy,
        env=env,
        seed=999,
        batch_size=64,
        ent_coef=0.001,
        learning_rate=0.0003,
        
        #policy_kwargs = dict ( activation_fn = th.nn.ReLU , net_arch =[netsize , netsize, dict ( pi =[32] , vf =[32]) ])
        policy_kwargs = dict(net_arch=dict(pi=[netsize,netsize], qf=[netsize,netsize]))
    )
    expert.learn(1_000_000)  # Note: change this to 100000 to train a decent expert.
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
    env = make_vec_env(env_id, n_envs=6, seed=0, vec_env_cls=SubprocVecEnv)

    netsizes = [8,16,32,64,128,256]
    for i in range(len(netsizes)):
        expert = train_expert(env, netsizes[i])
        transitions = sample_expert_transitions(expert, env_id='Ant-v4')
        
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=rng,
        )

        reward, _ = evaluate_policy(
            bc_trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=10,
            render=True,
        )
        print("Reward before training: {}, netsize : {}".format(reward, netsizes[i]))

        print("Training a policy using Behavior Cloning")
        bc_trainer.train(n_epochs=100)

        reward, _ = evaluate_policy(
            bc_trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=10,
            render=True,
        )
        print("Reward after training: {}, netsize : {}".format(reward, netsizes[i]))
