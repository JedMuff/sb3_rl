import time
import numpy as np

import gym
import panda_gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C # Seems to work better 1.39x improvementish. Uses all processors
from stable_baselines3 import PPO # Works worse with multiprocssing in Mujoco?
# Looking at CPU usage it seems to only utilise one processor at a time?

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

env_id = "PegBox-v0"
num_cpu = 8  # Number of processes to use
n_timesteps = 25000
algorithm = PPO

if __name__ == '__main__':
    vec_env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    multi_process_model = algorithm("MlpPolicy", vec_env, verbose=0)

    env = gym.make(env_id)
    single_process_model = algorithm('MlpPolicy', env, verbose=0)

    ## Initial Test of performance
    eval_env = gym.make(env_id)
    mean_reward, std_reward = evaluate_policy(multi_process_model, eval_env, n_eval_episodes=10)
    print(f'Initial Mean reward: {mean_reward} +/- {std_reward:.2f}')

    def testModel(model, n_timesteps):
        start_time = time.time()
        model.learn(n_timesteps)
        total_time = time.time() - start_time
        return total_time

    total_time_single = testModel(single_process_model, n_timesteps)
    print(f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS")

    total_time_multi = testModel(multi_process_model, n_timesteps)
    print(f"Took {total_time_multi:.2f}s for multi process version - {n_timesteps / total_time_multi:.2f} FPS")

    print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))

    # Evaluate the trained agents
    mean_reward, std_reward = evaluate_policy(single_process_model, eval_env, n_eval_episodes=10)
    print(f'Single Mean reward: {mean_reward} +/- {std_reward:.2f}')

    mean_reward, std_reward = evaluate_policy(multi_process_model, eval_env, n_eval_episodes=10)
    print(f'Multi Mean reward: {mean_reward} +/- {std_reward:.2f}')
