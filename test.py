import gym
import time
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def testModel(model, n_timesteps):
    start_time = time.time()
    model.learn(n_timesteps)
    total_time = time.time() - start_time
    return total_time

n_timesteps = 25000
env_id = "CartPole-v1" # "PegBox-v0" 
# Parallel environments
env_single = make_vec_env(env_id, n_envs=1)
single_process_model = PPO("MlpPolicy", env_single, verbose=0)

env_multi = make_vec_env(env_id, n_envs=8)
multi_process_model = PPO("MlpPolicy", env_multi, verbose=0)

total_time_single = testModel(single_process_model, n_timesteps)
print(f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS")

total_time_multi = testModel(multi_process_model, n_timesteps)
print(f"Took {total_time_multi:.2f}s for multi process version - {n_timesteps / total_time_multi:.2f} FPS")

print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))

