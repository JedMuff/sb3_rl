import time
import argparse
import datetime
import panda_gym
from sb_callbacks import CustomCallback

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="PPO_PegBox-v0", help="model name to use, e.g. type model_name in models/model_name")
parser.add_argument("--algorithm", type=str, default="PPO", help="Type of algorithm to use:\
    A2C, DDPG, DQN, PPO, SAC, TD3")
parser.add_argument("--ts", type=int, default=1000, help="Number of timesteps the RL algorithm trains for")
parser.add_argument("--env", type=str, default="PegBox-v0", help="Enviroment to use")
parser.add_argument("--cpus", type=int, default=4, help="Number of CPUs to use")
args = parser.parse_args()

algorithms = {"A2C":A2C, "DDPG":DDPG, "DQN":DQN, 
              "PPO":PPO, "SAC":SAC, "TD3":TD3}

env_id = args.env
n_timesteps = args.ts
algorithm = algorithms[args.algorithm]
dir_model = "models/" + args.model + '/'
num_cpu = args.cpus  # Number of processes to use

class randomise_on_reset(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv=venv)
    
    def reset(self):
        obs = self.venv.reset()
        self.env_method('set_random_task')
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info

if __name__ == '__main__':
    time1 = time.time()
    print(f'1/4: Making Vector Enviroment for {args.model}, Time: {0}...', flush=True)
    vec_env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method':'fork'})
    vec_env = randomise_on_reset(vec_env)
    callback = CustomCallback(path=dir_model)
    
    time2 = time.time()
    print(f'2/4: Making model for {args.model}, Time: {datetime.timedelta(seconds=time2-time1)}...', flush=True)

    model = algorithm("MlpPolicy", vec_env, verbose=0)

    time3 = time.time()
    print(f'3/4: Training starting for model {args.model}, Time: {datetime.timedelta(seconds=time3-time1)}...', flush=True)
    start_time = time.time()
    model.learn(n_timesteps, callback=callback)
    finish_time = time.time()
    total_time = finish_time - start_time
    print('4/4: Training completed. Processing Time: ' + str(datetime.timedelta(seconds=total_time)), flush=True)
    print(f'Time since start: {datetime.timedelta(seconds=finish_time-time1)}', flush=True)
    model.save(dir_model+args.model)
