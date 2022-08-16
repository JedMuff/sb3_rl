
import gym
import panda_gym
import argparse
from panda_gym.core import controllers
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

parser = argparse.ArgumentParser()
parser.add_argument("--render", action=argparse.BooleanOptionalAction, help="If True it renders the result of the model")
parser.add_argument("--model", type=str, default="PPO_PegBox-v0", help="model name to use, e.g. type model_name in models/model_name")
parser.add_argument("--algorithm", type=str, default="PPO", help="Type of algorithm to use:\
    A2C, DDPG, DQN, PPO, SAC, TD3")
parser.add_argument("--ts", type=int, default=5000, help="Number of timesteps the RL algorithm trains for")
parser.add_argument("--env", type=str, default="PegBox-v0", help="Enviroment to use")
parser.add_argument("--init", action=argparse.BooleanOptionalAction, default=True, help="Enviroment to use")
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Enviroment is random?")
parser.add_argument("--repeats", type=int, default=1, help="how many times to repeat render")

args = parser.parse_args()

algorithms = {"A2C":A2C, "DDPG":DDPG, "DQN":DQN, 
              "PPO":PPO, "SAC":SAC, "TD3":TD3}

env_id = args.env
n_timesteps = args.ts
algorithm = algorithms[args.algorithm]
saved_model = "models/" + args.model + '/' + args.model

## Initial Test of performance
eval_env = gym.make(env_id)
if args.init:
    dummy_model = algorithm("MlpPolicy", eval_env, verbose=0)
    mean_reward, std_reward = evaluate_policy(dummy_model, eval_env, n_eval_episodes=10)
    print(f'Initial Mean reward: {mean_reward} +/- {std_reward:.2f}')

# Evaluate the trained agents
model = algorithm.load(saved_model)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'New Mean reward: {mean_reward} +/- {std_reward:.2f}')

def hold(ts, env):
    # env = gym.make(args.env)
    controller = controllers.TorqueController(env, gravity_compensation=True)
    for t in range(ts):
        torque = controller.get_control(0)
        env.apply_joint_motor_command(torque)
        env.sim.step()
        env.render()

# Render
if args.render:
    for _ in range(args.repeats):
        obs = eval_env.reset()
        if args.random:
            eval_env.set_random_task()

        hold(2000, eval_env)

        for t in range(args.ts):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = eval_env.step(action)
            eval_env.render()


