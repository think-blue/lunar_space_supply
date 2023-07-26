from ray.rllib.algorithms import Algorithm
from lunarenvironment import LunarEnvironment
import json

checkpoint = "/home/chinmayd/ray_results/A2C_LunarEnvironment_2023-07-19_00-29-43a06s79he/checkpoint_000501"
a2c = Algorithm.from_checkpoint(checkpoint=checkpoint)

with open("env_config.json", "rb") as env_file:
    env_config = json.load(env_file)
env = LunarEnvironment(env_config)
obs = env.reset()

episode_reward = 0
terminated, truncated = False, False
while not terminated and not truncated:
    action = a2c.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward