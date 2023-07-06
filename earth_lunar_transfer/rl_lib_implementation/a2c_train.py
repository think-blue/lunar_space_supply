import ray
from ray.rllib.algorithms import a2c
import json
from gym_compatible_env import LunarEnvironment

with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)

ray.init()
algo = a2c.A2C(env=LunarEnvironment, config={
    "env_config": env_config})

while True:
    print(algo.train())
