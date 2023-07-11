import numpy as np
import ray
from gym_compatible_env import LunarEnvironment
from ray.rllib.utils import check_env
import json

with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)

lunar_env = LunarEnvironment(env_config)

sample_1 = lunar_env.reset()
sample_2 = lunar_env.action_space.sample()
sample_3 = lunar_env.observation_space.sample()

obs = lunar_env.step(np.array([-1, -1, -1]))