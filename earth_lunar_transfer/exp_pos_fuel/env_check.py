import numpy as np
import ray
from lunarenvironment import LunarEnvironment
from ray.rllib.utils import check_env
import json
import numpy as np
import matplotlib.pyplot as plt

with open("env_config_test.json", "rb") as config_file:
    env_config = json.load(config_file)

lunar_env = LunarEnvironment(env_config)

lunar_env.reset()


