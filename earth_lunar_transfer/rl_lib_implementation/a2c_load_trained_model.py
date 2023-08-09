import numpy as np
from matplotlib import pyplot as plt
from ray.rllib.algorithms import Algorithm

from lunarenvironment_2 import LunarEnvironment2
from lunarenvironment import LunarEnvironment
import json

# checkpoint = "/home/gkalra/ray_results/A2C_LunarEnvironment_2023-07-19_00-29-43a06s79he/checkpoint_000501"
checkpoint = "/home/gkalra/ray_results/A2C_LunarEnvironment2_2023-07-31_17-48-03xyzd_gy0/checkpoint_000001"
a2c = Algorithm.from_checkpoint(checkpoint=checkpoint)

with open("env_config.json", "rb") as env_file:
    env_config = json.load(env_file)
env = LunarEnvironment2(env_config)
obs, _ = env.reset()

rewards = []
pos = []

terminated, truncated = False, False
while not terminated and not truncated:
    action = a2c.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    pos.append(obs['position'])

print(pos)
pos = np.array(pos)
plt.plot(rewards)
plt.show()
plt.plot(pos[:, 0], marker='o')
plt.plot(pos[:, 1], marker='*')
plt.plot(pos[:, 2], marker='+')
plt.show()
pass
