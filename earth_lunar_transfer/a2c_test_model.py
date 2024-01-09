"""
This script runs a trained RL model for a single episode and plots the metrics
(Rewards, and velocity, position errors)
The experiment class, trained model checkpoint, and the experiment config needs to be loaded.
"""

from ray.rllib.algorithms import Algorithm
import json
import matplotlib.pyplot as plt
import numpy as np

# load the relevant experiment/environment module here
from earth_lunar_transfer.exp_time_step.exp_directed_force.lunarenvironment_directed_force import LunarEnvForceHelper

# load the experiment checkpoint here
checkpoint = "/home/chinmayd/ray_results/PPO_LunarEnvForceHelper_2023-09-12_22-31-15r58fiom2/checkpoint_000051"

# load the desired experiment's config here
with open("../earth_lunar_transfer/configs/env_config_test.json", "rb") as env_file:
    env_config_test = json.load(env_file)

with open("../earth_lunar_transfer/configs/env_config_train.json", "rb") as env_file:
    env_config_train = json.load(env_file)

algo = Algorithm.from_checkpoint(checkpoint)
env = LunarEnvForceHelper(env_config_train)
obs, _ = env.reset()
rewards = []

delta_velocity_array = []
delta_position_array = []

forces = []
actions = []

terminated, truncated = False, False
while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    actions.append(action)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    delta_position_array.append(env.delta_position), delta_velocity_array.append(env.delta_velocity)

delta_position_norm = np.linalg.norm(np.array(delta_position_array), axis=-1)
delta_velocity_norm = np.linalg.norm(np.array(delta_velocity_array), axis=-1)

plt.plot(delta_position_norm)
plt.title("delta_position_norm")
plt.show()

plt.plot(delta_velocity_norm)
plt.title("delta_vel_norm")
plt.show()

plt.plot(rewards), plt.title("reward"), plt.show()
plt.plot(np.cumsum(rewards)), plt.title("cum. Reward"), plt.show()


env.simulate(env.epoch_history, env.position_history, env.source_planet, env.destination_planet, "./myfile.html",
             env.start_position, env.target_position)

