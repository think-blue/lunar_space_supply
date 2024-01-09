import os

import numpy as np
import ray
from lunarenvironment import LunarEnvironment
from ray.rllib.utils import check_env
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)

lunar_env = LunarEnvironment(env_config)

lunar_env.reset()

position_array = []
velocity_array = []
forces = []

# loaded_tensors = []
# with open("tensor_file2.pt", 'rb') as file:
#     while True:
#         try:
#             tensor = torch.load(file)
#             loaded_tensors.append(tensor)
#             break
#         except EOFError:
#             break

# loaded_tensors = loaded_tensors[0]
# i = 0
# while i < len(loaded_tensors):
#     action = np.array(loaded_tensors[i: i+3])
#     state, reward, terminated, truncated, info = lunar_env.step(action)
#     forces.append(lunar_env.forces)
#     position_array.append(lunar_env.spacecraft_position)
#     velocity_array.append(lunar_env.spacecraft_velocity)

# i += 3

for i in range(300):
    #     # action = np.random.uniform(-.1, .1, [3,])
    action = np.array([10, 10, 10])
    # if i < 500:
    #     action = 3 * (lunar_env.target_position - lunar_env.spacecraft_position) / np.linalg.norm(
    #         lunar_env.target_position - lunar_env.spacecraft_position)
    # elif 500 < i < 1200:
    #     action = 10 * (lunar_env.target_position - lunar_env.spacecraft_position) / np.linalg.norm(
    #         lunar_env.target_position - lunar_env.spacecraft_position)
    # elif 1200 < i < 2500:
    #     action = 20 * (lunar_env.target_position - lunar_env.spacecraft_position) / np.linalg.norm(
    #         lunar_env.target_position - lunar_env.spacecraft_position)
    # elif 2500 < i < 2550:
    #     action = 30 * (lunar_env.target_position - lunar_env.spacecraft_position) / np.linalg.norm(
    #         lunar_env.target_position - lunar_env.spacecraft_position)
    # else:
    #     action = 1 * (lunar_env.target_position - lunar_env.spacecraft_position) / np.linalg.norm(
    #         lunar_env.target_position - lunar_env.spacecraft_position)

    state, reward, terminated, truncated, info = lunar_env.step(action)
    forces.append(lunar_env.forces)
    position_array.append(lunar_env.spacecraft_position)
    velocity_array.append(lunar_env.spacecraft_velocity)

norm = np.linalg.norm(position_array, axis=1)
vel_norm = np.linalg.norm(velocity_array, axis=1)

plt.plot(norm)
plt.title("position_norm")
plt.show()

position_array = np.array(position_array)
plt.plot(position_array[:, 0])
plt.plot(position_array[:, 1])
plt.plot(position_array[:, 2])
plt.title("position")
plt.show()

plt.plot(vel_norm)
plt.title("vel_norm")
plt.show()

velocity_array = np.array(velocity_array)
plt.plot(velocity_array[:, 0])
plt.plot(velocity_array[:, 1])
plt.plot(velocity_array[:, 2])
plt.title("velocity")
plt.show()

forces = np.array(forces)
forces_mag = np.linalg.norm(forces, axis=-1)
legend = ["spacecraft", "sun", "earth", "moon"]
for i in range(4):
    plt.plot(forces_mag[:, i], label=legend[i])
plt.legend()
plt.title("forces")
plt.show()

lunar_env.simulate(lunar_env.epoch_history, lunar_env.position_history, lunar_env.source_planet,
                   lunar_env.destination_planet,
                   "temp.html",
                   lunar_env.start_position, lunar_env.target_position)

# lunar_env.reset()
