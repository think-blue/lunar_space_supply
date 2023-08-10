from ray.rllib.algorithms import Algorithm
from lunarenvironment import LunarEnvironment
import json
import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.algorithms.a2c import A2CConfig

checkpoint = "/home/chinmayd/ray_results/A2C_LunarEnvironment_2023-08-08_02-57-14f0vq808b/checkpoint_000801"

with open("env_config_test.json", "rb") as env_file:
    env_config_test = json.load(env_file)

with open("env_config_test.json", "rb") as env_file:
    env_config_train = json.load(env_file)

a2c_config = (
    A2CConfig()
    .environment(env=LunarEnvironment, env_config=env_config_train)
    .rollouts(num_rollout_workers=2, num_envs_per_worker=2)
    .resources(num_gpus=1)
    .training(train_batch_size=500)
    .evaluation(evaluation_num_workers=1)
)

a2c_algo = a2c_config.build()
a2c_algo.restore(checkpoint_path=checkpoint)

env = LunarEnvironment(env_config_test)
obs, _ = env.reset()

rewards = []
velocity_array = []
delta_velocity_array = []

position_array = []
delta_position_array = []

forces = []

terminated, truncated = False, False
while not terminated and not truncated:
    # action = np.array([0, 0, 0])
    action = a2c_algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    delta_position_array.append(env.delta_position), delta_velocity_array.append(env.delta_velocity)
    position_array.append(env.delta_position), velocity_array.append(env.delta_velocity)

position_array = np.array(position_array)
velocity_array = np.array(velocity_array)
norm = np.linalg.norm(position_array, axis=1)
vel_norm = np.linalg.norm(velocity_array, axis=1)

plt.plot(norm)
plt.title("position_norm")
plt.show()

plt.plot(position_array[:, 0])
plt.plot(position_array[:, 1])
plt.plot(position_array[:, 2])
plt.title("position")
plt.show()

plt.plot(vel_norm)
plt.title("vel_norm")
plt.show()

plt.plot(velocity_array[:, 0])
plt.plot(velocity_array[:, 1])
plt.plot(velocity_array[:, 2])
plt.title("velocity")
plt.show()

# forces = np.array(forces)
# forces_mag = np.linalg.norm(forces, axis=-1)
# legend = ["spacecraft", "sun", "earth", "moon"]
# for i in range(4):
#     plt.plot(forces_mag[:, i], label=legend[i])
# plt.legend()
# plt.title("forces")
# plt.show()

env.simulate(env.epoch_history, env.position_history, env.source_planet, env.destination_planet, ".", env.start_position, env.target_position)




pass
# env.simulate(env.epoch_history, env.position_history, env.source_planet, env.destination_planet, "../data/file.htm")