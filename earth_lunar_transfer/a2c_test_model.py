from ray.rllib.algorithms import Algorithm
# from earth_lunar_transfer.exp_time_step.exp_position.lunarenvironment import LunarEnvPosition
# from earth_lunar_transfer.exp_time_step.exp_no_gravity.lunarenvironment_negative import LunarEnvNoGravity
# from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment
from earth_lunar_transfer.exp_time_step.exp_gravity_states.lunarenvironment_direction_based import LunarEnvForceHelper
from earth_lunar_transfer.exp_time_step.exp_gravity_states.lunarenvironment_direction_based_position import LunarEnvForceHelper


import json
import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ppo import PPOConfig


# checkpoint = "/home/chinmayd/ray_results/A2C_LunarEnvPosition_2023-08-22_11-43-42awd88n17/checkpoint_001002"
# checkpoint = "/home/chinmayd/ray_results/PPO_LunarEnvNoGravity_2023-08-23_21-17-25bcxc6xs1/checkpoint_000702"
# checkpoint = "/home/chinmayd/ray_results/PPO_LunarEnvNoGravity_2023-08-23_21-56-41ip3g7ho8/checkpoint_003501"
# checkpoint = "/home/chinmayd/ray_results/PPO_LunarEnvNoGravity_2023-08-23_21-17-25bcxc6xs1/checkpoint_004402"

# PPO Discrete Negative Reward
# checkpoint = "/home/chinmayd/ray_results/PPO_LunarEnvForceHelper_2023-08-25_13-17-472vlgxp50/checkpoint_007901"

# ppo discrete direction based reward
# checkpoint = "/home/chinmayd/ray_results/PPO_LunarEnvForceHelper_2023-08-25_19-03-17nyhj4zv6/checkpoint_008001"
checkpoint = "/home/chinmayd/ray_results/PPO_LunarEnvForceHelper_2023-08-28_22-31-31m4zpdfwo/checkpoint_002401"

with open("env_config_test.json", "rb") as env_file:
    env_config_test = json.load(env_file)

with open("env_config_train.json", "rb") as env_file:
    env_config_train = json.load(env_file)

a2c_config = (
    PPOConfig()
        .environment(env=LunarEnvForceHelper, env_config=env_config_test)
        .training(grad_clip=3, lr_schedule=[[0, 1e-6], #0.00002 this works
                                            [2e6, 0.00000006],
                                            [20000000, 0.000000000001]])
        .rollouts(num_rollout_workers=5, num_envs_per_worker=5)
        .evaluation(evaluation_num_workers=1)
        .framework("torch")
        .debugging(log_level="INFO")
        .offline_data(output="/nvme/lunar_space_supply/data/training_data/logs")
)

a2c_algo = a2c_config.build()
a2c_algo.restore(checkpoint_path=checkpoint)

env = LunarEnvForceHelper(env_config_test)
obs, _ = env.reset()

rewards = []

delta_velocity_array = []
delta_position_array = []

forces = []
actions = []

terminated, truncated = False, False
# env.spacecraft_position = np.array(env.source_planet.eph(env.current_epoch)[0]) + np.array([2035383.3886006193, 2035383.3886006188, 2777890.732222725])
# env.spacecraft_velocity = np.array(env.source_planet.eph(env.current_epoch)[1]) + np.array([-735.0637737897396, -735.0637737897393, 1060.278455294141])
# env.spacecraft_velocity = np.array([80, 0, 0])
while not terminated and not truncated:
    # action = np.array([0, 0, 0])
    action = a2c_algo.compute_single_action(obs)
    print(action)
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

# plt.plot(velocity_array[:, 0])
# plt.plot(velocity_array[:, 1])
# plt.plot(velocity_array[:, 2])
# plt.title("velocity")
# plt.show()

# forces = np.array(forces)
# forces_mag = np.linalg.norm(forces, axis=-1)
# legend = ["spacecraft", "sun", "earth", "moon"]
# for i in range(4):
#     plt.plot(forces_mag[:, i], label=legend[i])
# plt.legend()
# plt.title("forces")
# plt.show()

env.simulate(env.epoch_history, env.position_history, env.source_planet, env.destination_planet, "./myfile.html", env.start_position, env.target_position)

pass
# essh -N -f -L localhost:5000:localhost:5000 chinmayd@10.131.4.220nv.simulate(env.epoch_history, env.position_history, env.source_planet, env.destination_planet, "../data/file.htm")