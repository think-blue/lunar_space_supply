import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

"""
plots the rewards for the episode with min position and velocity errors.
Also can check ranges of the values
"""

# from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment
from earth_lunar_transfer.experiments.exp_time_step.exp_gravity_states.lunarenvironment_direction_based import \
    LunarEnvForceHelper

with open("/nvme/lunar_space_supply/earth_lunar_transfer/reference_exp/env_config_test.json", "rb") as config_file:
    env_config = json.load(config_file)

lunar_env = LunarEnvForceHelper(env_config)
data = pd.read_csv(
    "/nvme/lunar_space_supply/data/training_data/exp_gravity_states_direction_reward/140057304227152.csv", header=0,
    index_col=False)
delta_pos = data.iloc[:, 7:10]
delta_vel = data.iloc[:, 10:13]

data["delta_pos_magnitude"] = np.linalg.norm(delta_pos, axis=-1)
data["delta_vel_magnitude"] = np.linalg.norm(delta_vel, axis=-1)

min_data_pos_row = data.loc[data["delta_pos_magnitude"] == np.min(data["delta_pos_magnitude"]), :]
min_data_vel_row = data.loc[data["delta_vel_magnitude"] == np.min(data["delta_vel_magnitude"]), :]

min_vel_episode = min_data_vel_row["episode"].unique()[0]
min_pos_episode = min_data_pos_row["episode"].unique()[0]

min_pos_data = data[data["episode"] == min_pos_episode]
min_vel_data = data[data["episode"] == min_vel_episode]

min_vel_data.iloc[:, 16:20:1].plot()
plt.show()

min_pos_data.iloc[:, 16:20:1].plot()
plt.show()

distance_from_moon = min_pos_data.iloc[:, 1:4] - np.array(
    [lunar_env.source_planet.eph(min_pos_data.iloc[i, 14])[0] for i in range(len(min_pos_data))])
dis_norm = np.linalg.norm(distance_from_moon, axis=-1)
plt.title("distance from moon")
plt.plot(dis_norm)
plt.show()

lunar_env.reset()
lunar_env.simulate(epoch_history=min_pos_data["epoch"],
                   position_history=min_pos_data.iloc[:, 1:4],
                   source_planet=lunar_env.source_planet,
                   destination_planet=lunar_env.destination_planet,
                   path="temp.html",
                   source_point=lunar_env.start_position,
                   destination_point=lunar_env.target_position)
pass
