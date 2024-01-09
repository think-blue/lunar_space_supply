"""
script to test the functioning of the environment
"""
from earth_lunar_transfer.exp_time_step.exp_position.lunarenvironment import LunarEnvPosition
# from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment
# from earth_lunar_transfer.exp_time_step.exp_gravity_states.lunarenvironment_direction_based import LunarEnvForceHelper
from earth_lunar_transfer.exp_time_step.exp_directed_force.lunarenvironment_directed_force import LunarEnvForceHelper
import json
import numpy as np
import matplotlib.pyplot as plt

with open("/nvme/lunar_space_supply/earth_lunar_transfer/env_config_test.json", "rb") as config_file:
    env_config = json.load(config_file)

lunar_env = LunarEnvForceHelper(env_config)

a = lunar_env.reset()

position_array = []
velocity_array = []
forces = []
moon_pos = []
delta_pos = []
actions = []
action = np.array([0, 0, 0])

# lunar_env.spacecraft_position = np.array(lunar_env.source_planet.eph(lunar_env.current_epoch)[0]) + np.array(
#     [2035383.3886006193, 2035383.3886006188, 2777890.732222725])
# lunar_env.spacecraft_velocity = np.array(lunar_env.source_planet.eph(lunar_env.current_epoch)[1]) + np.array(
#     [-735.0637737897396, -735.0637737897393, 1060.278455294141])

mass = []

for step in range(100):
    # action = np.random.uniform(-.1, .1, [3,])
    # action = np.array([0, 0, 0])
    previous_delta_pos = lunar_env.delta_position
    state, reward, terminated, truncated, info = lunar_env.step(action)
    new_delta_pos = lunar_env.delta_position
    # print(state["velocity"])
    forces.append(lunar_env.forces)
    moon_pos.append(lunar_env.source_planet.eph(lunar_env.current_epoch)[0])
    position_array.append(lunar_env.spacecraft_position)
    velocity_array.append(lunar_env.spacecraft_velocity)
    delta_pos.append(lunar_env.delta_position)
    mass.append(lunar_env.fuel_mass)

    distance_error = np.linalg.norm(lunar_env.delta_position)
    unit_direction_vector = -(lunar_env.delta_position / distance_error)

    # action_un = np.clip(10 * (-new_delta_pos), -50, 50) + np.clip(100 * (-new_delta_pos - (-previous_delta_pos)), -50, 50)

    # action_un = (100 * (-new_delta_pos)) + (1000 * (-new_delta_pos - (-previous_delta_pos)))
    # print(action_un)
    # action = 100 * (-(lunar_env.spacecraft_position - lunar_env.source_planet.eph(lunar_env.current_epoch)[
    #     0]) / np.linalg.norm(lunar_env.spacecraft_position - lunar_env.source_planet.eph(lunar_env.current_epoch)[0]))
    # action = action_un / 4e8
    # action = np.array([0, 0, 0])
    # if 80 < step < 10000:
    #     action = 2 * (lunar_env.spacecraft_velocity / np.linalg.norm(lunar_env.spacecraft_velocity))
    #     action = np.array([0, 0,0])
    # else:
    #     action = np.array([0, 0, 0])
    action = np.array([50, 50, 50]) / 4e8
    print(np.linalg.norm(lunar_env.forces[0]))
    # if iter < 140:
    #     action = 10 * unit_direction_vector  * (np.linalg.norm(lunar_env.delta_position) / 4e8)
    # else:
    #     action = 100 * unit_direction_vector * (np.linalg.norm(lunar_env.delta_position) / 4e8)
    actions.append(action)
    if lunar_env.fuel_mass < 0:
        break

print(lunar_env.fuel_mass)
position_array = np.array(position_array)
velocity_array = np.array(velocity_array)
moon_pos = np.array(moon_pos)
delta_pos = np.array(delta_pos)
action_array = np.array(actions)

delta_pos_norm = np.linalg.norm(delta_pos, axis=-1)
pos_norm = np.linalg.norm(position_array, axis=1)
vel_norm = np.linalg.norm(velocity_array, axis=1)
moon_pos_norm = np.linalg.norm(moon_pos, axis=-1)

plt.plot(delta_pos_norm)
plt.title("delta_pos")
plt.show()

plt.plot(np.linalg.norm(position_array - moon_pos, axis=-1), label="spacecraft")
plt.title("distance from moon")
plt.legend()
plt.show()

plt.plot(position_array[:, 0])
plt.plot(position_array[:, 1])
plt.plot(position_array[:, 2])
plt.title("position")
plt.show()

plt.plot(vel_norm)
plt.title("vel_norm")
plt.show()

# plt.plot(velocity_array[:, 0])
# plt.plot(velocity_array[:, 1])
# plt.plot(velocity_array[:, 2])
# plt.title("velocity")
# plt.show()

forces = np.array(forces)
forces_mag = np.linalg.norm(forces, axis=-1)
legend = ["spacecraft", "sun", "earth", "moon"]
for i in range(4):
    plt.plot(forces_mag[:, i], label=legend[i])
plt.legend()
plt.title("forces")
plt.show()

lunar_env.simulate(lunar_env.epoch_history, lunar_env.position_history, lunar_env.source_planet,
                   lunar_env.destination_planet, "./temp.html", lunar_env.start_position, lunar_env.target_position)

lunar_env.reset()
