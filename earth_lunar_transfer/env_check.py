from earth_lunar_transfer.exp_time_step.exp_position.lunarenvironment import LunarEnvPosition
import json
import numpy as np
import matplotlib.pyplot as plt

with open("/nvme/lunar_space_supply/earth_lunar_transfer/env_config_test.json", "rb") as config_file:
    env_config = json.load(config_file)

lunar_env = LunarEnvPosition(env_config)

a = lunar_env.reset()

position_array = []
velocity_array = []
forces = []
moon_pos = []
delta_pos = []
action = np.array([0, 0, 0])

# lunar_env.spacecraft_position = np.array(lunar_env.source_planet.eph(lunar_env.current_epoch)[0]) + np.array([2035383.3886006193, 2035383.3886006188, 2777890.732222725])
# lunar_env.spacecraft_velocity = np.array(lunar_env.source_planet.eph(lunar_env.current_epoch)[1]) + np.array([-735.0637737897396, -735.0637737897393, 1060.278455294141])

for iter in range(5000):
    # action = np.random.uniform(-.1, .1, [3,])
    action = np.array([0, 0, 0])
    state, reward, terminated, truncated, info = lunar_env.step(action)
    print(state["velocity"])
    forces.append(lunar_env.forces)
    moon_pos.append(lunar_env.source_planet.eph(lunar_env.current_epoch)[0])
    position_array.append(lunar_env.spacecraft_position)
    velocity_array.append(lunar_env.spacecraft_velocity)
    delta_pos.append(lunar_env.delta_position)

    unit_direction_vector = -(lunar_env.delta_position / np.linalg.norm(lunar_env.delta_position))
    # if iter < 140:
    # action = 10 * unit_direction_vector  * (np.linalg.norm(lunar_env.delta_position) / 4e8)
    # else:
    #     action = 100 * unit_direction_vector

print(lunar_env.fuel_mass)
position_array = np.array(position_array)
velocity_array = np.array(velocity_array)
moon_pos = np.array(moon_pos)
delta_pos = np.array(delta_pos)

delta_pos_norm = np.linalg.norm(position_array, axis=-1)
pos_norm = np.linalg.norm(position_array, axis=1)
vel_norm = np.linalg.norm(velocity_array, axis=1)
moon_pos_norm = np.linalg.norm(moon_pos, axis=-1)


plt.plot(delta_pos_norm)
plt.title("delta_pos")
plt.show()

plt.plot(np.linalg.norm(pos_norm - moon_pos_norm), label="spacecraft")
plt.title("position_norm")
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

lunar_env.simulate(lunar_env.epoch_history, lunar_env.position_history, lunar_env.source_planet, lunar_env.destination_planet, "./elliptical.html", lunar_env.start_position, lunar_env.target_position)


lunar_env.reset()



