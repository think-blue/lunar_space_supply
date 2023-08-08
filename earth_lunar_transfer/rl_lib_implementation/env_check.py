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

position_array = []
velocity_array = []
forces = []

for _ in range(10000):
    # action = np.random.uniform(-.1, .1, [3,])
    action = np.array([0, 0, 0])
    state, reward, terminated, truncated, info = lunar_env.step(action)
    forces.append(lunar_env.forces)
    position_array.append(lunar_env.spacecraft_position)
    velocity_array.append(lunar_env.spacecraft_velocity)

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

forces = np.array(forces)
forces_mag = np.linalg.norm(forces, axis=-1)
legend = ["spacecraft", "sun", "earth", "moon"]
for i in range(4):
    plt.plot(forces_mag[:, i], label=legend[i])
plt.legend()
plt.title("forces")
plt.show()

lunar_env.simulate(lunar_env.epoch_history, lunar_env.position_history, lunar_env.source_planet, lunar_env.destination_planet, ".", lunar_env.start_position, lunar_env.target_position)


lunar_env.reset()


