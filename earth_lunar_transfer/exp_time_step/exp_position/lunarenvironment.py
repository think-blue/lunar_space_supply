import csv
import os.path

import numpy as np
import math
import sys
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment


class LunarEnvPosition(LunarEnvironment):
    def __init__(self, env_config):
        LunarEnvironment.__init__(self, env_config=env_config)


    def _get_reward(self):
        """
        reward is based on only position
        exceeding time step is penalised
        discount factor is there
        """
        # static destination based on the end epoch
        dest_position = self.target_position
        position_error = (self.spacecraft_position - dest_position)
        positional_error_magnitude = np.linalg.norm(position_error) / (
                self.EARTH_MOON_MEAN_DISTANCE - self.destination_object_orbit_radius)
        positional_reward = - positional_error_magnitude

        time_penalty = 0
        if self.time_step > self.max_time_steps:
            time_penalty = -10

        discounted_reward = 1 * math.pow(self.env_config["discount_factor"], self.time_step)
        reward = discounted_reward + positional_reward + time_penalty
        # print(positional_reward, mass_reward, velocity_reward)
        reward_components = [discounted_reward, positional_reward, time_penalty]
        return reward, reward_components