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

        position_threshold = 500e3
        velocity_threshold = 400

        goal_achieved_reward = 0
        if np.linalg.norm(self.delta_position) <= position_threshold:
            self.terminated_condition = True
            goal_achieved_reward = 20

        time_penalty = 0
        if self.time_step > self.max_time_steps:
            self.truncated_condition = True
            time_penalty = -10

        fuel_penalty = 0
        if self.fuel_mass < 0:
            self.truncated_condition = True
            fuel_penalty = -10

        regions_penalty = 0
        if np.linalg.norm(self.spacecraft_position - self.source_planet.eph(self.current_epoch)[0]) < 1737e3 + 300e3:
            regions_penalty = -10
            self.truncated_condition = True

        if np.linalg.norm(self.spacecraft_position - self.destination_planet.eph(self.current_epoch)[0]) < 6738e3 + 300e3:
            regions_penalty = -10
            self.truncated_condition = True

        # static destination based on the end epoch
        dest_position = self.target_position
        position_error = (self.spacecraft_position - dest_position)
        positional_error_magnitude = np.linalg.norm(position_error) / (
                self.EARTH_MOON_MEAN_DISTANCE - self.destination_object_orbit_radius)
        positional_reward = - positional_error_magnitude

        #todo: going towards or away from the end goal
        discounted_reward = 2 * math.pow(self.env_config["discount_factor"], self.time_step)
        reward = discounted_reward + positional_reward + time_penalty + goal_achieved_reward + fuel_penalty + regions_penalty
        # print(positional_reward, mass_reward, velocity_reward)
        reward_components = [discounted_reward, positional_reward, time_penalty, goal_achieved_reward, fuel_penalty, regions_penalty]

        return reward, reward_components, self.truncated_condition, self.terminated_condition