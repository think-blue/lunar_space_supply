import csv
import os.path

import numpy as np
import math
import sys
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment


class LunarEnvPositionDeltaPosReward(LunarEnvironment):
    def __init__(self, env_config):

        LunarEnvironment.__init__(self, env_config=env_config)


    def _get_reward(self):
        """
        reward is based on only position
        exceeding time step is penalised
        discount factor is there
        """

        position_threshold = 10000
        velocity_threshold = 40
        positional_error_magnitude = np.linalg.norm(self.delta_position)
        normalised_delta_pos = self.normalised_state['delta_position']
        normalised_positional_error_magnitude = np.linalg.norm(normalised_delta_pos)

        percentage_change = 100*(1 - normalised_positional_error_magnitude/self.previous_norm_pos_error)
        
        goal_achieved_reward = 0
        if positional_error_magnitude <= position_threshold:
            self.terminated_condition = True
            goal_achieved_reward = 2000

        time_penalty = 0
        if self.time_step > self.max_time_steps:
            self.truncated_condition = True
            time_penalty = -1000

        fuel_penalty = 0
        if self.fuel_mass < 0:
            self.truncated_condition = True
            fuel_penalty = -1000

        # static destination based on the end epoch
        positional_reward = 100*percentage_change
        # import pdb; pdb.set_trace()
        
        self.previous_norm_pos_error = normalised_positional_error_magnitude

        # discounted_reward = 2 * math.pow(self.env_config["discount_factor"], self.time_step)
        reward = positional_reward + time_penalty + goal_achieved_reward + fuel_penalty
        # print(positional_reward, mass_reward, velocity_reward)
        reward_components = [positional_reward, time_penalty, goal_achieved_reward, fuel_penalty]

        return reward, reward_components, self.truncated_condition, self.terminated_condition