import csv
import os.path

import numpy as np
import math
import sys
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
# sys.path.append("/nvme/lunar_space_supply/earth_lunar_transfer")
from earth_lunar_transfer.reference_exp.lunarenvironment import LunarEnvironment
import mlflow
from scipy.integrate import odeint
from gymnasium.spaces import MultiDiscrete


class LunarEnvNoGravity(LunarEnvironment):
    """
    major changes:
        - no gravity
        - action is acceleration instead of force
        - initial velocity is zero
        - previous position history is defined
        - reward is based on direction vector
    """

    def __init__(self, env_config):
        LunarEnvironment.__init__(self, env_config=env_config)
        self.MU_MOON = 0
        self.MU_EARTH = 0
        self.MU_SUN = 0

        if self.env_config["action_space"] == "discrete":
            # todo: make it (2*n + 1)
            self.action_space = MultiDiscrete([10, 10, 10])

    def reset(self, *, seed=None, options=None):
        """resets the environment to the initial state based
         on the environment config parameters passed"""
        self.fuel_mass = self.env_config["fuel_mass"]
        self.current_epoch = self.env_config["start_epoch"]

        spacecraft_mass = self.env_config["payload_mass"] + self.env_config["fuel_mass"]
        source_planet_eph = self.source_planet.eph(self.current_epoch)
        spacecraft_initial_speed = self.get_orbital_speed(self.source_object_orbit_radius, self.MU_MOON)
        spacecraft_position, spacecraft_velocity = self.get_eph_from_orbital_angles(self.source_azimuthal_angle,
                                                                                    self.source_inclination_angle,
                                                                                    self.source_object_orbit_radius,
                                                                                    spacecraft_initial_speed,
                                                                                    source_planet_eph)
        spacecraft_velocity = np.array([0, 0, 0])
        destination_planet_eph = self.destination_planet.eph(self.current_epoch)
        target_speed = self.get_orbital_speed(self.destination_object_orbit_radius, self.MU_EARTH)
        target_position, target_velocity = self.get_eph_from_orbital_angles(self.dest_azimuthal_angle,
                                                                            self.dest_inclination_angle,
                                                                            self.destination_object_orbit_radius,
                                                                            target_speed,
                                                                            destination_planet_eph)
        self.time_step = 0
        self.state = dict(
            delta_position=spacecraft_position - target_position,
            delta_velocity=spacecraft_velocity - target_velocity,
            mass=np.array([spacecraft_mass]),
            position=spacecraft_position,
            time_step=np.array([self.time_step]),
            velocity=spacecraft_velocity)

        self.spacecraft_mass = self.state["mass"].item()
        self.spacecraft_position = self.state["position"]
        self.spacecraft_velocity = self.state["velocity"]
        self.delta_position = self.state["delta_position"]
        self.delta_velocity = self.state["delta_velocity"]
        self.time_step = self.state["time_step"].item()
        self.reward = 0
        self.reward_components = None

        self.start_position = self.spacecraft_position
        self.start_velocity = self.spacecraft_velocity
        self.target_position = target_position
        self.target_velocity = target_velocity

        self.previous_spacecraft_position = self.spacecraft_position
        self.previous_spacecraft_velocity = self.spacecraft_velocity
        if self.env_config["mlflow_configured"]:
            with mlflow.start_run(run_id=self.env_config["mlflow_run_id"]):
                mlflow.log_param(f"start_speed_{self.object_id}", np.linalg.norm(spacecraft_velocity))
                mlflow.log_param(f"target_speed_{self.object_id}", np.linalg.norm(target_velocity))
                mlflow.log_param(f"target_velocity_x_{self.object_id}", target_velocity[0])
                mlflow.log_param(f"target_velocity_y_{self.object_id}", target_velocity[1])
                mlflow.log_param(f"target_velocity_z_{self.object_id}", target_velocity[2])
                mlflow.log_param(f"target_position_x_{self.object_id}", target_position[0])
                mlflow.log_param(f"target_position_y_{self.object_id}", target_position[1])
                mlflow.log_param(f"target_position_z_{self.object_id}", target_position[2])
                pass

        # write episode history and reset history
        self._write_episode_history()
        self.fuel_mass_history = []
        self.position_history = []
        self.velocity_history = []
        self.delta_position_history = []
        self.delta_velocity_history = []
        self.time_step_history = []
        self.epoch_history = []
        self.reward_history = []
        self.reward_components_history = []

        self.episode_count += 1

        info = {}

        self.normalised_state = self._normalise_state(self.state)

        self.terminated_condition = False
        self.truncated_condition = False

        return self.normalised_state, info

    def accelerate(self, state_0, time, thrust, spacecraft_mass, epoch):
        """
        thrust is acceleration (action is acceleration)
        """
        position = state_0[0:3]
        velocity = state_0[3:]

        r_vector_sun = np.array(self.sun.eph(epoch))[0] - position
        r_mag_sun = np.linalg.norm(r_vector_sun)

        r_vector_moon = np.array(self.source_planet.eph(epoch))[0] - position
        r_mag_moon = np.linalg.norm(r_vector_moon)

        r_vector_earth = np.array(self.destination_planet.eph(epoch))[0] - position
        r_mag_earth = np.linalg.norm(r_vector_earth)

        state_0 = (velocity,
                   thrust +
                   self.MU_SUN / np.power(r_mag_sun, 3) * r_vector_sun +
                   self.MU_EARTH / np.power(r_mag_earth, 3) * r_vector_earth +
                   self.MU_MOON / np.power(r_mag_moon, 3) * r_vector_moon)
        return np.concatenate(state_0)

    def _get_reward(self):
        position_threshold = 600e3
        velocity_threshold = 400

        goal_achieved_reward = 0
        if np.linalg.norm(self.delta_position) <= position_threshold:
            self.terminated_condition = True
            goal_achieved_reward = 1000

        time_penalty = 0
        if self.time_step > self.max_time_steps:
            self.truncated_condition = True
            time_penalty = -10

        # fuel_penalty = 0
        # if self.fuel_mass < 0:
        #     self.truncated_condition = True
        #     fuel_penalty = -10

        # moon_region_penalty = 0
        # if np.linalg.norm(self.spacecraft_position - self.source_planet.eph(self.current_epoch)[0]) < 1737e3 + 300e3:
        #     moon_region_penalty = -10
        #     self.truncated_condition = True

        # earth_region_penalty = 0
        # if np.linalg.norm(
        #         self.spacecraft_position - self.destination_planet.eph(self.current_epoch)[0]) < 6738e3 + 300e3:
        #     earth_region_penalty = -10
        #     self.truncated_condition = True

        space_penalty = 0
        # if np.linalg.norm(self.spacecraft_position) > 1.5e9:
        #     space_penalty = -600
        #     self.truncated_condition = True


        # based on magnitude
        change_in_delta_mag = ( np.linalg.norm(self.previous_spacecraft_position - self.target_position)
                               - np.linalg.norm(self.delta_position) ) \
                              / (self.EARTH_MOON_MEAN_DISTANCE - self.destination_object_orbit_radius)
        delta_mag_reward = 100 * change_in_delta_mag


        # based on direction
        intended_direction_vector = self.target_position - self.previous_spacecraft_position
        direction_vector = self.spacecraft_position - self.previous_spacecraft_position
        cosine_similarity = np.inner(intended_direction_vector, direction_vector) \
                            / (np.linalg.norm(intended_direction_vector) * np.linalg.norm(direction_vector))
        cosine_reward = 2 * cosine_similarity

        reward = delta_mag_reward + goal_achieved_reward  + cosine_reward
        # print(positional_reward, mass_reward, velocity_reward)
        reward_components = [delta_mag_reward, cosine_reward, goal_achieved_reward]

        return reward, reward_components, self.truncated_condition, self.terminated_condition
