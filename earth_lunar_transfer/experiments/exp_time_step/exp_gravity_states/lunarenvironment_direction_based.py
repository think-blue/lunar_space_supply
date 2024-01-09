from experiments.reference_exp.lunarenvironment import LunarEnvironment
from gymnasium.spaces import Box, Dict
import numpy as np
from scipy.integrate import odeint


class LunarEnvForceHelper(LunarEnvironment):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.gravitation_force = None
        self.observation_space = Dict(
            {
                "position": Box(low=-800, high=800, shape=(3,)),
                "velocity": Box(low=-800, high=800, shape=(3,)),
                "mass": Box(low=0, high=1, shape=(1,)),
                "delta_position": Box(low=-800, high=800, shape=(3,)),
                "delta_velocity": Box(low=-500, high=500, shape=(3,)),
                "time_step": Box(low=0, high=1.2, shape=(1,)),
                "acting_force": Box(low=-50, high=50, shape=(3,)),
                "moon_pos": Box(low=-800, high=800, shape=(3,)),
                "earth_pos": (Box(low=-800, high=800, shape=(3,)))
            }
        )
        # print(self.observation_space)

    def reset(self, *, seed=None, options=None):
        self.normalised_state, info = super().reset()

        ###### extra addition
        self.state["earth_pos"] = np.array(self.destination_planet.eph(self.current_epoch)[0])
        self.state["moon_pos"] = np.array(self.source_planet.eph(self.current_epoch)[0])
        self.state["acting_force"] = np.array(self.forces[1] + self.forces[2] + self.forces[3])

        self.normalised_state["earth_pos"] = self.state["earth_pos"] / self.EARTH_MOON_MEAN_DISTANCE
        self.normalised_state["moon_pos"] = self.state["moon_pos"] / self.EARTH_MOON_MEAN_DISTANCE
        self.normalised_state["acting_force"] = self.state["acting_force"] / 4.
        return self.normalised_state, info

    def step(self, action):

        self._store_episode_history()

        self.previous_spacecraft_position = self.spacecraft_position
        self.previous_spacecraft_velocity = self.spacecraft_velocity

        # todo: test this function out
        if self.env_config["action_space"] == "discrete":
            action = self.transform_action(action)
        self.action = action

        time_delta = self.time_step_duration * 24 * 3600  # in seconds
        num_steps = self.integration_steps
        time_array = np.arange(0, time_delta, num_steps)
        detailed_spacecraft_state = odeint(self.accelerate,
                                           y0=np.concatenate([self.spacecraft_position, self.spacecraft_velocity],
                                                             axis=0),
                                           # t = time_array,
                                           t=[0, time_delta],
                                           # todo: verify this function and it's working
                                           args=(action, (self.payload_mass + self.fuel_mass),
                                                 self.current_epoch))
        ##########################

        ######################################

        # todo: check what odeint.T does
        spacecraft_pos = np.array(detailed_spacecraft_state[-1, :3])
        spacecraft_vel = np.array(detailed_spacecraft_state[-1, 3:])

        # todo: verify mass ejected function
        mass_ejected = self._mass_ejected(action, len(time_array))

        self._update_state(
            fuel_mass=self.fuel_mass - mass_ejected,
            position=spacecraft_pos,
            velocity=spacecraft_vel,
            epoch=self.current_epoch + self.time_step_duration,
            time_step=self.time_step + 1,
            target_position=None,
            target_velocity=None
        )

        self.forces = self.accelerate_components(np.concatenate([self.spacecraft_position, self.spacecraft_velocity],
                                                                axis=0), action, self.payload_mass + self.fuel_mass,
                                                 self.current_epoch)

        self.reward, self.reward_components, self.truncated_condition, self.terminated_condition = self._get_reward()

        self.state = dict(
            delta_position=self.delta_position,
            delta_velocity=self.delta_velocity,
            mass=np.array([self.spacecraft_mass]),
            position=self.spacecraft_position,
            time_step=np.array([self.time_step]),
            velocity=self.spacecraft_velocity,

        )

        self.normalised_state = self._normalise_state(self.state)

        ############################
        self.state["earth_pos"] = np.array(self.destination_planet.eph(self.current_epoch)[0])
        self.state["moon_pos"] = np.array(self.source_planet.eph(self.current_epoch)[0])
        self.state["acting_force"] = np.array(self.forces[1] + self.forces[2] + self.forces[3])
        self.normalised_state["earth_pos"] = self.state["earth_pos"] / self.EARTH_MOON_MEAN_DISTANCE
        self.normalised_state["moon_pos"] = self.state["moon_pos"] / self.EARTH_MOON_MEAN_DISTANCE
        self.normalised_state["acting_force"] = self.state["acting_force"] / 4.
        #############################

        self._store_episode_history()
        info = {}
        return self.normalised_state, self.reward, self.terminated_condition, self.truncated_condition, info

    def _get_reward(self):
        position_threshold = 600e3
        velocity_threshold = 400

        goal_achieved_reward = 0
        if np.linalg.norm(self.delta_position) <= position_threshold and np.linalg.norm(
                self.delta_velocity) <= velocity_threshold:
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

        moon_region_penalty = 0
        if np.linalg.norm(self.spacecraft_position - self.source_planet.eph(self.current_epoch)[0]) < 1737e3 + 300e3:
            moon_region_penalty = -1000
            self.truncated_condition = True

        earth_region_penalty = 0
        if np.linalg.norm(
                self.spacecraft_position - self.destination_planet.eph(self.current_epoch)[0]) < 6738e3 + 300e3:
            earth_region_penalty = -1000
            self.truncated_condition = True

        # space_penalty = 0
        # if np.linalg.norm(self.spacecraft_position) > 1.5e9:
        #     space_penalty = -1000
        #     self.truncated_condition = True

        # based on magnitude (position)
        change_in_delta_mag = (np.linalg.norm(self.previous_spacecraft_position - self.target_position)
                               - np.linalg.norm(self.delta_position)) \
                              / (self.EARTH_MOON_MEAN_DISTANCE - self.destination_object_orbit_radius)
        delta_mag_reward = 100 * 5 * change_in_delta_mag

        # based on magnitude (velocity)
        change_in_delta_mag_vel = (np.linalg.norm(self.previous_spacecraft_velocity - self.target_velocity)
                                   - np.linalg.norm(self.delta_velocity)) \
                                  / (self.MOON_SPEED_WRT_EARTH)
        delta_mag_reward_vel = 50 * change_in_delta_mag_vel

        # based on direction (position)
        intended_direction_vector = self.target_position - self.previous_spacecraft_position
        direction_vector = self.spacecraft_position - self.previous_spacecraft_position
        cosine_similarity = np.inner(intended_direction_vector, direction_vector) \
                            / (np.linalg.norm(intended_direction_vector) * np.linalg.norm(direction_vector))
        cosine_reward = 10 * cosine_similarity

        # based on direction (velocity)
        cosine_similarity_vel_old = np.inner(self.target_velocity, self.previous_spacecraft_velocity) / (
                np.linalg.norm(self.target_velocity) * np.linalg.norm(self.previous_spacecraft_velocity)
        )
        cosine_similarity_vel_new = np.inner(self.target_velocity, self.spacecraft_velocity) / (
                np.linalg.norm(self.target_velocity) * np.linalg.norm(self.spacecraft_velocity))

        cosine_vel_reward = cosine_similarity_vel_new - cosine_similarity_vel_old

        reward = delta_mag_reward + delta_mag_reward_vel + cosine_reward + \
                 + time_penalty + fuel_penalty + earth_region_penalty + moon_region_penalty + goal_achieved_reward

        # print(positional_reward, mass_reward, velocity_reward)
        reward_components = [delta_mag_reward, cosine_reward, delta_mag_reward_vel, time_penalty, fuel_penalty,
                             earth_region_penalty, moon_region_penalty, goal_achieved_reward]

        return reward, reward_components, self.truncated_condition, self.terminated_condition
