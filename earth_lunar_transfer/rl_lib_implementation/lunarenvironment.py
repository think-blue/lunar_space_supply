import csv
import os.path

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiDiscrete
import pykep as pk
from scipy.integrate import odeint
import plotly.graph_objects as go
from time import time


class LunarEnvironment(gym.Env, object):
    MU_MOON = 4.9048695e12
    MU_EARTH = pk.MU_EARTH
    MU_SUN = 0  # pk.MU_SUN
    EARTH_MOON_MEAN_DISTANCE = 384467e3
    MOON_DISTANCE_FROM_BARYCENTER = 384467e3 - 4671e3
    MOON_SPEED_WRT_EARTH = 1023.056

    def __init__(self, env_config):
        """the environment where the
        target position and target velocity is fixed"""

        self.render_mode = "ansi"
        self.env_config = env_config
        if self.env_config["action_space"] == "discrete":
            self.action_space = MultiDiscrete([10, 10, 10])
        else:
            self.action_space = Box(-1, 1, (3,), dtype=np.float32)

        self.observation_space = Dict(
            {
                "position": Box(low=-10, high=10, shape=(3,)),
                "velocity": Box(low=-50, high=50, shape=(3,)),
                "mass": Box(low=0, high=1, shape=(1,)),
                "delta_position": Box(low=-10, high=10, shape=(3,)),
                "delta_velocity": Box(low=-50, high=50, shape=(3,)),
                # todo: debug this time step getting out of bounds
                "time_step": Box(low=0, high=1.2, shape=(1,))
            }
        )
        self.reward_range = None

        # spacecraft variables
        self.payload_mass = env_config["payload_mass"]
        self.specific_impulse = env_config["specific_impulse"]

        # time variables
        self.num_epochs = env_config["num_days"]  # 5 days
        self.time_step_duration = env_config["time_step_duration"]  # 1/48
        self.max_time_steps = self.num_epochs / self.time_step_duration

        # orbit's radius
        self.source_object_orbit_radius = env_config["source_object_orbit_radius"]
        self.destination_object_orbit_radius = env_config["dest_object_orbit_radius"]
        self.source_inclination_angle = env_config["source_inclination_angle"]  # phi
        self.source_azimuthal_angle = env_config["source_azimuthal_angle"]  # theta
        self.dest_inclination_angle = env_config["dest_inclination_angle"]  # phi
        self.dest_azimuthal_angle = env_config["dest_azimuthal_angle"]  # theta

        # planets
        kernel_path = env_config["kernel_path"]
        pk.util.load_spice_kernel(kernel_path)

        self.observer = env_config["observer_body"]
        self.source_planet = pk.planet.spice(env_config["source_planet"], self.observer)
        self.destination_planet = pk.planet.spice(env_config["dest_planet"], self.observer)
        self.sun = pk.planet.spice("sun", self.observer)

        # integration_variables
        self.integration_steps = env_config["integration_steps"]

        # changing variables
        self.fuel_mass = None
        self.current_epoch = None
        self.target_position = None
        self.target_velocity = None

        # state variables
        self.spacecraft_mass = None
        self.spacecraft_velocity = None
        self.spacecraft_position = None
        self.delta_position = None
        self.delta_velocity = None
        self.time_step = None

        self.position_history = []
        self.velocity_history = []
        self.epoch_history = []

        self.training_data_path = env_config["training_data_path"]
        self.write_flag = 0
        print(id(self))

    # def __new__(cls, env_config, *args, **kwargs):
    #     if not hasattr(cls, '_instance'):
    #         cls._instance = super().__new__(cls, *args, **kwargs)
    #     return cls._instance

    def reset(self, *, seed=None, options=None):
        """resets the environment to the initial state based on the environment config parameters passed"""
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
        destination_planet_eph = self.destination_planet.eph(self.current_epoch)
        target_speed = self.get_orbital_speed(self.destination_object_orbit_radius, self.MU_EARTH)
        target_position, target_velocity = self.get_eph_from_orbital_angles(self.dest_azimuthal_angle,
                                                                            self.dest_inclination_angle,
                                                                            self.destination_object_orbit_radius,
                                                                            target_speed,
                                                                            destination_planet_eph)
        self.time_step = 0
        state = dict(
            delta_position=spacecraft_position - target_position,
            delta_velocity=spacecraft_velocity - target_velocity,
            mass=np.array([spacecraft_mass]),
            position=spacecraft_position,
            time_step=np.array([self.time_step]),
            velocity=spacecraft_velocity)

        self.spacecraft_mass = state["mass"].item()
        self.spacecraft_position = state["position"]
        self.spacecraft_velocity = state["velocity"]
        self.delta_position = state["delta_position"]
        self.delta_velocity = state["delta_velocity"]
        self.time_step = state["time_step"].item()

        self.target_position = target_position
        self.target_velocity = target_velocity

        info = {}

        state = self._normalise_state(state)
        return state, info

    @staticmethod
    def get_orbital_speed(radius, mu):
        speed = np.sqrt(mu/radius)
        return speed

    @staticmethod
    def get_eph_from_orbital_angles(theta, phi, radius, speed, moon_eph):
        relative_position = radius * np.array([np.sin(theta) * np.cos(phi),
                                               np.sin(theta) * np.sin(phi),
                                               np.cos(theta)])
        position = np.array(moon_eph[0]) + relative_position

        relative_velocity = speed * np.array([np.sin(theta + np.pi / 2) * np.cos(phi),
                                              np.sin(theta + np.pi / 2) * np.sin(phi),
                                              np.cos(theta + np.pi / 2)])
        velocity = relative_velocity + np.array(moon_eph[1])
        return position, velocity

    def step(self, action):

        # todo: test this function out
        if self.env_config["action_space"] == "discrete":
            action = self.transform_action(action)

        # todo: add threshold for position and velocity
        position_threshold = 0
        velocity_threshold = 0

        # terminal state
        terminated = False
        truncated = False
        info = {}

        if np.linalg.norm(self.delta_position) <= position_threshold \
                and np.linalg.norm(self.delta_velocity) <= velocity_threshold:
            terminated = True

        # truncated state time limit, out of fuel todo: add out of bounds values
        if self.time_step > self.max_time_steps or self.fuel_mass < 0:
            truncated = True

        time_delta = self.time_step_duration * 24 * 3600  # in seconds
        num_steps = self.integration_steps

        # todo: verify time array
        time_array = np.arange(0, time_delta, num_steps)
        # implement new state of mass
        detailed_spacecraft_state = odeint(self.accelerate,
                                           y0=np.concatenate([self.spacecraft_position, self.spacecraft_velocity],
                                                             axis=0),
                                           t=[0, time_delta],
                                           # todo: verify this function and it's working
                                           args=(action, (self.payload_mass + self.fuel_mass),
                                                 self.current_epoch))
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

        reward = self._get_reward()

        state = dict(
            delta_position=self.delta_position,
            delta_velocity=self.delta_velocity,
            mass=np.array([self.spacecraft_mass]),
            position=self.spacecraft_position,
            time_step=np.array([self.time_step]),
            velocity=self.spacecraft_velocity,

        )

        state = self._normalise_state(state)

        # self._store_history()
        self.render()
        return state, reward, terminated, truncated, info

        # return reward, reward_components, detailed_spacecraft_state

    def _store_history(self):
        # todo: make it a csv writer
        self.position_history.append(self.spacecraft_position.tolist())
        self.velocity_history.append(self.spacecraft_velocity.tolist())
        self.epoch_history.append(self.current_epoch)

    def _get_reward(self):
        """
        Everything is in SI units
        """
        # static destination based on the end epoch
        dest_position = self.target_position
        position_error = (self.spacecraft_position - dest_position)
        positional_error_magnitude = np.linalg.norm(position_error) / (
                self.EARTH_MOON_MEAN_DISTANCE - self.destination_object_orbit_radius)
        positional_reward = - positional_error_magnitude

        mass_reward = -(1 - (self.fuel_mass / self.env_config["fuel_mass"]))

        velocity_reward = - np.linalg.norm(
            self.spacecraft_velocity - self.target_velocity) / self.MOON_SPEED_WRT_EARTH  # astronomical units
        reward = 10 + positional_reward + mass_reward + velocity_reward
        # print(positional_reward, mass_reward, velocity_reward)
        return reward

    def _update_state(self, fuel_mass, position, velocity, epoch, time_step, target_position, target_velocity):
        self.fuel_mass = fuel_mass
        self.spacecraft_mass = self.payload_mass + self.fuel_mass
        self.spacecraft_position = position
        self.spacecraft_velocity = velocity
        self.current_epoch = epoch
        self.time_step = time_step

        if target_velocity is not None and target_position is not None:
            self.target_velocity = target_velocity
            self.target_position = target_position

        self.delta_position = self.spacecraft_position - self.target_position
        self.delta_velocity = self.spacecraft_velocity - self.target_velocity

    def _normalise_state(self, state):
        state["mass"] = (state["mass"] - self.env_config["payload_mass"]) / (
            self.env_config["payload_mass"])  # min max norm
        state["position"] = state["position"] / self.EARTH_MOON_MEAN_DISTANCE
        # todo: check velocity normalisation values
        state["velocity"] = state["velocity"] / self.MOON_SPEED_WRT_EARTH
        state["delta_position"] = state["delta_position"] / (
                self.EARTH_MOON_MEAN_DISTANCE - self.destination_object_orbit_radius)
        state["delta_velocity"] = state["delta_velocity"] / self.MOON_SPEED_WRT_EARTH
        state["time_step"] = state["time_step"] / self.max_time_steps
        return state

    def render(self):
        # if not self.write_flag:
        #     with open("csv_file.txt", "w") as csv_file:
        #         csv_writer = csv.writer(csv_file)
        #         csv_writer.writerow(["epoch", "x", "y", "z"])
        #         self.write_flag = 1

        # create file
        file_name = f"{id(self)}.csv"
        # if self.write_flag == 0:
        #     current_time = time()
        #     file_name = f"{current_time}.csv"
        #     self.write_flag = 1

        with open(f"{self.training_data_path}/{file_name}", "a") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_epoch] + self.spacecraft_position.tolist())
        # print("in render")
        return None

    def _mass_ejected(self, thrust, time):
        g_0 = 9.8
        thrust_mag = np.linalg.norm(thrust)
        mass_derivative = thrust_mag / (g_0 * self.specific_impulse)
        return mass_derivative * time

    def accelerate(self, state, time, thrust, spacecraft_mass, epoch):
        position = state[0:3]
        velocity = state[3:]

        r_vector_sun = np.array(self.sun.eph(epoch))[0] - position
        r_mag_sun = np.linalg.norm(r_vector_sun)

        r_vector_moon = np.array(self.source_planet.eph(epoch))[0] - position
        r_mag_moon = np.linalg.norm(r_vector_moon)

        r_vector_earth = np.array(self.destination_planet.eph(epoch))[0] - position
        r_mag_earth = np.linalg.norm(r_vector_earth)

        acceleration = (velocity,
                        thrust / spacecraft_mass +
                        self.MU_SUN / np.power(r_mag_sun, 3) * r_vector_sun +
                        self.MU_EARTH / np.power(r_mag_earth, 3) * r_vector_earth +
                        self.MU_MOON / np.power(r_mag_moon, 3) * r_vector_moon)
        return np.concatenate(acceleration)

    @staticmethod
    def transform_action(action):
        output_start = -.1
        output_end = .1
        input_start = 0
        input_end = 9
        transformed_action = output_start + ((output_end - output_start) / (input_end - input_start)) * (
                action - input_start)
        return transformed_action

    @staticmethod
    def simulate(epoch_history, position_history, source_planet, destination_planet, path, display=False):
        source_data = []
        destination_data = []
        for epoch in epoch_history:
            source_data_epoch = source_planet.eph(epoch)[0]
            destination_data_epoch = destination_planet.eph(epoch)[0]
            source_data.append(source_data_epoch)
            destination_data.append(destination_data_epoch)
        source_data, destination_data, position_data = np.array(source_data), np.array(destination_data), np.array(
            position_history)

        figure_len = 9e8
        figure = go.Figure(
            data=[go.Scatter3d(x=[source_data[0, 0]],
                               y=[source_data[0, 1]],
                               z=[source_data[0, 2]],
                               name="moon"),
                  go.Scatter3d(x=[destination_data[0, 0]],
                               y=[destination_data[0, 1]],
                               z=[destination_data[0, 2]],
                               name="earth"),
                  go.Scatter3d(x=[position_data[0, 0]],
                               y=[position_data[0, 1]],
                               z=[position_data[0, 2]],
                               name="spacecraft")
                  ],
            layout=go.Layout(
                scene=dict(xaxis_range=[-figure_len, figure_len],
                           yaxis_range=[-figure_len, figure_len],
                           zaxis_range=[-figure_len, figure_len],
                           xaxis=dict(
                               backgroundcolor='rgb(255, 255, 255)'),
                           yaxis=dict(
                               backgroundcolor='rgb(255, 255, 255)'),
                           zaxis=dict(
                               backgroundcolor='rgb(255, 255, 255)')
                           ),
                updatemenus=[dict(type="buttons",
                                  buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None])])
                             ]
            ),
            frames=[go.Frame
                    (data=[go.Scatter3d(x=[source_data[i, 0]],
                                        y=[source_data[i, 1]],
                                        z=[source_data[i, 2]],
                                        name="moon"),
                           go.Scatter3d(x=[destination_data[i, 0]],
                                        y=[destination_data[i, 1]],
                                        z=[destination_data[i, 2]],
                                        name="earth"),
                           go.Scatter3d(x=[position_data[i, 0]],
                                        y=[position_data[i, 1]],
                                        z=[position_data[i, 2]],
                                        name="spacecraft")]
                     ) for i in range(len(source_data))]
        )

        figure.write_html(path)

        if display:
            figure.show()
