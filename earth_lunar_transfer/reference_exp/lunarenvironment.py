import csv
import os.path

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiDiscrete
import pykep as pk
from pyparsing import nestedExpr
from scipy.integrate import odeint
import plotly.graph_objects as go
import mlflow


class LunarEnvironment(gym.Env, object):
    """
    description of the environment
    """
    MU_MOON = 4.9048695e12
    MU_EARTH = pk.MU_EARTH
    MU_SUN = 0  # pk.MU_SUN
    EARTH_MOON_MEAN_DISTANCE = 384467e3
    MOON_DISTANCE_FROM_BARYCENTER = 384467e3 - 4671e3
    MOON_SPEED_WRT_EARTH = 1023.056

    def __init__(self, env_config):
        """the environment where the
        target position and target velocity is fixed"""

        self.action = None
        self.start_velocity = None
        self.start_position = None
        self.forces = None
        self.state = None
        self.normalised_state = None
        self.reward = None
        self.reward_components = None
        self.render_mode = "ansi"
        self.env_config = env_config

        if self.env_config["action_space"] == "discrete":
            # make it (2*n + 1)
            self.action_space = MultiDiscrete([11, 11, 11])
        else:
            self.action_space = Box(-env_config["max_thrust"], env_config["max_thrust"], (3,), dtype=np.float32)

        self.observation_space = Dict(
            {
                "position": Box(low=-800, high=800, shape=(3,)),
                # "velocity": Box(low=-500, high=500, shape=(3,)),
                "mass": Box(low=0, high=1, shape=(1,)),
                "delta_position": Box(low=-800, high=800, shape=(3,)),
                # "delta_velocity": Box(low=-500, high=500, shape=(3,)),
                # todo: debug this time step getting out of bounds
                # "time_step": Box(low=0, high=1.2, shape=(1,)),
                # todo: resultant force and moon and earth position
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

        self.previous_spacecraft_position = None
        self.previous_spacecraft_velocity = None

        self.fuel_mass_history = []
        self.position_history = []
        # self.velocity_history = []
        self.delta_position_history = []
        # self.delta_velocity_history = []
        self.time_step_history = []
        self.epoch_history = []
        self.reward_history = []
        self.reward_components_history = []

        self.training_data_path = env_config["data_path"]
        self.object_id = f"{id(self)}"
        self.experiment_name = env_config["exp_name"]
        self.episode_count = 0

        self.terminated_condition = False
        self.truncated_condition = False

        if not os.path.exists(os.path.join(self.training_data_path, self.experiment_name)):
            os.makedirs(os.path.join(self.training_data_path, self.experiment_name))

        self.save_training_data_path = os.path.join(self.training_data_path, self.experiment_name,
                                                    self.object_id + ".csv")

        if env_config["mlflow_configured"]:
            with mlflow.start_run(run_id=env_config["mlflow_run_id"], nested=True):
                mlflow.log_param(f"train_csv_data_path_{id(self)}", self.save_training_data_path)
                pass
        with open(f"{self.save_training_data_path}", "a") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["mass", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
                                 "delta_pos_x", "delta_pos_y", "delta_pos_z", "delta_vel_x", "delta_vel_y",
                                 "delta_vel_z",
                                 "time_step", "epoch", "episode", "reward"] + [f"reward_{i}" for i in range(10)])

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
        self.state = dict(
            delta_position=spacecraft_position - target_position,
            # delta_velocity=spacecraft_velocity - target_velocity,
            mass=np.array([spacecraft_mass]),
            position=spacecraft_position,
            # time_step=np.array([self.time_step]),
            # velocity=spacecraft_velocity
        )

        self.spacecraft_mass = self.state["mass"].item()
        self.spacecraft_position = self.state["position"]
        self.spacecraft_velocity = spacecraft_velocity #self.state["velocity"]
        self.delta_position = self.state["delta_position"]
        # self.delta_velocity = self.state["delta_velocity"]
        self.time_step = self.time_step #self.state["time_step"].item()
        self.reward = 0
        self.reward_components = None

        self.start_position = self.spacecraft_position
        self.start_velocity = self.spacecraft_velocity
        self.target_position = target_position
        self.target_velocity = target_velocity

        self.previous_spacecraft_position = self.spacecraft_position
        self.previous_spacecraft_velocity = self.spacecraft_velocity

        action = np.array([0, 0, 0])
        self.forces = self.accelerate_components(np.concatenate([self.spacecraft_position, self.spacecraft_velocity],
                                                             axis=0),action , self.payload_mass + self.fuel_mass, self.current_epoch)

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

    @staticmethod
    def get_orbital_speed(radius, mu):
        speed = np.sqrt(mu / radius)
        return speed

    @staticmethod
    def get_eph_from_orbital_angles(theta, phi, radius, speed, planet_eph):
        relative_position = radius * np.array([np.sin(theta) * np.cos(phi),
                                               np.sin(theta) * np.sin(phi),
                                               np.cos(theta)])
        position = np.array(planet_eph[0]) + relative_position

        relative_velocity = speed * np.array([np.sin(theta + np.pi / 2) * np.cos(phi),
                                              np.sin(theta + np.pi / 2) * np.sin(phi),
                                              np.cos(theta + np.pi / 2)])
        velocity = relative_velocity + np.array(planet_eph[1])
        return position, velocity

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
            # delta_velocity=self.delta_velocity,
            mass=np.array([self.spacecraft_mass]),
            position=self.spacecraft_position,
            # time_step=np.array([self.time_step]),
            # velocity=self.spacecraft_velocity,

        )

        self.normalised_state = self._normalise_state(self.state)

        self._store_episode_history()
        info = {}
        return self.normalised_state, self.reward, self.terminated_condition, self.truncated_condition, info

    def _store_episode_history(self):
        # todo: make it a csv writer
        if self.reward_components is not None:
            self.position_history.append(self.spacecraft_position.tolist())
            self.velocity_history.append(self.spacecraft_velocity.tolist())
            self.epoch_history.append(self.current_epoch)
            self.delta_position_history.append(self.delta_position)
            self.delta_velocity_history.append(self.delta_velocity)
            self.fuel_mass_history.append(self.fuel_mass)
            self.time_step_history.append(self.time_step)
            self.reward_history.append(self.reward)
            self.reward_components_history.append(self.reward_components)

    def _write_episode_history(self):
        with open(f"{self.save_training_data_path}", "a") as csv_file:
            csv_writer = csv.writer(csv_file)
            episode_array = np.ones(shape=(len(self.fuel_mass_history), 1)) * self.episode_count

            if not self.position_history == []:
                data = np.hstack(
                    [
                        np.array(self.fuel_mass_history).reshape(-1, 1),
                        self.position_history,
                        self.velocity_history,
                        self.delta_position_history,
                        self.delta_velocity_history,
                        np.array(self.time_step_history).reshape(-1, 1),
                        np.array(self.epoch_history).reshape(-1, 1),
                        episode_array,
                        np.array(self.reward_history).reshape(-1, 1),
                        self.reward_components_history
                    ]
                )
                csv_writer.writerows(data.tolist())

    def _get_reward(self):
        """
        Everything is in SI units
        """

        position_threshold = 10000
        velocity_threshold = 40

        goal_achieved_reward = 0
        if np.linalg.norm(self.delta_position) <= position_threshold \
                and np.linalg.norm(self.delta_velocity) <= velocity_threshold:
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

        # todo: add out of bounds values

        # static destination based on the end epoch
        dest_position = self.target_position
        position_error = (self.spacecraft_position - dest_position)
        positional_error_magnitude = np.linalg.norm(position_error) / (
                self.EARTH_MOON_MEAN_DISTANCE - self.destination_object_orbit_radius)
        positional_reward = - positional_error_magnitude

        mass_reward = -(1 - (self.fuel_mass / self.env_config["fuel_mass"]))

        velocity_reward = - np.linalg.norm(
            self.spacecraft_velocity - self.target_velocity) / self.MOON_SPEED_WRT_EARTH  # astronomical units

        reward = 10 + positional_reward + mass_reward + velocity_reward + time_penalty + fuel_penalty + goal_achieved_reward

        reward_components = [positional_reward, mass_reward, velocity_reward, time_penalty, fuel_penalty,
                             goal_achieved_reward]
        return reward, reward_components, self.truncated_condition, self.terminated_condition

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
        n_state = {
            "mass": (state["mass"] - self.env_config["payload_mass"]) / (
                self.env_config["payload_mass"]),
            "position": state["position"] / self.EARTH_MOON_MEAN_DISTANCE,
            # "velocity": state["velocity"] / self.MOON_SPEED_WRT_EARTH * 3,  # np.linalg.norm(self.target_velocity)
            "delta_position": state["delta_position"] / (
                    self.EARTH_MOON_MEAN_DISTANCE - self.destination_object_orbit_radius),
            # "delta_velocity": state["delta_velocity"] / self.MOON_SPEED_WRT_EARTH * 3,
            # "time_step": state["time_step"] / self.max_time_steps
        }
        # todo: check velocity normalisation values
        # state["velocity"] = state["velocity"] / self.MOON_SPEED_WRT_EARTH

        return n_state

    def render(self):
        return None

    def _mass_ejected(self, thrust, time):
        g_0 = 9.8
        thrust_mag = np.linalg.norm(thrust)
        mass_derivative = thrust_mag / (g_0 * self.specific_impulse)
        return mass_derivative * time

    def accelerate(self, state_0, time, thrust, spacecraft_mass, epoch):
        position = state_0[0:3]
        velocity = state_0[3:]

        r_vector_sun = np.array(self.sun.eph(epoch))[0] - position
        r_mag_sun = np.linalg.norm(r_vector_sun)

        r_vector_moon = np.array(self.source_planet.eph(epoch))[0] - position
        r_mag_moon = np.linalg.norm(r_vector_moon)

        r_vector_earth = np.array(self.destination_planet.eph(epoch))[0] - position
        r_mag_earth = np.linalg.norm(r_vector_earth)

        state_0 = (velocity,
                   thrust / spacecraft_mass +
                   self.MU_SUN / np.power(r_mag_sun, 3) * r_vector_sun +
                   self.MU_EARTH / np.power(r_mag_earth, 3) * r_vector_earth +
                   self.MU_MOON / np.power(r_mag_moon, 3) * r_vector_moon)
        return np.concatenate(state_0)

    def accelerate_components(self, state_0, thrust, spacecraft_mass, epoch):
        position = state_0[0:3]
        velocity = state_0[3:]

        r_vector_sun = np.array(self.sun.eph(epoch))[0] - position
        r_mag_sun = np.linalg.norm(r_vector_sun)

        r_vector_moon = np.array(self.source_planet.eph(epoch))[0] - position
        r_mag_moon = np.linalg.norm(r_vector_moon)

        r_vector_earth = np.array(self.destination_planet.eph(epoch))[0] - position
        r_mag_earth = np.linalg.norm(r_vector_earth)
        return (
            thrust / spacecraft_mass,
            self.MU_SUN / np.power(r_mag_sun, 3) * r_vector_sun,
            self.MU_EARTH / np.power(r_mag_earth, 3) * r_vector_earth,
            self.MU_MOON / np.power(r_mag_moon, 3) * r_vector_moon
        )


    def transform_action(self, action):
        output_start = - self.env_config["max_thrust"]
        output_end = self.env_config["max_thrust"]
        input_start = 0
        input_end = self.action_space.nvec[0] - 1
        transformed_action = output_start + ((output_end - output_start) / (input_end - input_start)) * (
                action - input_start)
        return transformed_action

    @staticmethod
    def simulate(epoch_history, position_history, source_planet, destination_planet, path, source_point,
                 destination_point, display=False):
        source_data, destination_data = LunarEnvironment._get_planetary_data(destination_planet, epoch_history,
                                                                             source_planet)
        source_data, destination_data, position_data = np.array(source_data), np.array(destination_data), np.array(
            position_history)
        source_point = go.Scatter3d(x=[source_point[0], destination_point[0]],
                                    y=[source_point[1], destination_point[1]],
                                    z=[source_point[2], destination_point[2]],
                                    mode="markers")
        spacecraft_plot = go.Scatter3d(x=position_data[:, 0],
                                       y=position_data[:, 1],
                                       z=position_data[:, 2],
                                       mode="lines",
                                       name="Spacecraft")
        source_plot = go.Scatter3d(x=source_data[:, 0],
                                   y=source_data[:, 1],
                                   z=source_data[:, 2],
                                   mode="lines",
                                   name="moon")
        dest_plot = go.Scatter3d(x=destination_data[:, 0],
                                 y=destination_data[:, 1],
                                 z=destination_data[:, 2],
                                 mode="lines",
                                 name="earth")

        figure = go.Figure(data=[source_point, spacecraft_plot, source_plot, dest_plot])
        figure.update_layout(scene_aspectmode='data')
        figure.write_html(f"{path}")

    @staticmethod
    def animate(epoch_history, position_history, source_planet, destination_planet, path, display=False):
        source_data, destination_data = LunarEnvironment._get_planetary_data(destination_planet, epoch_history,
                                                                             source_planet)
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
                  go.Scatter3d(x=position_data[:, 0],
                               y=position_data[:, 1],
                               z=position_data[:, 2],
                               name="spacecraft",
                               mode="lines")
                  ],
            layout=go.Layout(
                scene=dict(xaxis_range=[-figure_len, figure_len],
                           yaxis_range=[-figure_len, figure_len],
                           zaxis_range=[-figure_len, figure_len],
                           xaxis=dict(
                               backgroundcolor='rgb(128, 128, 128)'),
                           yaxis=dict(
                               backgroundcolor='rgb(128, 128, 128)'),
                           zaxis=dict(
                               backgroundcolor='rgb(128, 128, 128)')
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
                                        name="spacecraft", mode="markers+lines")]
                     ) for i in range(len(source_data))]
        )

        figure.write_html(path)

        if display:
            figure.show()

    @staticmethod
    def _get_planetary_data(destination_planet, epoch_history, source_planet):
        source_data = []
        destination_data = []
        for epoch in epoch_history:
            source_data_epoch = source_planet.eph(epoch)[0]
            destination_data_epoch = destination_planet.eph(epoch)[0]
            source_data.append(source_data_epoch)
            destination_data.append(destination_data_epoch)
        return source_data, destination_data
