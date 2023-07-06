import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import pykep as pk
from scipy.integrate import odeint


class LunarEnvironment(gym.Env):
    MU_MOON = 4.9048695e12
    MU_EARTH = pk.MU_EARTH
    MU_SUN = pk.MU_SUN

    def __init__(self, env_config):
        """the environment where the
        target position and target velocity is fixed"""

        self.env_config = env_config
        self.action_space = Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = Dict(
            {
                "position": Box(low=-1000, high=1000, shape=(3,)),
                "velocity": Box(low=-1000, high=1000, shape=(3,)),
                "mass": Box(low=0, high=1000, shape=(1,)),
                "delta_position": Box(low=-1000, high=1000, shape=(3,)),
                "delta_velocity": Box(low=-1000, high=1000, shape=(3,)),
                "time_step": Box(low=0, high=100, shape=(1,))
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
        pk.util.load_spice_kernel("../kernels/de441.bsp")
        self.source_planet = pk.planet.spice(env_config["source_planet"])
        self.destination_planet = pk.planet.spice(env_config["dest_planet"])

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

    def reset(self, seed=None, options=None):
        """resets the environment to the initial state based on the environment config parameters passed"""
        self.fuel_mass = self.env_config["fuel_mass"]
        self.current_epoch = self.env_config["start_epoch"]

        spacecraft_mass = self.env_config["payload_mass"] + self.env_config["fuel_mass"]
        source_planet_eph = self.source_planet.eph(self.current_epoch)
        spacecraft_initial_speed = self._get_orbital_speed(self.source_object_orbit_radius, self.MU_MOON)
        spacecraft_position, spacecraft_velocity = self._get_eph_from_orbital_angles(self.source_azimuthal_angle,
                                                                                     self.source_inclination_angle,
                                                                                     self.source_object_orbit_radius,
                                                                                     spacecraft_initial_speed,
                                                                                     source_planet_eph)
        destination_planet_eph = self.destination_planet.eph(self.current_epoch)
        target_speed = self._get_orbital_speed(self.destination_object_orbit_radius, self.MU_EARTH)
        target_position, target_velocity = self._get_eph_from_orbital_angles(self.dest_azimuthal_angle,
                                                                             self.dest_inclination_angle,
                                                                             self.destination_object_orbit_radius,
                                                                             target_speed,
                                                                             destination_planet_eph)
        self.time_step = 0
        state = dict(position=spacecraft_position, velocity=spacecraft_velocity,
                     mass=spacecraft_mass,
                     delta_position=spacecraft_position - target_position,
                     delta_velocity=spacecraft_velocity - target_velocity,
                     time_step=self.time_step)

        self.spacecraft_mass = state["mass"]
        self.spacecraft_position = state["position"]
        self.spacecraft_velocity = state["velocity"]
        self.delta_position = state["delta_position"]
        self.delta_velocity = state["delta_velocity"]
        self.time_step = state["time_step"]

        self.target_position = target_position
        self.target_velocity = target_velocity
        return state, None

    def _get_orbital_speed(self, radius, mu):
        speed = np.sqrt(2 * mu) / radius
        return speed

    def _get_eph_from_orbital_angles(self, theta, phi, radius, speed, moon_eph):
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
        # todo: add threshold for position and velocity
        position_threshold = 0
        velocity_threshold = 0

        # terminal state
        terminated = False
        truncated = False
        info = None

        if np.linalg.norm(self.delta_position) <= position_threshold \
                and np.linalg.norm(self.delta_velocity) <= velocity_threshold:
            terminated = True

        # truncated state time limit, out of bounds, out of fuel todo: add out of bounds values
        if self.time_step > self.max_time_steps:
            truncated = True

        time_delta = self.time_step_duration * 24 * 3600  # in seconds
        num_steps = self.integration_steps

        # todo: verify time array
        time_array = np.arange(0, time_delta, num_steps)
        # implement new state of mass
        detailed_spacecraft_state = odeint(self.accelerate,
                                           y0=np.concatenate([self.spacecraft_position, self.spacecraft_velocity],
                                                             axis=0),
                                           t=time_array,
                                           # todo: verify this function and it's working
                                           args=(action, (self.payload_mass + self.fuel_mass), self.source_planet,
                                                 self.destination_planet,
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

        reward, reward_components = self._get_reward()

        state = dict(position=self.spacecraft_position,
                     velocity=self.spacecraft_velocity,
                     mass=self.spacecraft_mass,
                     delta_position=self.delta_position,
                     delta_velocity=self.delta_velocity,
                     time_step=self.time_step)

        return state, reward, terminated, truncated, info

        # return reward, reward_components, detailed_spacecraft_state

    def _get_reward(self):
        """
        Everything is in SI units
        """
        # static destination based on the end epoch
        dest_position = self.target_position
        position_error = (self.spacecraft_position - dest_position) / pk.AU  # astronomical units
        positional_error_magnitude = np.linalg.norm(position_error)
        positional_reward = - positional_error_magnitude

        mass_reward = -(1 - (self.fuel_mass / self.env_config["fuel_mass"]))

        velocity_error = np.linalg.norm(self.spacecraft_velocity - self.target_velocity) / pk.AU  # astronomical units
        reward = 100 + positional_reward + mass_reward + velocity_error

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

    def render(self):
        return None

    def _mass_ejected(self, thrust, time):
        g_0 = 9.8
        thrust_mag = np.linalg.norm(thrust)
        mass_derivative = thrust_mag / (g_0 * self.specific_impulse)
        return mass_derivative * time

    def accelerate(self, state, time, thrust, spacecraft_mass, source, dest, epoch):
        position = state[0:3]
        velocity = state[3:]

        r_vector_sun = -position
        r_mag_sun = np.linalg.norm(r_vector_sun)

        r_vector_moon = np.array(source.eph(epoch))[0] - position
        r_mag_moon = np.linalg.norm(r_vector_moon)

        r_vector_earth = np.array(dest.eph(epoch))[0] - position
        r_mag_earth = np.linalg.norm(r_vector_earth)

        acceleration = (velocity,
                        thrust / spacecraft_mass +
                        self.MU_SUN / np.power(r_mag_sun, 3) * r_vector_sun +
                        self.MU_EARTH / np.power(r_mag_earth, 3) * r_vector_earth +
                        self.MU_MOON / np.power(r_mag_moon, 3) * r_vector_moon)
        return np.concatenate(acceleration)
