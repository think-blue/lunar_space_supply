import numpy as np
from pykep import MU_SUN, AU, MU_EARTH
from scipy.integrate import odeint

MU_MOON = 4.9048695e12


def accelerate(state, time, thrust, spacecraft_mass, source, dest, epoch):
    """
    :param state: position and velocity of spacecaft
    :param time: time_range
    :param thrust: thrus in x, y, and z
    :param spacecraft_mass: mass of spacecraft
    :return:
    """
    position = state[0:3]
    velocity = state[3:]

    r_vector_sun = -position
    r_mag_sun = np.linalg.norm(r_vector_sun)

    r_vector_moon = np.array(source.eph(epoch))[0] - position
    r_mag_moon = np.linalg.norm(r_vector_moon)

    r_vector_earth = np.array(dest.eph(epoch))[0] - position
    r_mag_earth = np.linalg.norm(r_vector_earth)

    acceleration = (velocity,
                    thrust / spacecraft_mass + \
                    MU_SUN / np.power(r_mag_sun, 3) * r_vector_sun + \
                    MU_EARTH / np.power(r_mag_earth, 3) * r_vector_earth + \
                    MU_MOON / np.power(r_mag_moon, 3) * r_vector_moon)
    return np.concatenate(acceleration)


class Spacecraft:
    def __init__(self, payload, fuel_mass, source, destination, specific_impulse, epoch, reward_coefficients=None):
        """

        :param payload: mass in KGs (without fuel)
        :param fuel_mass: mass in KG (fuel mass)
        :param source: jpl planet
        :param destination: jpl planet
        :param specific_impulse: Isp in seconds
        :param epoch: pk.epoch
        :param reward_coefficients:
        """
        initial_position = np.array(source.eph(epoch))[0] + np.array([25000e3, 0, 0])
        initial_velocity = source.eph(epoch)[1]
        self.source = source
        self.velocity = initial_velocity
        self.position = initial_position
        self.payload = payload
        self.fuel_mass = fuel_mass
        self.specific_impulse = specific_impulse
        self.epoch = epoch
        self.destination = destination

        self.fuel_mass_history = [fuel_mass]
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.epoch_history = [epoch]

        if reward_coefficients is None:
            self.__reward_coefficients = (0.5, -500, -0.05, -250, 0.25)
        else:
            self.__reward_coefficients = reward_coefficients

    @property
    def state(self):
        destination_position = np.array(self.destination.eph(self.epoch))[0] - np.array([60000e3])
        return np.concatenate((
            self.position,
            self.velocity,
            np.array([self.payload + self.fuel_mass]),
            self.position - destination_position,
            np.array([len(self.epoch_history) - 1])
        ))

    def __update_state(self, fuel_mass, position, velocity, epoch):
        self.fuel_mass_history.append(fuel_mass)
        self.position_history.append(position)
        self.velocity_history.append(velocity)
        self.epoch_history.append(epoch)

        self.position = position
        self.velocity = velocity
        self.fuel_mass = fuel_mass
        self.epoch = epoch

    def reset_state(self, payload, fuel_mass, source, destination, specific_impulse, epoch):
        initial_position = np.array(source.eph(epoch))[0] + np.array([25000e3, 0, 0])
        initial_velocity = source.eph(epoch)[1]
        self.payload = payload
        self.fuel_mass = fuel_mass
        self.position = initial_position
        self.velocity = initial_velocity
        self.epoch = epoch
        self.destination = destination
        self.specific_impulse = specific_impulse

        self.fuel_mass_history = [fuel_mass]
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.epoch_history = [epoch]

    def __reward(self, target_epoch=None):
        # currently based on reward from positional errors

        if target_epoch is None:
            dest_position = np.array(self.destination.eph(self.epoch))[0] + np.array([25000e3, 0, 0])
        else:
            dest_position = np.array(self.destination.eph(target_epoch)) + np.array([25000e3, 0, 0])
        position_error = (self.position - dest_position) / AU  # In astronomical units
        c1, c2, c3, c4, c5 = self.__reward_coefficients
        positional_error_magnitude = np.linalg.norm(position_error)
        positional_mag = np.linalg.norm(self.position) / AU

        positional_reward = - positional_error_magnitude * 10
        third_reward = c1 * np.exp(c3)
        mass_reward = -c5 * (1 - (self.fuel_mass / self.fuel_mass_history[0]))

        reward = 100 + positional_reward + third_reward + mass_reward

        return reward, [positional_reward, third_reward, mass_reward]
        # debug inputs: 4th term coming out to be zero? 3rd term is constant 0.4756

    def __mass_ejected(self, thrust, time):
        G_0 = 9.8
        thrust_mag = np.linalg.norm(thrust)
        dmass_dt = thrust_mag / (G_0 * self.specific_impulse)
        return dmass_dt * time

    def accelerate(self, thrust, next_epoch, num_steps=100, target_epoch=None):
        time_delta = (next_epoch.mjd2000 - self.epoch.mjd2000) * 24 * 3600  # in seconds
        time_array = np.arange(0, time_delta, num_steps)
        # implement new state of mass
        detailed_spacecraft_state = odeint(accelerate,
                                           y0=np.concatenate([self.position, self.velocity], axis=0),
                                           t=time_array,
                                           args=(thrust, (self.payload + self.fuel_mass), self.source, self.destination,
                                                 self.epoch))
        spacecraft_pos = np.array(detailed_spacecraft_state[-1, :3])
        spacecraft_vel = np.array(detailed_spacecraft_state[-1, 3:])
        mass_ejected = self.__mass_ejected(thrust, len(time_array))

        self.__update_state(
            fuel_mass=self.fuel_mass - mass_ejected,
            position=spacecraft_pos,
            velocity=spacecraft_vel,
            epoch=next_epoch
        )
        reward, reward_components = self.__reward(target_epoch=target_epoch)
        return reward, reward_components, detailed_spacecraft_state
