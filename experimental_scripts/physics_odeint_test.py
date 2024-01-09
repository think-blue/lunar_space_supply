import json
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiDiscrete
import pykep as pk
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

MU_MOON = 4.9048695e12
MU_EARTH = pk.MU_EARTH
MU_SUN = 0  # pk.MU_SUN
EARTH_MOON_MEAN_DISTANCE = 384467e3
MOON_DISTANCE_FROM_BARYCENTER = 384467e3 - 4671e3
MOON_SPEED_WRT_EARTH = 1023.056

with open("../earth_lunar_transfer/configs/env_config_train.json", "rb") as config_file:
    env_config = json.load(config_file)

action_space = Box(-1, 1, (3,), dtype=np.float32)
observation_space = Dict(
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

payload_mass = env_config["payload_mass"]
specific_impulse = env_config["specific_impulse"]

# time variables
num_epochs = env_config["num_days"]  # 5 days
time_step_duration = env_config["time_step_duration"]  # 1/48
max_time_steps = num_epochs / time_step_duration

# orbit's radius
source_object_orbit_radius = env_config["source_object_orbit_radius"]
destination_object_orbit_radius = env_config["dest_object_orbit_radius"]
source_inclination_angle = env_config["source_inclination_angle"]  # phi
source_azimuthal_angle = env_config["source_azimuthal_angle"]  # theta
dest_inclination_angle = env_config["dest_inclination_angle"]  # phi
dest_azimuthal_angle = env_config["dest_azimuthal_angle"]  # theta

# planets
kernel_path = env_config["kernel_path"]
pk.util.load_spice_kernel(kernel_path)

observer = env_config["observer_body"]
source_planet = pk.planet.spice(env_config["source_planet"], observer)
destination_planet = pk.planet.spice(env_config["dest_planet"], observer)
sun = pk.planet.spice("sun", observer)

# integration_variables
integration_steps = env_config["integration_steps"]

fuel_mass = env_config["fuel_mass"]
current_epoch = env_config["start_epoch"]

def get_orbital_speed(radius, mu):
    speed = np.sqrt(2 * mu) / radius
    return speed

def get_orbital_speed_2(radius, mu):
    speed = np.sqrt(mu/radius)
    return speed

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

def mass_ejected(thrust, time):
    g_0 = 9.8
    thrust_mag = np.linalg.norm(thrust)
    mass_derivative = thrust_mag / (g_0 * specific_impulse)
    return mass_derivative * time

def accelerate(state, time, thrust, spacecraft_mass, epoch):
    position = state[0:3]
    velocity = state[3:]

    r_vector_sun = np.array(sun.eph(epoch))[0] - position
    r_mag_sun = np.linalg.norm(r_vector_sun)

    r_vector_moon = np.array(source_planet.eph(epoch))[0] - position
    r_mag_moon = np.linalg.norm(r_vector_moon)

    r_vector_earth = np.array(destination_planet.eph(epoch))[0] - position
    r_mag_earth = np.linalg.norm(r_vector_earth)

    acceleration = (velocity,
                    thrust / spacecraft_mass +
                    MU_SUN / np.power(r_mag_sun, 3) * r_vector_sun +
                    MU_EARTH / np.power(r_mag_earth, 3) * r_vector_earth +
                    MU_MOON / np.power(r_mag_moon, 3) * r_vector_moon)
    return np.concatenate(acceleration)

spacecraft_mass = env_config["payload_mass"] + env_config["fuel_mass"]
source_planet_eph = source_planet.eph(current_epoch)
spacecraft_initial_speed = get_orbital_speed_2(source_object_orbit_radius, MU_MOON)
spacecraft_position, spacecraft_velocity = get_eph_from_orbital_angles(source_azimuthal_angle,
                                                                            source_inclination_angle,
                                                                            source_object_orbit_radius,
                                                                            spacecraft_initial_speed,
                                                                            source_planet_eph)
destination_planet_eph = destination_planet.eph(current_epoch)
target_speed = get_orbital_speed_2(destination_object_orbit_radius, MU_EARTH)
target_position, target_velocity = get_eph_from_orbital_angles(dest_azimuthal_angle,
                                                                    dest_inclination_angle,
                                                                    destination_object_orbit_radius,
                                                                    target_speed,
                                                                    destination_planet_eph)
time_step = 0
state = dict(
    delta_position=spacecraft_position - target_position,
    delta_velocity=spacecraft_velocity - target_velocity,
    mass=np.array([spacecraft_mass]),
    position=spacecraft_position,
    time_step=np.array([time_step]),
    velocity=spacecraft_velocity)

spacecraft_mass = state["mass"].item()
spacecraft_position = state["position"]
spacecraft_velocity = state["velocity"]
delta_position = state["delta_position"]
delta_velocity = state["delta_velocity"]
time_step = state["time_step"].item()

target_position = target_position
target_velocity = target_velocity

n_steps = 2500

time_delta = time_step_duration * 24 * 3600
num_steps = integration_steps

time_array = np.arange(0, time_delta, num_steps)

# action = np.array([0.05, 0.05, 0.05])
action = np.array([0.0, 0.0, 0.0])

print("calculating with time delta")
pos_list = np.array([spacecraft_position])
vel_list = np.array([spacecraft_velocity])
epoch_list = np.array([current_epoch])
for _ in range(n_steps):
    detailed_spacecraft_state = odeint(accelerate,
                                        y0=np.concatenate([spacecraft_position, spacecraft_velocity],
                                                            axis=0),
                                        t=[0, time_delta],
                                        # todo: verify this function and it's working
                                        args=(action, (payload_mass + fuel_mass),
                                                current_epoch))
    # todo: check what odeint.T does
    spacecraft_position = np.array(detailed_spacecraft_state[-1, :3])
    spacecraft_velocity = np.array(detailed_spacecraft_state[-1, 3:])
    pos_list = np.vstack([pos_list, spacecraft_position])
    vel_list = np.vstack([vel_list, spacecraft_velocity])

    # todo: verify mass ejected function
    ejected_mass = mass_ejected(action, len(time_array))

    fuel_mass -= ejected_mass
    current_epoch += time_step_duration
    epoch_list = np.append(epoch_list, current_epoch)


payload_mass = env_config["payload_mass"]
fuel_mass = env_config["fuel_mass"]
current_epoch = env_config["start_epoch"]
spacecraft_mass = env_config["payload_mass"] + env_config["fuel_mass"]
source_planet_eph = source_planet.eph(current_epoch)
spacecraft_initial_speed = get_orbital_speed_2(source_object_orbit_radius, MU_MOON)
spacecraft_position, spacecraft_velocity = get_eph_from_orbital_angles(source_azimuthal_angle,
                                                                            source_inclination_angle,
                                                                            source_object_orbit_radius,
                                                                            spacecraft_initial_speed,
                                                                            source_planet_eph)
print("calculating with time array")
pos_list_2 = np.array([spacecraft_position])
vel_list_2 = np.array([spacecraft_velocity])
epoch_list_2 = np.array([current_epoch])
for _ in range(n_steps):
    detailed_spacecraft_state = odeint(accelerate,
                                        y0=np.concatenate([spacecraft_position, spacecraft_velocity],
                                                            axis=0),
                                        t=time_array,
                                        # todo: verify this function and it's working
                                        args=(action, (payload_mass + fuel_mass),
                                                current_epoch))
    # todo: check what odeint.T does
    spacecraft_position = np.array(detailed_spacecraft_state[-1, :3])
    spacecraft_velocity = np.array(detailed_spacecraft_state[-1, 3:])
    pos_list_2 = np.vstack([pos_list_2, spacecraft_position])
    vel_list_2 = np.vstack([vel_list_2, spacecraft_velocity])

    # todo: verify mass ejected function
    ejected_mass = mass_ejected(action, len(time_array))

    fuel_mass -= ejected_mass
    current_epoch += time_step_duration
    epoch_list_2 = np.append(epoch_list_2, current_epoch)

payload_mass = env_config["payload_mass"]
fuel_mass = env_config["fuel_mass"]
current_epoch = env_config["start_epoch"]
spacecraft_mass = env_config["payload_mass"] + env_config["fuel_mass"]
source_planet_eph = source_planet.eph(current_epoch)
spacecraft_initial_speed = get_orbital_speed(source_object_orbit_radius, MU_MOON)
spacecraft_position, spacecraft_velocity = get_eph_from_orbital_angles(source_azimuthal_angle,
                                                                            source_inclination_angle,
                                                                            source_object_orbit_radius,
                                                                            spacecraft_initial_speed,
                                                                            source_planet_eph)

print("Old orbit vel func calculating with time delta")
pos_list_3 = np.array([spacecraft_position])
vel_list_3 = np.array([spacecraft_velocity])
epoch_list_3 = np.array([current_epoch])
for _ in range(n_steps):
    detailed_spacecraft_state = odeint(accelerate,
                                        y0=np.concatenate([spacecraft_position, spacecraft_velocity],
                                                            axis=0),
                                        t=[0, time_delta],
                                        # todo: verify this function and it's working
                                        args=(action, (payload_mass + fuel_mass),
                                                current_epoch))
    # todo: check what odeint.T does
    spacecraft_position = np.array(detailed_spacecraft_state[-1, :3])
    spacecraft_velocity = np.array(detailed_spacecraft_state[-1, 3:])
    pos_list_3 = np.vstack([pos_list_3, spacecraft_position])
    vel_list_3 = np.vstack([vel_list_3, spacecraft_velocity])

    # todo: verify mass ejected function
    ejected_mass = mass_ejected(action, len(time_array))

    fuel_mass -= ejected_mass
    current_epoch += time_step_duration
    epoch_list_3 = np.append(epoch_list_3, current_epoch)

payload_mass = env_config["payload_mass"]
fuel_mass = env_config["fuel_mass"]
current_epoch = env_config["start_epoch"]
spacecraft_mass = env_config["payload_mass"] + env_config["fuel_mass"]
source_planet_eph = source_planet.eph(current_epoch)
spacecraft_initial_speed = get_orbital_speed(source_object_orbit_radius, MU_MOON)
spacecraft_position, spacecraft_velocity = get_eph_from_orbital_angles(source_azimuthal_angle,
                                                                            source_inclination_angle,
                                                                            source_object_orbit_radius,
                                                                            spacecraft_initial_speed,
                                                                            source_planet_eph)

print("Old orbit vel func calculating with time array")
pos_list_4 = np.array([spacecraft_position])
vel_list_4 = np.array([spacecraft_velocity])
epoch_list_4 = np.array([current_epoch])
for _ in range(n_steps):
    detailed_spacecraft_state = odeint(accelerate,
                                        y0=np.concatenate([spacecraft_position, spacecraft_velocity],
                                                            axis=0),
                                        t=time_array,
                                        # todo: verify this function and it's working
                                        args=(action, (payload_mass + fuel_mass),
                                                current_epoch))
    # todo: check what odeint.T does
    spacecraft_position = np.array(detailed_spacecraft_state[-1, :3])
    spacecraft_velocity = np.array(detailed_spacecraft_state[-1, 3:])
    pos_list_4 = np.vstack([pos_list_4, spacecraft_position])
    vel_list_4 = np.vstack([vel_list_4, spacecraft_velocity])

    # todo: verify mass ejected function
    ejected_mass = mass_ejected(action, len(time_array))

    fuel_mass -= ejected_mass
    current_epoch += time_step_duration
    epoch_list_4 = np.append(epoch_list_4, current_epoch)

def plot_positions():
    plt.subplot(131)
    plt.plot(epoch_list, pos_list[:,0], label="time delta")
    plt.plot(epoch_list_2, pos_list_2[:,0], label="time array")
    # plt.plot(epoch_list, pos_list_3[:,0], label="Old func time delta")
    # plt.plot(epoch_list_2, pos_list_4[:,0], label="Old func time array")
    plt.title("x-axis")
    plt.xlabel("Epoch")
    plt.ylabel("position")
    plt.legend()
    plt.subplot(132)
    plt.plot(epoch_list, pos_list[:,1], label="time delta")
    plt.plot(epoch_list_2, pos_list_2[:,1], label="time array")
    # plt.plot(epoch_list, pos_list_3[:,1], label="Old func time delta")
    # plt.plot(epoch_list_2, pos_list_4[:,1], label="Old func time array")
    plt.title("y-axis")
    plt.xlabel("Epoch")
    plt.ylabel("position")
    plt.legend()
    plt.subplot(133)
    plt.plot(epoch_list, pos_list[:,2], label="time delta")
    plt.plot(epoch_list_2, pos_list_2[:,2], label="time array")
    # plt.plot(epoch_list, pos_list_3[:,2], label="Old func time delta")
    # plt.plot(epoch_list_2, pos_list_4[:,2], label="Old func time array")
    plt.title("z-axis")
    plt.xlabel("Epoch")
    plt.ylabel("position")
    plt.legend()
    plt.show()

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
        data=[go.Scatter3d(x=[source_data[:, 0]],
                            y=[source_data[:, 1]],
                            z=[source_data[:, 2]],
                            name="moon"),
                go.Scatter3d(x=[destination_data[:, 0]],
                            y=[destination_data[:, 1]],
                            z=[destination_data[:, 2]],
                            name="earth"),
                go.Scatter3d(x=[position_data[:, 0]],
                            y=[position_data[:, 1]],
                            z=[position_data[:, 2]],
                            name="spacecraft", marker_size=2)
                ],
        layout=go.Layout(
            scene=dict(xaxis_range=[-figure_len, figure_len],
                        yaxis_range=[-figure_len, figure_len],
                        zaxis_range=[-figure_len, figure_len],
                        xaxis=dict(
                            backgroundcolor='rgb(0, 0, 0)'),
                        yaxis=dict(
                            backgroundcolor='rgb(0, 0, 0)'),
                        zaxis=dict(
                            backgroundcolor='rgb(0, 0, 0)')
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
                                    name="spacecraft", marker_size=2)]
                    ) for i in range(len(source_data))]
    )

    figure.write_html(path)

    if display:
        figure.show()


def plot_3d_line_plot(position_list):
    import plotly.express as px
    
    pos_df = pd.DataFrame(position_list, columns=["X", "Y", "Z"])
    fig = px.line_3d(pos_df, x="X", y="Y", z="Z")
    fig.show()

plot_positions()
# plot_3d_line_plot(pos_list_4)
# simulate(epoch_list_4, pos_list_4, source_planet, destination_planet, "./data/plot.html", True)