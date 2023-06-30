import pykep as pk
from matplotlib import pyplot as plt
from plotly import io as pio, graph_objects as go

from earth_lunar_transfer.earth_lunar_old_implemntation.common_module import create_planets, create_position_data


def simulate_spacecraft(states, spacecraft):
    pio.renderers.default = "browser"

    # Constants
    sun_radius = 696340000
    earth_radius = pk.planet.jpl_lp('earth').radius

    # generating data
    pk.util.load_spice_kernel("../../kernels/de441.bsp")
    earth = pk.planet.spice("earth")
    moon = pk.planet.spice('moon')
    planet_list = ['moon', 'earth']
    planets = create_planets(planet_names=planet_list)
    epochs = spacecraft.epoch_history[:-1]
    position_data, velocity_data = create_position_data(planets, epochs, states)
    scaling_factor = earth_radius / 2  # pk.AU
    position_data_au = position_data / scaling_factor

    # visualisation
    planets_vis = [go.Scatter3d(x=[position_data_au[0, planet_num, 0]],
                                y=[position_data_au[0, planet_num, 1]],
                                z=[position_data_au[0, planet_num, 2]],
                                mode="markers")
                   for planet_num in range(position_data_au.shape[1])]

    for planet_num in range(position_data_au.shape[1]):
        planets_vis.append(go.Scatter3d(x=position_data_au[:, planet_num, 0],
                                        y=position_data_au[:, planet_num, 1],
                                        z=position_data_au[:, planet_num, 2],
                                        mode="lines"))
    planets_vis.append(go.Scatter3d(x=[0], y=[0], z=[0], name="Sun", mode="markers",
                                    marker=dict(size=sun_radius / (5 * scaling_factor),
                                                sizemode="diameter")))
    figure_len = 1000000
    figure = go.Figure(
        data=planets_vis,
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
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None])]
            )
            ]
        ),
        frames=[go.Frame(data=[go.Scatter3d(x=position_data_au[frame:frame + 1, planet_num, 0],
                                            y=position_data_au[frame:frame + 1, planet_num, 1],
                                            z=position_data_au[frame:frame + 1, planet_num, 2],
                                            mode="markers")
                               for planet_num in range(position_data_au.shape[1])])
                for frame in range(position_data_au.shape[0] - 1)]
    )
    figure.write_html("visualisation.htm")


def plot_losses(actor_losses, critic_losses):
    fig, ax = plt.subplots(2, 1, sharex='all')

    ax[0].plot(actor_losses, label="actor_loss")
    ax[0].legend()
    ax[1].plot(critic_losses, label="critic_loss")
    ax[1].legend()
    plt.show()
