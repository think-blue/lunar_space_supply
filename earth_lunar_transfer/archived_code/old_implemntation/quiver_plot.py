import numpy as np
import pykep as pk
from plotly import graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

# moon = pk.planet.jpl_lp("moon")
# earth = pk.planet.jpl_lp("earth")
# sun = pk.planet.jpl_lp("sun")

start_epoch = 0

pykep_epoch = pk.epoch(start_epoch, julian_date_type="mjd2000")

# in SI units (meters)
earth_loc = np.array([0, 0])
sun_loc = np.array([150e9, 0])
moon_loc = np.array([384400e3, 0])

# in SI units
gravitational_parameter_sun = 0
gravitational_parameter_earth = pk.MU_EARTH
gravitational_parameter_moon = 4.9048695e12

x_loc = np.arange(-400000, 800000, 700) * 1000
y_loc = np.arange(-600000, 600000, 700) * 1000

force_fields = np.empty((len(x_loc), len(y_loc)))
u_comp = np.empty((len(x_loc), len(y_loc)))
v_comp = np.empty((len(x_loc), len(y_loc)))

for x_index, x in enumerate(x_loc):
    for y_index, y in enumerate(y_loc):
        coordinate = np.array([x, y])
        r_vector_earth = earth_loc - coordinate
        r_vector_moon = moon_loc - coordinate
        r_vector_sun = sun_loc - coordinate

        force_field = (gravitational_parameter_earth / (np.linalg.norm(r_vector_earth) ** 3)) * r_vector_earth + \
                      (gravitational_parameter_sun / (np.linalg.norm(r_vector_sun) ** 3)) * r_vector_sun + \
                      (gravitational_parameter_moon / (np.linalg.norm(r_vector_moon) ** 3)) * r_vector_moon
        force_field_mag = np.linalg.norm(force_field)
        if 0.002936 <= force_field_mag <= 0.002937:
            print("inside the if condition")
            u_comp[x_index, y_index] = force_field[0]
            v_comp[x_index, y_index] = force_field[1]
            force_fields[x_index, y_index] = force_field_mag
        else:
            u_comp[x_index, y_index] = np.nan
            v_comp[x_index, y_index] = np.nan
            force_fields[x_index, y_index] = 0

# force_field_df = pd.DataFrame(data=force_fields)
# force_field_df.describe()
#
# # cut-off and normalise
# force_field_df[force_field_df > 0.02] = 0.02
# # force_field_df = force_field_df / 0.02
#
# plt.imshow(np.log(force_field_df))
# plt.colorbar()
# plt.show()
#
# figure = go.Figure(
#     data=go.Surface(
#         z=force_field_df,
#         x=y_loc,
#         y=x_loc,
#         contours={
#             # "x": dict(show=True, start=0, end=100000, size=1000, color="white"),
#             "z": dict(show=True, start=-7, end=0, size=0.05, color="white")
#         }
#     ))
#
# # figure.update_traces(contours_z=dict(show=True, usecolormap=True,
# #                                      highlightcolor="limegreen", project_z=True))
# # figure.add_trace(go.Scatter3d(y=[earth_loc[0]], x=[earth_loc[1]], z=[0]))
# # figure.add_trace(go.Scatter3d(y=[moon_loc[0]], x=[moon_loc[1]], z=[0]))
#
# figure.write_html("gravity_lines.html")
# figure.show()

import plotly.figure_factory as ff

x, y = np.meshgrid(np.arange(len(x_loc)), np.arange(len(y_loc)))
step = 2
u_comp = (u_comp - np.nanmin(u_comp)) / np.nanmax(u_comp)
v_comp = (v_comp - np.nanmin(v_comp)) / np.nanmax(v_comp)
plt.figure(figsize=(20, 20))
plt.quiver(y_loc[::step], x_loc[::step], u_comp[::step, ::step], v_comp[::step, ::step], units='xy', scale=10000)
plt.scatter(x=0, y=0, label="earth")
plt.scatter(x=moon_loc[1], y=moon_loc[0], label="moon")
plt.legend()
plt.show()

pass

# fig = ff.create_quiver(x=x_loc,
#                        y=y_loc,
#                        u=u_comp,
#                        v=v_comp,
#                        scale=0.1)
# fig.show()

# plt.imshow(np.log(force_fields))
# plt.colorbar()
# plt.show()
# np.unravel_index()
