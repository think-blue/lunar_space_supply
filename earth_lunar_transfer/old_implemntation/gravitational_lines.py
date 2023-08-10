import numpy as np
import pykep as pk
from plotly import graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

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
gravitational_parameter_sun = pk.MU_SUN
gravitational_parameter_earth = pk.MU_EARTH
gravitational_parameter_moon = 4.9048695e12

x_loc = np.arange(-400000, 800000, 1000) * 1000
y_loc = np.arange(-600000, 600000, 1000) * 1000

force_fields = np.empty((len(x_loc), len(y_loc)))
u_vec = np.empty((len(x_loc), len(y_loc)))
v_vec = np.empty((len(x_loc), len(y_loc)))

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
        force_fields[x_index, y_index] = force_field_mag

        if .00285 <= force_field_mag <= .00286:
            u_vec[x_index, y_index] = force_field[0]
            v_vec[x_index, y_index] = force_field[1]
        else:
            u_vec[x_index, y_index] = np.nan
            v_vec[x_index, y_index] = np.nan

force_field_df = pd.DataFrame(data=force_fields)
force_field_df.describe()

# cut-off and normalise
# force_field_df[force_field_df > 0.02] = 0.02
# force_field_df = force_field_df / 0.02

plt.imshow(np.log(force_field_df))
plt.colorbar()
plt.show()
figure = ff.create_quiver(x=x_loc,
                          y=y_loc,
                          u=u_vec,
                          v=v_vec)
figure.add_trace(
    go.Surface(
        # z=np.log(force_field_df),
        z=np.log(force_field_df),
        x=x_loc,
        y=y_loc,
        contours={
            # "x": dict(show=True, start=0, end=100000, size=1000, color="white"),
            "z": dict(show=True, start=-7, end=0, size=0.05, color="white")
        }
    ))

# figure.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                      highlightcolor="limegreen", project_z=True))
# figure.add_trace(go.Scatter3d(y=[earth_loc[0]], x=[earth_loc[1]], z=[0]))
# figure.add_trace(go.Scatter3d(y=[moon_loc[0]], x=[moon_loc[1]], z=[0]))

figure.write_html("gravity_lines.html")
figure.show()
