import numpy as np
import pykep as pk
from plotly import graph_objects as go

# moon = pk.planet.jpl_lp("moon")
# earth = pk.planet.jpl_lp("earth")
# sun = pk.planet.jpl_lp("sun")

start_epoch = 0

pykep_epoch = pk.epoch(start_epoch, julian_date_type="mjd2000")

earth_loc = np.array([0, 0, 0])
sun_loc = np.array([150e6, 0, 0])
moon_loc = np.array([384400e3, 0, 0])

gravitational_parameter_sun = pk.MU_SUN
gravitational_parameter_earth = pk.MU_EARTH
gravitational_parameter_moon = 4.9048695e12

x_loc = np.arange(1000, 350000, 1000) * 1000
y_loc = np.arange(-100000, 100000, 1000) * 1000
z_loc = np.arange(-100000, 100000, 1000) * 1000

force_fields = np.empty((len(x_loc), len(y_loc), len(z_loc)))

for x_index, x in enumerate(x_loc):
    for y_index, y in enumerate(y_loc):
        for z_index, z in enumerate(z_loc):
            coordinate = np.array([x, y, z])
            r_vector_earth = earth_loc - coordinate
            r_vector_moon = moon_loc - coordinate
            r_vector_sun = sun_loc - coordinate

            force_field = (gravitational_parameter_earth / (np.linalg.norm(r_vector_earth) ** 3)) * r_vector_earth + \
                          (gravitational_parameter_sun / (np.linalg.norm(r_vector_sun) ** 3)) * r_vector_sun + \
                          (gravitational_parameter_moon / (np.linalg.norm(r_vector_moon) ** 3)) * r_vector_moon
            force_field_mag = np.linalg.norm(force_field)
            force_fields[x_index, y_index] = force_field_mag

import matplotlib.pyplot as plt

plt.imshow(force_fields)
plt.colorbar()
plt.show()

figure = go.Figure(data=go.Surface(
    z=force_fields,
    x=x_loc,
    y=y_loc))
figure.show()
