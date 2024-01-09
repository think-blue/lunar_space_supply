from math import pi
import numpy as np
import pykep as pk
import matplotlib.pyplot as plt

r_array = []
for epoch in np.arange(0, 10 * pi, 0.01):
    r, v = pk.par2ic(E=[20000e3, 0.2, 0, 0, 0, epoch],
                     mu=4.9e12)
    r_array.append(r)

r_array = np.array(r_array)
plt.plot(np.linalg.norm(r_array, axis=-1))
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_array[:, 0], r_array[:, 1], r_array[:, 2])
ax.scatter(0, 0, 0)
plt.show()

# force = 10 * (r / np.linalg.norm(r))
# pk.propagate_lagrangian(r0=r, v0=v, m0=1000, thrust=force, tof=pi/4, mu=4.9e12)

