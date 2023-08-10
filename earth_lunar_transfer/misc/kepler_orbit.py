from math import pi
import numpy as np
import pykep as pk
import matplotlib.pyplot as plt

r_array = []
for epoch in np.arange(0, 10 * pi, 0.01):
    r, v = pk.par2ic([20000e3, 0.8, pi / 2, pi / 4, pi / 4, epoch], 4.9e12)
    r_array.append(r)

r_array = np.array(r_array)
plt.plot(np.linalg.norm(r_array, axis=-1))
plt.show()
