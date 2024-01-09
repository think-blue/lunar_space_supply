import pykep as pk

pk.util.load_spice_kernel("../kernels/de441.bsp")

moon = pk.planet.spice("moon")
earth = pk.planet.spice("earth")
sun = pk.planet.spice("sun")

pod_1 = [0, 0, 0]
pod_2 = [0, 0, 0]
pod_3 = [0, 0, 0]
pod_4 = [0, 0, 0]
pod_5 = [0, 0, 0]

time_steps = 200

earth_loc = earth.eph() - earth.eph()
moon_loc = moon.eph() - moon.eph()
