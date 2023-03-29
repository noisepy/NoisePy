import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
import obspy

sys.path.insert(1, "../../src")
import noise_module

"""
This script compares several different ways for smoothing a signal,
including convolution(one-side average), running average mean with
numba compiled and a function from obspy
"""

N = 40
a = np.random.rand(
    500,
).astype(np.float32)
b = noise_module.running_abs_mean(a, N)
c = noise_module.moving_ave(a, N)
d = noise_module.moving_ave1(a, N)
e = obspy.signal.util.smooth(a, N)

plt.plot(a, "r")
plt.plot(b, "g")
plt.plot(c, "b")
plt.plot(c, "c")
plt.plot(e, "y")

modes = ["original", "running_abs_mean", "moving_ave", "moving_ave 1", "util smooth"]
plt.legend(modes, loc="upper right")
plt.show()
