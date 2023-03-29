import sys
import time

import pyasdf

sys.path.insert(1, "../../src")
import noise_module

"""
this script loads an example of daily ccfs and compare the computing
time of doing cc with one hour sequence with that with a day. the final
goal is to see how much time it saves when we optimize reading the data
for step 2
"""

comp = "EHZ"
day = "2010_01_10"
maxlag = 800
downsample_rate = 20
dt = 1 / downsample_rate
method = "decon"

sfile = "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT_opt/N.ATDH.h5"
rfile = "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT_opt/N.CHHH.h5"

ds = pyasdf.ASDFDataSet(sfile, mode="r")
dr = pyasdf.ASDFDataSet(rfile, mode="r")

spect_s = ds.auxiliary_data[comp][day].data[:, :]
spect_r = dr.auxiliary_data[comp][day].data[:, :]
Nfft = ds.auxiliary_data[comp][day].parameters["nfft"]
Nseg = ds.auxiliary_data[comp][day].parameters["nseg"]

t0 = time.time()
corr = noise_module.optimized_correlate1(
    spect_s[:, :], spect_r[:, :], int(maxlag), dt, Nfft, Nseg, method
)
t1 = time.time()
corr = noise_module.optimized_correlate1(
    spect_s[1, :], spect_r[1, :], int(maxlag), dt, Nfft, 1, method
)
t2 = time.time()
print(
    "it takes %f for %d segments and %f for one segment" % ((t1 - t0), Nseg, (t2 - t1))
)
temp = (t1 - t0) / Nseg
print("so it is %f times faster for each segment" % ((t2 - t1) / temp))
