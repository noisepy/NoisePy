import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import scipy
from obspy.signal.invsim import cosine_sac_taper, evalresp, invert_spectrum
from obspy.signal.util import _npts2nfft

# -----directory to station list and response files--------
# resp_dir = '/Users/chengxin/Documents/Harvard/Kanto_basin/instrument/resp_all'
# locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations.txt'
resp_dir = (
    "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/instrument/resp_4types"
)
locations = "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/instrument/resp_4types/station.lst"

# -----common variables for extracting resp using evalresp function------
water_level = 60
prefilt = [0.04, 0.05, 2, 3]
downsamp_freq = 20
dt = 1 / downsamp_freq
cc_len = 3600
step = 1800
npts = cc_len * downsamp_freq * 24
Nfft = _npts2nfft(npts)
tdate = obspy.UTCDateTime("2011-11-3T16:30:00.000")

# -----read the station list------
locs = pd.read_csv(locations)
nsta = len(locs)

for ii in range(nsta):
    station = locs.iloc[ii]["station"]
    network = locs.iloc[ii]["network"]
    print("work on station " + station)

    tfiles = glob.glob(os.path.join(resp_dir, "RESP." + station + "*"))
    if len(tfiles) == 0:
        print("cannot find resp files for station " + station)

    tfile = tfiles[0]
    comp = tfile.split(".")[2]

    # ---extract the resp------
    respz, freq = evalresp(
        dt,
        Nfft,
        tfile,
        tdate,
        station=station,
        channel=comp,
        network=network,
        locid="*",
        units="VEL",
        freq=True,
        debug=False,
    )
    plt.subplot(211)
    plt.loglog(freq, np.absolute(respz))
    invert_spectrum(respz, water_level)
    plt.subplot(212)
    plt.loglog(freq, np.absolute(respz))
    # cos_win = cosine_sac_taper(freq, flimit=prefilt)
    # respz *=cos_win
    plt.show()

    output = os.path.join(resp_dir, "resp." + station + ".npy")
    np.save(output, [np.float32(freq), np.complex64(respz)])
