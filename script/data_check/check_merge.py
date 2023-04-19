import sys

import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client

from noisepy.seis import noise_module

sys.path.insert(1, "../../src")

# download the data
time1 = "2016-07-13T00:00:00.000000Z"
time2 = "2016-07-15T00:00:00.000000Z"

client = Client("IRIS")
tr = client.get_waveforms(
    network="XD",
    station="MD12",
    channel="BHZ",
    location="*",
    starttime=time1,
    endtime=time2,
    attach_response=True,
)

# source = obspy.read('/Users/chengxin/Documents/Harvard/JAKARTA/JKA20miniSEED/JKA20131010073600.CHZ')
# nst = noise_module.preprocess_raw(source,20,True)

# process the data
ntr = tr.copy()
mtr = noise_module.clean_daily_segments(ntr)

# plot the data
plt.subplot(211)
plt.plot(mtr[0].data, "r-")
indx = len(mtr[0].data)
plt.subplot(212)
plt.plot(ntr[0].data[:indx], "b-")
plt.show()
