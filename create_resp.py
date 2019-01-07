import numpy as np
import scipy
import obspy
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from obspy.signal.invsim import evalresp,invert_spectrum,cosine_sac_taper
from obspy.signal.util import _npts2nfft

#-----directory to station list and response files--------
resp_dir = '/Users/chengxin/Documents/Harvard/Kanto_basin/instrument/resp_all'
locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations.txt'

#-----common variables for extracting resp using evalresp function------
water_level = 60
prefilt = [0.008,0.01,4,6]
downsamp_freq=20
dt=1/downsamp_freq
cc_len=3600
step=1800 
npts=cc_len*downsamp_freq*24
Nfft=_npts2nfft(npts)
tdate = obspy.UTCDateTime("2011-11-3T16:30:00.000")

#-----read the station list------
locs = pd.read_csv(locations)
nsta = len(locs)

for ii in range(nsta):
    station = locs.iloc[ii]['station']
    network = locs.iloc[ii]['network']
    print('work on station '+station)

    tfiles = glob.glob(os.path.join(resp_dir,'RESP.'+station+'*'))
    if len(tfiles)==0:
        print('cannot find resp files for station '+station)

    tfile = tfiles[0]
    comp = tfile.split('.')[2]

    #---extract the resp------
    respz,freq=evalresp(dt,Nfft,tfile,tdate,station=station,channel=comp,network=network,locid='*',units='VEL',freq=True,debug=False)
    #plt.subplot(211)
    #plt.loglog(freq,np.absolute(respz))
    invert_spectrum(respz, water_level)
    #plt.subplot(212)
    #plt.loglog(freq,np.absolute(respz))
    #plt.show()

    output = os.path.join(resp_dir,'resp_spectrum/resp.'+station+'.npy')
    np.save(output,respz)
