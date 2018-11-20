
"""

This script download data from IRIS-DMC for Cascadia, cleans up the traces, 
and save the data into ASDF data format.

author: Marine Denolle (mdenolle@fas.harvard.edu) - 11/16/18
"""

## download modules
import obspy
from obspy import *
from datetime import datetime
import os, glob
from os import mkdir
from obspy.io import sac

from obspy.geodetics import locations2degrees
from obspy.geodetics import gps2dist_azimuth

from obspy.io.xseed import Parser
from obspy.clients.fdsn import Client
import noise_module
import pyasdf
import numpy as np


## download parameters
client = Client('IRIS')                     # client
NewFreq = 100                               # resampling at X samples per seconds
direc="./ASDF"   # storage folder  
pre_filt = [0.0005, 0.001, 40,50]           # some broadband filtering 
year = 2013                                 # year of data
lamin,lomin,lamax,lomax=42,-129,50,-121     # regional box: min lat, min lon, max lat, max lon
chan='HH*'                                  # channel to download 
net="*"                                     # network to download
sta="*"                                     # station to download
remove_response=True                    # boolean to remove instrumental response



# initialize
starttime=obspy.UTCDateTime(year,1,1)       
endtime=obspy.UTCDateTime(year+1,1,1)
data_type="continuous"              
inv = client.get_stations(network=net, station=sta, channel=chan, location='*', \
    starttime = starttime, endtime=endtime,minlatitude=lamin, maxlatitude=lamax, \
    minlongitude=lomin, maxlongitude=lomax,level="response")
print(inv)
# loop through networks
for K in inv:
    dir1 = direc + "/" + str(K.code)
    # loop through stations
    for sta in K:
        net_sta = K.code + "." + sta.code
        # loop through channels
        for chan in sta:
            f1 = direc + "/" + str(K.code) + "."  + str(sta.code) + "." + str(chan.code) \
                 + "." + str(year) + ".h5"     # filename
            if os.path.isfile(f1):
                pass
            
            # station inventory
            sta_inv = inv.select(station=sta.code,channel=chan.code, starttime=starttime, endtime=endtime)
            # make an ASDF file with one year worth of data there, cut into days
            ds = pyasdf.ASDFDataSet(f1,compression="gzip-3")
            ds.add_stationxml(sta_inv)

            # loop through months
            for im in range(1,12):
                # loop through days
                for iday in range(1,31):
                    try:
                        t1=obspy.UTCDateTime(datetime(year,im,iday))
                    except Exception as e:
                        pass
                    try:
                        t2=obspy.UTCDateTime(datetime(year,im,iday+1))
                    except Exception as e:
                        t2=obspy.UTCDateTime(datetime(year,im+1,1))
                        
                    # sanity checks
                    print(K.code + "." + sta.code + "." + chan.code)
                    print(t1)
                    print(t2)
                    try:
                        # get data
                        tr = client.get_waveforms(network=K.code, station=sta.code, channel=chan.code, location='*', \
                            starttime = t1, endtime=t2,attach_response=True)
                        
                        # clean up data
                        tr = noise_module.process_raw(tr, NewFreq,resp=remove_response,inv=sta_inv)

                        # add data to H5 file
                        tr[0].data = tr[0].data.astype(np.float32)
                        print(sta_inv)
                        ds.add_waveforms(tr,tag="raw_recordings")
                        print(ds) # sanity check
                    except Exception as e:
                        print(e)
                        pass