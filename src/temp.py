import sys
import time
import obspy
import pyasdf
import os, glob
import numpy as np
import pandas as pd
import noise_module
from mpi4py import MPI
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# download parameters
client    = Client('IRIS')                                      # client/data center. see https://docs.obspy.org/packages/obspy.clients.fdsn.html for a list
samp_freq = 1                                                   # targeted sampling rate at X samples per seconds 
rm_resp   = 'no'                                                # select 'no' to not remove response and use 'inv','spectrum','RESP', or 'polozeros' to remove response
respdir   = './'                                                # directory where resp files are located (required if rm_resp is neither 'no' nor 'inv')
freqmin   = 0.05                                                # pre filtering frequency bandwidth
freqmax   = 2                                                   # note this cannot exceed Nquist freq                         

chan = ['BHZ','BHZ']                                            # channel if down_list=false (format like "HN?" not work here)
net  = ['TA','TA']                                              # network list 
sta  = ['K62A','K63A']                                          # station (using a station list is way either compared to specifying stations one by one)
start_date = ["2014_01_01_0_0_0"]                               # start date of download
end_date   = ["2014_01_02_0_0_0"]                               # end date of download
inc_hours  = 24                                                 # length of data for each request (in hour)
nsta       = len(sta)

# save prepro parameters into a dic
prepro_para = {'rm_resp':rm_resp,'respdir':respdir,'freqmin':freqmin,'freqmax':freqmax,'samp_freq':samp_freq,'start_date':\
    start_date,'end_date':end_date,'inc_hours':inc_hours}

# convert time info to UTC format
starttime = obspy.UTCDateTime(start_date[0])       
endtime   = obspy.UTCDateTime(end_date[0])

# another format of time info needed for get_station and get_waveform
s1,s2 = noise_module.get_event_list(start_date[0],end_date[0],inc_hours)
date_info = {'starttime':starttime,'endtime':endtime} 

# write into ASDF file
ff=os.path.join('./',s1+'T'+s2+'.h5')
with pyasdf.ASDFDataSet(ff,mpi=False,compression="gzip-3",mode='w') as ds:

    # loop through each station
    for ista in range(nsta):

        # get inventory for each station
        try:
            sta_inv = client.get_stations(network=net[ista],station=sta[ista],\
                location='*',starttime=s1,endtime=s2,level="response")
        except Exception as e:
            print(e);continue

        # add the inventory into ASDF        
        try:
            ds.add_stationxml(sta_inv) 
        except Exception: 
            pass   

        try:
            # get data
            tr = client.get_waveforms(network=net[ista],station=sta[ista],\
                channel=chan[ista],location='*',starttime=s1,endtime=s2)
        except Exception as e:
            print(e,'for',sta[ista]);continue
            
        # preprocess to clean data  
        print('working on station '+sta[ista])
        tr = noise_module.preprocess_raw(tr,sta_inv,prepro_para,date_info)

        if len(tr):
            new_tags = '{0:s}_00'.format(chan[ista].lower())
            ds.add_waveforms(tr,tag=new_tags)
