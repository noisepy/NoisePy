"""
This script download data from IRIS-DMC for Cascadia, cleans up the traces, 
and save the data into ASDF data format.

author: Marine Denolle (mdenolle@fas.harvard.edu) - 11/16/18

modified by Chengxin Jiang on Feb.18.2019 to make it flexiable for downloading 
data in a range of days instead of a whole year.

add a subfunction to output the station list to a CSV file and indicate the
provenance of the downloaded ASDF files. (Feb.22.2019)
"""

## download modules
import time
import obspy
import pyasdf
import os, glob
import noise_module
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


direc="/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/data_download"

## download parameters
client = Client('IRIS')                         # client
NewFreq = 10                                    # resampling at X samples per seconds 
pre_filt = [0.0005, 0.001, 40,50]               # some broadband filtering                                    # year of data
lamin,lomin,lamax,lomax=42,-122,50,-120         # regional box: min lat, min lon, max lat, max lon
chan='BH*'                                      # channel to download 
net="TA"                                        # network to download
sta="F05D"                                      # station to download
start_date = '2016_05_01'
end_date   = '2016_05_05'
inc_days   = 5                                  # number of days for each request

checkt  = True                                  # check for traces with points bewtween sample intervals
resp    = False                                 # boolean to remove instrumental response
respdir = 'resp_10hz'
pre_filt = [0.04,0.05,4,5]
output_CSV=True                                 # output station.list to a CSV file to be used in later stacking steps
flag = True                                    # print progress when running the script

#---provence of the data in ASDF files--
if resp and checkt:
    tags = 'time_resp'
elif checkt:
    tags = 'time_checked'
elif resp:
    tags = 'resp_removed'
else:
    tags = 'raw-recordings'

#----check whether folder exists------
if not os.path.isdir(direc):
    os.mkdir(direc)

#-------initialize time information------
starttime=obspy.UTCDateTime(int(start_date[:4]),int(start_date[5:7]),int(start_date[8:]))       
endtime=obspy.UTCDateTime(int(end_date[:4]),int(end_date[5:7]),int(end_date[8:]))

#-----in case there are no data here------
try:
    inv = client.get_stations(network=net, station=sta, channel=chan, location='*', \
        starttime = starttime, endtime=endtime,minlatitude=lamin, maxlatitude=lamax, \
        minlongitude=lomin, maxlongitude=lomax,level="response")
except Exception as e:
    print('Abort! '+type(e))
    exit()

if flag:
    print(inv)

if output_CSV:
    noise_module.make_stationlist_CSV(inv,direc)

# loop through networks
for K in inv:
    # loop through stations
    for sta in K:
        net_sta = K.code + "." + sta.code

        # write all channels into one ASDF file
        f1 = direc + "/" + str(K.code) + "."  + str(sta.code) + ".h5"
        
        if os.path.isfile(f1):
            raise IOError('file %s already exists!' % f1)
        
        with pyasdf.ASDFDataSet(f1,compression="gzip-3") as ds:
                            
            # one asdf file only takes one station inventory: only takes lat, lon and elevation information
            sta_inv = inv.select(station=sta.code,channel=sta[0].code, starttime=starttime, endtime=endtime)
            ds.add_stationxml(sta_inv)

            # loop through channels
            for chan in sta:

                #----get a list of all days within the targeted period range----
                all_days = noise_module.get_event_list(start_date,end_date,inc_days)

                #---------loop through the days--------
                for ii in range(len(all_days)-1):
                    day1  = all_days[ii]
                    day2  = all_days[ii+1]
                    year1 = int(day1[:4])
                    year2 = int(day2[:4])
                    mon1  = int(day1[5:7])
                    mon2  = int(day2[5:7])
                    iday1 = int(day1[8:]) 
                    iday2 = int(day2[8:])

                    t1=obspy.UTCDateTime(year1,mon1,iday1)
                    t2=obspy.UTCDateTime(year2,mon2,iday2)
                            
                    # sanity checks
                    if flag:
                        print(K.code + "." + sta.code + "." + chan.code+' at '+str(t1)+'.'+str(t2))
                    
                    try:
                        # get data
                        t0=time.time()
                        tr = client.get_waveforms(network=K.code, station=sta.code, channel=chan.code, location='*', \
                            starttime = t1, endtime=t2, attach_response=True)
                        t1=time.time()

                    except Exception as e:
                        print(e)
                        continue
                        
                    if len(tr):
                        # clean up data
                        t2=time.time()
                        tr = noise_module.preprocess_raw(tr,NewFreq,checkt,pre_filt,resp,respdir)
                        t3=time.time()

                        # only keep the one with good data after processing
                        if len(tr)>0:
                            if len(tr)==1:
                                new_tags = tags+'_{0:04d}_{1:02d}_{2:02d}_{3}'.format(tr[0].stats.starttime.year,\
                                    tr[0].stats.starttime.month,tr[0].stats.starttime.day,chan.code.lower())
                                ds.add_waveforms(tr,tag=new_tags)
                            else:
                                for ii in range(len(tr)):
                                    new_tags = tags+'_{0:04d}_{1:02d}_{2:02d}_{3}'.format(tr[ii].stats.starttime.year,\
                                        tr[ii].stats.starttime.month,tr[ii].stats.starttime.day,chan.code.lower())
                                    ds.add_waveforms(tr[ii],tag=new_tags)

                        if flag:
                            print(ds) # sanity check
                            print('downloading data %6.2f s; pre-process %6.2f s' % ((t1-t0),(t3-t2)))

