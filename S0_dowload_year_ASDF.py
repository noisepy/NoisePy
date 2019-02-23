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
import obspy
from obspy import UTCDateTime
import os, glob
from obspy.clients.fdsn import Client
import noise_module
import pyasdf
import numpy as np


direc="/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/data_download"

## download parameters
client = Client('IRIS')                         # client
NewFreq = 10                                    # resampling at X samples per seconds 
pre_filt = [0.0005, 0.001, 40,50]               # some broadband filtering                                    # year of data
lamin,lomin,lamax,lomax=42,-122,50,-120         # regional box: min lat, min lon, max lat, max lon
chan='BHZ'                                      # channel to download 
net="TA"                                        # network to download
sta="*"                                         # station to download
start_date = '2012_01_01'
end_date   = '2012_01_10'

pre_processing=True
remove_response=True                            # boolean to remove instrumental response
output_CSV=True                                 # output station.list to a CSV file to be used in later stacking steps
flag = False                                    # print progress when running the script

#---provence of the data in ASDF files--
if remove_response and pre_processing:
    tags = 'preprocessed_responses_removed'
elif remove_response:
    tags = 'responses_removed'
elif pre_processing:
    tags = 'preprocessed'
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
        # loop through channels
        for chan in sta:
            f1 = direc + "/" + str(K.code) + "."  + str(sta.code) + "." + str(chan.code) \
                 + "." + start_date[:4] + ".h5"     # filename
            if os.path.isfile(f1):
                pass
            
            # station inventory
            sta_inv = inv.select(station=sta.code,channel=chan.code, starttime=starttime, endtime=endtime)
            # make an ASDF file with one year worth of data there, cut into days
            ds = pyasdf.ASDFDataSet(f1,compression="gzip-3")
            ds.add_stationxml(sta_inv)

            #----get a list of all days within the targeted period range----
            all_days = noise_module.get_event_list(start_date,end_date)

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
                    tr = client.get_waveforms(network=K.code, station=sta.code, channel=chan.code, location='*', \
                        starttime = t1-180, endtime=t2+180, attach_response=True)
                    # comment from Marine: request data with extra time series to cut edges effects during pre processing
                except Exception as e:
                    print(e)
                    continue
                    
                # clean up data
                if pre_processing:
                    tr = noise_module.process_raw(tr, NewFreq)

                # add data to H5 file
                tr[0].data = tr[0].data.astype(np.float32)
                ds.add_waveforms(tr,tag=tags)

                if flag:
                    print(ds) # sanity check

