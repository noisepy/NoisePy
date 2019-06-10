
## download modules
import time
import obspy
import pyasdf
import os, glob
import numpy as np
import noise_module
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

'''
This script:
    1) downloads data your choice of Client or pre-compiled station list, 
    2) cleans up the traces (options: gaps, removing instrumental response,
                            downsampling)
    3) saves the data into ASDF data format.

authors: Marine Denolle (mdenolle@fas.harvard.edu) - 11/16/18,06/08/19
         Chengxin Jiang (chengxin_jiang@fas.harvard.edu) - 02/22/19
         
Note: segmentation fault while manipulating obspy stream can come from too large memory: 
- reduce the inc_day variable

A beginning of nice NoisePy journey! 
'''

#----paths and filenames-------------
direc  = "./data_download"                      # where to put the data
dlist  = os.path.join(direc,'station.lst')      # CSV file for station location info

#----check whether folder exists------
if not os.path.isdir(direc):
    os.mkdir(direc)

## download parameters
client = Client('GEONET')                         # client / data center
NewFreq = 10                                    # resampling at X samples per seconds 
down_list = False                               # download stations from pre-compiled list
checkt   = True                                 # check for traces with points bewtween sample intervals
resp     = False                                # boolean to remove instrumental response
resp_dir = 'none'                               # output response directory
pre_filt = [0.04,0.05,4,5]                      # pre filtering frequency bandwidth
oput_CSV = True                                 # output station.list to a CSV file to be used in later stacking steps
flag     = True                                 # print progress when running the script
inc_days   = 2                                  # number of days for each request
lamin,lomin,lamax,lomax=-46.5,168,-38,175#46.9,-123,48.8,-121.1   # regional box: min lat, min lon, max lat, max lon
chan= ['HH*']                                     # channel if down_list=false
net = ["NZ"]                                      # network if 
sta = ["MQZ"]                                   # station (do either one station or *)
start_date = ["2018_05_01"]                       # start date of download
end_date   = ["2018_05_05"]                       # end date of download

 
 

 # --------------- CODE STARTS -------------------------
starttime = obspy.UTCDateTime(int(start_date[0][:4]),int(start_date[0][5:7]),int(start_date[0][8:]))       
endtime   = obspy.UTCDateTime(int(end_date[0][:4]),int(end_date[0][5:7]),int(end_date[0][8:]))
all_days = noise_module.get_event_list(start_date[0],end_date[0],inc_days)

#---processing information: provenance for the ASDF files--
tags=""
if resp and checkt:
    tags = 'time_resp'
elif checkt:
    tags = tags + '_time_checked'
elif resp:
    tags = tags + '_resp_removed'
else:
    tags = tags + '_raw-recordings'

if down_list:
    if not os.path.isfile(dlist):
        raise IOError('file %s not exist! double check!' % dlist)

    #----read station info------
    locs = pd.read_csv(dlist)
    nsta = len(locs)
    chan=np.chararray(nsta,itemsize=3,unicode=True)
    net=np.chararray(nsta,itemsize=2,unicode=True)
    sta=np.chararray(nsta,itemsize=5,unicode=True)
    location=np.chararray(nsta,itemsize=2,unicode=True)
    lat=np.array(nsta,dtype=np.float);lon=np.array(nsta,dtype=np.float)
    start_date=np.chararray(nsta,itemsize=30,unicode=True)
    end_date=np.chararray(nsta,itemsize=30,unicode=True)
    #----loop through each station----
    for ii in range(nsta):
        chan[ii] = locs.iloc[ii]['channel']
        net[ii]  = locs.iloc[ii]['network']
        sta[ii]  = locs.iloc[ii]['station']
        location[ii] = "__"
        lat[ii]  = locs.iloc[ii]['latitude']
        lon[ii]  = locs.iloc[ii]['longitude']
        start_date[ii]= locs.iloc[ii]['start_date']
        end_date[ii]   = locs.iloc[ii]['end_date']

        #----the region to ensure station is unique-----
        latmin = lat-0.2
        latmax = lat+0.2
        lonmin = lon-0.2
        lonmax = lon+0.2
else:
    try:
        inv = client.get_stations(network=net[0], station=sta[0], channel=chan[0], location='*', \
            starttime = starttime, endtime=endtime,minlatitude=lamin, maxlatitude=lamax, \
            minlongitude=lomin, maxlongitude=lomax,level="response")
        if flag:
            print(inv)
    except Exception as e:
        print('Abort! '+str(e))
        exit()

    # calculate the total number of channels to download
    nsta=0
    for K in inv:
        for sta1 in K:
            for chan1 in sta1:
                nsta+=1
    # declare arrays of metadata
    chan=np.chararray(nsta,itemsize=3,unicode=True)
    net=np.chararray(nsta,itemsize=2,unicode=True)
    sta=np.chararray(nsta,itemsize=5,unicode=True)
    location=np.chararray(nsta,itemsize=2,unicode=True)
    lat=np.array(nsta,dtype=np.float);lon=np.array(nsta,dtype=np.float)
    start_date=np.chararray(nsta,itemsize=30,unicode=True)
    end_date=np.chararray(nsta,itemsize=30,unicode=True)
    ii=0
    for K in inv:
        for sta1 in K:
            for chan1 in sta1:
                chan[ii]=chan1.code
                sta[ii]=sta1.code
                net[ii]=K.code
                location[ii]=chan1.location_code
                start_date[ii]=str(starttime)
                end_date[ii]=str(endtime)
                ii+=1
    if oput_CSV:
        noise_module.make_stationlist_CSV(inv,direc)

# labels for stations
point=np.chararray(nsta,itemsize=1,unicode=True);point='.'
net_sta=net + point + sta + point + location + point + chan
print(net_sta)

# loop through each channel
for i in range(nsta):

    # filename of the ASDF file
    f1 = direc + "/" + net[i] + "." + sta[i] + "." + location[i] + ".h5"
    if os.path.isfile(f1):
        print('file %s already exists!' % f1)
    # get the specific inventory
    if len(start_date)>2:
        s1=start_date[i]
        s2=end_date[i]
    else:
        s1=starttime
        s2=endtime
    # get channel inventory
    sta_inv = client.get_stations(network=net[i],station=sta[i],\
        channel=chan[i],starttime = s1, endtime=s2,\
            location=location[i],level="response")

    # open the ASDF file
    with pyasdf.ASDFDataSet(f1,compression="gzip-3") as ds:
      #------add the inventory for all components + all time of this tation-------                   
        if (not ds.waveforms.list()) :
            ds.add_stationxml(sta_inv)
        # elif not (ds.waveforms[net_sta[i]].list()):
            # ds.add_stationxml(sta_inv)


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
            tr=[]              
            try:
                # get data
                t0=time.time()
                tr = client.get_waveforms(network=net[i], station=sta[i], \
                    channel=chan[i], location=location[i], \
                    starttime = t1, endtime=t2)
                t1=time.time()

            except Exception as e:
                print(e)
                continue
                
            if len(tr):
                # clean up data
                t2=time.time()
                tr = noise_module.preprocess_raw(tr,sta_inv,NewFreq,checkt,pre_filt,resp,resp_dir)
                t3=time.time()

                # only keep the one with good data after processing
                if len(tr)>0:
                    if len(tr)==1:
                        new_tags = tags+'_{0:04d}_{1:02d}_{2:02d}_{3}'.format(tr[0].stats.starttime.year,\
                            tr[0].stats.starttime.month,tr[0].stats.starttime.day,chan[i].lower())
                        print(new_tags)
                        ds.add_waveforms(tr,tag=new_tags)
                    else:
                        for ii in range(len(tr)):
                            new_tags = tags+'_{0:04d}_{1:02d}_{2:02d}_{3}'.format(tr[ii].stats.starttime.year,\
                                tr[ii].stats.starttime.month,tr[ii].stats.starttime.day,chan[i].lower())
                            ds.add_waveforms(tr[ii],tag=new_tags)

                if flag:
                    print(ds) # sanity check
                    print('downloading data %6.2f s; pre-process %6.2f s' % ((t1-t0),(t3-t2)))
