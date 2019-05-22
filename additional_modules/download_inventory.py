import obspy
import pyasdf
import os, glob
import noise_module
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

'''
Download the station inventory only (this is useful when you have station
that the instrument has been changed during the time)
'''

#------absolute path for the data-----
rootpath  = '/Users/chengxin/Documents/Harvard/Seattle'
invdir    = os.path.join(rootpath,'resp_all')
slocation = os.path.join(rootpath,'*h5')
afiles    = glob.glob(slocation)

#----download data----
client = Client('IRIS')
if not os.path.isdir(invdir):
    os.mkdir(invdir)

#---loop through all files---
for ii in range(len(afiles)):
    tfile = afiles[ii]
    net   = tfile.split('/')[-1].split('.')[0]
    sta   = tfile.split('/')[-1].split('.')[1]

    #-----outfile names-----
    outfile = os.path.join(invdir,net+'.'+sta+'.XML')

    with pyasdf.ASDFDataSet(afiles[ii],mode='r') as ds:
        slist = ds.waveforms.list()
        if slist:
            rlist = ds.waveforms[slist[0]].get_waveform_tags()
            if rlist:
                temp1 = rlist[0]
                temp2 = rlist[-1]

                t1=obspy.UTCDateTime(int(temp1.split('_')[2]),int(temp1.split('_')[3]),int(temp1.split('_')[4]))
                t2=obspy.UTCDateTime(int(temp2.split('_')[2]),int(temp2.split('_')[3]),int(temp2.split('_')[4]))
                tinv = ds.waveforms[slist[0]]['StationXML']
                chan = tinv[0][0][0].code[0:2]+'*'

                try:
                    inv = client.get_stations(network=net, station=sta, channel=chan, starttime = t1, endtime=t2, level="response")
                    inv.write(outfile,format='StationXML')
                except Exception as error:
                    print(error)
                    pass