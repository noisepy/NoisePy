import os
import glob
import obspy
import pyasdf 
import numpy as np 

'''
this script helps remove the redudent channels for stations of more than 3 channels for a certain time 
period that is downloaded and stored in ASDF files
'''

# data dir
DATADIR   = '/Users/chengxin/Documents/NoisePy_example/SCAL/RAW_DATA'

# get all ASDF file containing noise data
sfiles = glob.glob(os.path.join(DATADIR,'*.h5'))
chan2keep = 'bh'
ncomp     = 3

# loop through all data
for sfile in sfiles:
    with pyasdf.ASDFDataSet(sfile) as ds:
        alist = ds.waveforms.list()

        # loop through each source
        for ilist in alist:
            chan_list = ds.waveforms[ilist].get_waveform_tags()
            tcomp = len(chan_list)

            # if more than needed
            if tcomp > ncomp:
                # loop through all channels
                for ichan in chan_list:
                    if ichan[:2] == chan2keep:
                        continue
                    else:
                        del ds.waveforms[ilist][ichan]