import os
import sys
import glob
from datetime import datetime
import numpy as np
import scipy
from scipy.fftpack.helper import next_fast_len
import obspy
import matplotlib.pyplot as plt
import noise_module
import time
import pyasdf
import pandas as pd
from mpi4py import MPI

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
Main script of the NoisePy package.

It pre-processs the noise data of user-defined length using the parameters given below
and get FFT trace for each station in a ASDF format. 
by C.Jiang, M.Denolle, T.Clements
'''

t00=time.time()

########################################
#########PARAMETER SECTION##############
########################################

#------absolute path parameters-------
rootpath  = '/mnt/data0/NZ/XCORR'                        # root path for this data processing
FFTDIR    = os.path.join(rootpath,'FFT/')                # dir to store FFT data
CCFDIR    = os.path.join(rootpath,'CCF')                 # dir to store CC data
event     = os.path.join(rootpath,'noise_data/Event_*')  # dir where noise data is located
if (len(glob.glob(event))==0): 
    raise ValueError('No data file in %s',event)

#-----------other useful input files-----------
locations = os.path.join(rootpath,'locations.txt')       # station list - requred for data in SAC/miniSEED format 
resp_dir  = os.path.join(rootpath,'response')            # only needed when resp set to something other than 'inv'
metadata  = os.path.join(rootpath,'metadata.txt')        # keep a record of used parameters

#-------some control parameters--------
input_fmt   = 'sac'            # string: 'asdf', 'sac','mseed' 
prepro      = False             # preprocess the data (correct time/downsampling/trim data/response removal)?
to_whiten   = False             # False (no whitening), or running-mean, one-bit normalization
time_norm   = False             # False (no time normalization), or running-mean, one-bit normalization
rm_resp     = False             # False (not removing instr), or "polozeros", "RESP_files", "spectrum", "inv"
cc_method   = 'deconv'          # select between raw, deconv and coherency
save_fft    = True              # True to save fft data, or False
flag        = False             # print intermediate variables and computing time for debugging purpose

# pre-processing parameters 
samp_freq = 10                  # targeted sampling rate 
dt        = 1/samp_freq
cc_len    = 3600                # basic unit of data length for fft (s)
step      = 1800                # overlapping between each cc_len (s)
freqmin   = 0.05                # min frequency for the filter
freqmax   = 4                   # max frequency for the filter
smooth_N  = 100                 # moving window length for time/freq domain normalization if selected

# information on data chunck
start_date = ["2018_05_01"]     # start date of downloaded data or locally-stored data
end_date   = ["2018_05_31"]     # end date of data
inc_days   = 20                 # basic unit length of data for pre-processing and storage

# make a dictionary to store all variables: also for later cc
para_dic={'samp_freq':samp_freq,'dt':dt,'cc_len':cc_len,'step':step,\
    'freqmin':freqmin,'freqmax':freqmax,'rm_resp':rm_resp,'resp_dir':resp_dir,\
    'prepro':prepro,'to_whiten':to_whiten,'time_norm':time_norm,\
    'cc_method':cc_method,'smooth_N':smooth_N,'data_format':input_fmt,\
    'station.list':locations,'rootpath':rootpath,'FFTDIR':FFTDIR,\
    'start_date':start_date[0],'end_date':end_date[0],'inc_days':inc_days}

#--save fft metadata for later use--
fout = open(metadata,'w')
fout.write(str(para_dic))
fout.close()

