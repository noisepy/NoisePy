import os
import sys
import glob
import time
import scipy
import obspy
import pyasdf
import numpy as np
import pandas as pd
import noise_module
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.fftpack.helper import next_fast_len

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
Main script of the NoisePy package.
by C.Jiang, M.Denolle, T.Clements
'''

t00=time.time()

########################################
#########PARAMETER SECTION##############
########################################

#------absolute path parameters-------
rootpath  = '/mnt/data0/NZ/XCORR'                       # root path for this data processing
FFTDIR    = os.path.join(rootpath,'FFT')                # dir to store FFT data
CCFDIR    = os.path.join(rootpath,'CCF')                # dir to store CC data
data_dir  = os.path.join(rootpath,'RAW_DATA')           # dir where noise data is located
if (len(glob.glob(data_dir))==0): 
    raise ValueError('No data file in %s',data_dir)

# load useful download info
dfile = os.path.join(data_dir,'download_info.txt')
down_info = eval(open(dfile).read())
samp_freq = down_info['samp_freq']
freqmin   = down_info['freqmin']
freqmax   = down_info['freqmax']
start_date = down_info['start_date']
end_date   = down_info['end_date']
inc_hours  = down_info['inc_hours']

#-------some control parameters--------
input_fmt   = 'asdf'            # string: 'asdf', 'sac','mseed' 
to_whiten   = False             # False (no whitening), or running-mean, one-bit normalization
time_norm   = False             # False (no time normalization), or running-mean, one-bit normalization
cc_method   = 'deconv'          # select between raw, deconv and coherency
save_fft    = True              # True to save fft data, or False
flag        = False             # print intermediate variables and computing time for debugging purpose

# pre-processing parameters 
dt        = 1/samp_freq
cc_len    = 3600                # basic unit of data length for fft (s)
step      = 1800                # overlapping between each cc_len (s)
smooth_N  = 100                 # moving window length for time/freq domain normalization if selected

# criteria for data selection
max_over_std = 10                           # maximum threshold between the maximum absolute amplitude and the STD of the time series
max_kurtosis = 10                           # max kurtosis allowed.

# maximum memory allowed per core in Gb.
MAX_MEM = 4.0

# make a dictionary to store all variables: also for later cc
para_dic={'samp_freq':samp_freq,'dt':dt,'cc_len':cc_len,'step':step,'freqmin':freqmin,\
    'freqmax':freqmax,'to_whiten':to_whiten,'time_norm':time_norm,'cc_method':cc_method,\
    'smooth_N':smooth_N,'data_format':input_fmt,'rootpath':rootpath,'FFTDIR':FFTDIR,\
    'start_date':start_date[0],'end_date':end_date[0],'inc_hours':inc_hours}
# save fft metadata for future reference
fc_metadata  = os.path.join(rootpath,'fft_cc_data.txt')        # keep a record of used parameters


#######################################
###########PROCESSING SECTION##########
#######################################

#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if save_fft:
        if not os.path.isdir(FFTDIR):os.mkdir(FFTDIR)
    if not os.path.isdir(CCFDIR):os.mkdir(CCFDIR)
    
    # save metadata 
    fout = open(fc_metadata,'w')
    fout.write(str(para_dic));fout.close()

    # set variables to broadcast
    tdir = sorted(glob.glob(data_dir))
    nchunck = len(tdir)
    splits  = nchunck
    if nchunck==0:
        raise IOError('Abort! no available seismic files for doing FFT')
else:
    splits,tdir = [None for _ in range(2)]

# broadcast the variables
splits = comm.bcast(splits,root=0)
tdir  = comm.bcast(tdir,root=0)
extra = splits % size

# MPI loop: loop through each user-defined time chunck
for ick in range (rank,splits+size-extra,size):
    if ick<splits:
        t10=time.time()   
            
        ds=pyasdf.ASDFDataSet(tdir[ick],mode='r')         
        sta_list = ds.waveforms.list()    
        nsta     = len(sta_list)
        if (nsta==0):
            if flag:
                print('no data in %s'%tdir[ick]);continue

        # crude estimation on memory requirement (assume float32)
        nsec_chunck = inc_hours/24*86400
        nseg_chunck = np.floor(nsec_chunck/cc_len)
        npts_chunck = nseg_chunck*cc_len*samp_freq 
        memory_size = nsta*npts_chunck*4/1024/1024/1024
        if memory_size > MAX_MEM:
            print('Memory exceeds %s GB! No enough memory to load %s h of data all once!' % (MAX_MEM,inc_hours))

        # loop through all stations
        for ista in range(nsta):
            tmps = sta_list[ista]
            # get station and inventory
            network = tmps.split('.')[0]
            station = tmps.split('.')[1]
            inv1 = ds.waveforms[tmps]['StationXML']

            if inv1[0][0][0].location_code:
                location = inv1[0][0][0].location_code
            else: location = '00'

            if flag:print("working on station %s " % station)

            #---------construct a pd structure for fft_parameter functions later----------
            locs = pd.DataFrame([[inv1[0][0].latitude,inv1[0][0].longitude,inv1[0][0].elevation]],\
                columns=['latitude','longitude','elevation'])
            
            #------get day information: works better than just list the tags------
            all_tags = ds.waveforms[tmps].get_waveform_tags()
            if len(all_tags)==0:continue

            #----loop through each stream----
            for itag in range(len(all_tags)):
                if flag:print("working on trace " + all_tags[itag])

                source = ds.waveforms[tmps][all_tags[itag]]
                comp = source[0].stats.channel4
                if len(source)==0:continue

                # cut daily-long data into smaller segments (dataS always in 2D)
                source_params,dataS_t,dataS,dataS_stats = noise_module.cut_trace_make_statis(para_dic,source,flag)
                N = dataS.shape[0]

                # do normalization if needed
                source_white = noise_module.noise_processing(para_dic,dataS,flag)
                Nfft = source_white.shape[1]
                if flag:print('N and Nfft are %d %d' %(N,Nfft))

                if save_fft:
                    # save FFTs into HDF5 format
                    crap=np.zeros(shape=(N,Nfft//2),dtype=np.complex64)
                    fft_h5=os.path.join(FFTDIR,tdir[ick])

                    if not os.path.isfile(fft_h5):
                        with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                            pass # create pyasdf file 
            
                    with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                        parameters = noise_module.fft_parameters(para_dic,source_params, \
                            locs.iloc[0],Nfft,dataS_t[:,0])
                        
                        savedate = '{0:s}_{1:s}_{2:s}_{3:s}'.format(station,network,comp,location)
                        path = savedate
                        data_type = str(comp)
                        if ((itag==0) and (len(fft_ds.waveforms[tmps]['StationXML']))==0):
                            fft_ds.add_stationxml(inv1)

                        crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                        fft_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                        print(savedate)

                # load the fft data in memory


        # make cross-correlations 

