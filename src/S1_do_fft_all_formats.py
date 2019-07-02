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
Step1 of the NoisePy package.

This script pre-processs the noise data for each single station using the parameters given below
and then stored the whitened and nomalized FFT trace for each station in a ASDF format. 
by C.Jiang, M.Denolle, T.Clements (Nov.09.2018)

Update history:
    - allow to handle SAC, MiniSeed and ASDF formate inputs in one script. (Apr.18.2019)

    - add a sub-function to make time-domain and freq-domain normalization according to 
    Bensen et al., 2007. (May.10.2019; see Notes for more related information)

    - cast fft parameters into a dictionary to pass between sub-functions. (Jun.16.2019)

    - simplify the main program by cast some processing steps into sub-functions. (Jun.28.2019)

Note:
    1. !!!!!!!!VERY IMPORTANT!!!!!!!
    We choose one-day (24 hours) as the basic length for data storage and processing. this means if 
    you input 10-day long continous data, we will break them into 1-day long and do processing
    thereafter on the 1-day long data. we remove the day if it is shorter than 0.5 day. for the
    daily-long segment, we tend to break them into smaller chunck with certain overlapping for CCFs
    in order to increase the SNR. If you want to keep longer duration as the basic length (2 days), 
    please check/modify the subfunction of preprocess_raw and the associated functions.

    2. Here we choose to compute all of the FFTs, even if the window contain earthquakes. We calculate 
    statistical metrics in each window and save that in the parameters of the FFT. The selection will
    be made in the cross correlation step.

    3. Note that neither time and frequency normalization is needed if coherency (potentially deconv
    as well) method is selected. See Seats et al., 2012 for details (DOI:10.1111/j.1365-246X.2011.05263.x)

To add: the kurtosis metrics.
'''

t00=time.time()

########################################
#########PARAMETER SECTION##############
########################################

#------absolute path parameters-------
rootpath  = '/mnt/data0/NZ/XCORR'                        # root path for this data processing
FFTDIR = os.path.join(rootpath,'FFT/')                   # dir to store FFT data
event = os.path.join(rootpath,'noise_data/Event_*')      # dir where noise data is located
if (len(glob.glob(event))==0): 
    raise ValueError('No data file in %s',event)

#-----------other useful input files-----------
locations = os.path.join(rootpath,'locations.txt')       # station list - requred for data in SAC/miniSEED format 
resp_dir  = os.path.join(rootpath,'response')            # only needed when resp set to something other than 'inv'
f_metadata = os.path.join(rootpath,'fft_metadata.txt')   # keep a record of used parameters

#-------some control parameters--------
input_fmt   = 'asdf'            # string: 'asdf', 'sac','mseed' 
prepro      = False             # preprocess the data (correct time/downsampling/trim data/response removal)?
to_whiten   = False             # False (no whitening), or running-mean, one-bit normalization
time_norm   = False             # False (no time normalization), or running-mean, one-bit normalization
rm_resp     = False             # False (not removing instr), or "polozeros", "RESP_files", "spectrum", "inv"
cc_method   = 'deconv'          # select between raw, deconv and coherency
flag        = False             # print intermediate variables and computing time for debugging purpose

#------pre-processing parameters-------
samp_freq = 10                  # targeted sampling rate 
dt        = 1/samp_freq
cc_len    = 3600                # basic unit of data length for fft (s)
step      = 1800                # overlapping between each cc_len (s)
freqmin   = 0.05                # min frequency for the filter
freqmax   = 4                   # max frequency for the filter
smooth_N  = 100                 # moving window length for time/freq domain normalization if selected

# data information: when to start, end and increment interval 
start_date = ["2018_05_01"]     # start date of downloaded data or locally-stored data
end_date   = ["2018_05_31"]     # end date of data
inc_days   = 20                 # basic unit length of data for pre-processing and storage

# make a dictionary to store all variables: also for later cc
fft_para={'samp_freq':samp_freq,'dt':dt,'cc_len':cc_len,'step':step,\
    'freqmin':freqmin,'freqmax':freqmax,'rm_resp':rm_resp,'resp_dir':resp_dir,\
    'prepro':prepro,'to_whiten':to_whiten,'time_norm':time_norm,\
    'cc_method':cc_method,'smooth_N':smooth_N,'data_format':input_fmt,\
    'station.list':locations,'rootpath':rootpath,'FFTDIR':FFTDIR,\
    'start_date':start_date[0],'end_date':end_date[0],'inc_days':inc_days}

#--save fft metadata for later use--
fout = open(f_metadata,'w')
fout.write(str(fft_para))
fout.close()


#######################################
###########PROCESSING SECTION##########
#######################################

#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#-------make sure only one input type is selected--------
if (not input_fmt):
    raise ValueError('Have to choose a format!')
elif ((input_fmt!='asdf') and (input_fmt!='sac') and (input_fmt!='mseed')):
    raise ValueError('data format not understood! either sac, mseed, or asdf')

if rank == 0:
    #-----check whether dir exist----
    if not os.path.isdir(FFTDIR):
        os.mkdir(FFTDIR)

    tdir = sorted(glob.glob(event))
    print(tdir)

    if input_fmt == 'asdf':
        splits = len(tdir)
        nsta=len(tdir)
    else:
        locs = pd.read_csv(locations)
        nsta = len(locs)
        splits = nsta

    if  nsta==0:
        raise IOError('Abort! no available seismic files for doing FFT')

else:
    if input_fmt == 'asdf':
        splits,tdir = [None for _ in range(2)]
    else:
        splits,locs,tdir = [None for _ in range(3)]

#-----broadcast the variables------
splits = comm.bcast(splits,root=0)
tdir = comm.bcast(tdir,root=0)
if input_fmt!='asdf':locs = comm.bcast(locs,root=0)
extra = splits % size

#--------MPI: loop through each station--------
for ista in range (rank,splits+size-extra,size):
    if ista<splits:
        t10=time.time()

        if input_fmt=='asdf':    
            
            ds=pyasdf.ASDFDataSet(tdir[ista],mode='r')          # read the file
            temp = ds.waveforms.list()                          # list all of the waveforms
            if (len(temp)==0) or (len(temp[0].split('.')) != 2):
                continue

            # get station name
            network = temp[0].split('.')[0]
            station = temp[0].split('.')[1]

            # assume inventory created
            inv1 = ds.waveforms[temp[0]]['StationXML']

            if inv1[0][0][0].location_code:
                location = inv1[0][0][0].location_code
            else:
                location = ''

            if flag:print("working on station %s " % station)

            #---------construct a pd structure for fft_parameter functions later----------
            locs = pd.DataFrame([[inv1[0][0].latitude,inv1[0][0].longitude,inv1[0][0].elevation]],\
                columns=['latitude','longitude','elevation'])
            
            #------get day information: works better than just list the tags------
            all_tags = ds.waveforms[temp[0]].get_waveform_tags()
            if len(all_tags)==0:continue

            #----loop through each stream----
            for itag in range(len(all_tags)):
                if flag:print("working on trace " + all_tags[itag])

                source = ds.waveforms[temp[0]][all_tags[itag]]
                comp = source[0].stats.channel
                if len(source)==0:continue

                # ensure that pre-processing was performed 
                if ((all_tags[itag].split('_')[0] != 'raw') and (prepro)):
                    print('warning! it appears pre-processing has not yet been performed! will do now')
                    t0=time.time()
                    source = noise_module.preprocess_raw(source,inv1,fft_para)
                    t1=time.time()
                    if flag:print("prepro takes %f s" % (t1-t0))

                # cut daily-long data into smaller segments (dataS always in 2D)
                source_params,dataS_t,dataS,dataS_stats = noise_module.cut_trace_make_statis(fft_para,source,flag)
                N = dataS.shape[0]

                # do normalization if needed
                source_white = noise_module.noise_processing(fft_para,dataS,flag)
                Nfft = source_white.shape[1]
                if flag:print('N and Nfft are %d %d' %(N,Nfft))

                # save FFTs into HDF5 format
                crap=np.zeros(shape=(N,Nfft//2),dtype=np.complex64)
                fft_h5 = os.path.join(FFTDIR,network+'.'+station+'.'+location+'.h5')

                if not os.path.isfile(fft_h5):
                    with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                        pass # create pyasdf file 
        
                with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                    parameters = noise_module.fft_parameters(fft_para,source_params, \
                        locs.iloc[0],comp,Nfft,dataS_t[:,0])
                    
                    savedate = '{0:04d}_{1:02d}_{2:02d}'.format(dataS_stats.starttime.year,\
                        dataS_stats.starttime.month,dataS_stats.starttime.day)
                    path = savedate
                    data_type = str(comp)
                    if ((itag==0) and (len(fft_ds.waveforms[temp[0]]['StationXML']))==0):
                        fft_ds.add_stationxml(inv1)

                    crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                    fft_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    print(savedate)

        else:

            #----loop through each day on each core----
            for jj in range (len(tdir)):
                station = locs.iloc[ista]['station']
                network = locs.iloc[ista]['network']
                try:
                    location = locs.iloc[ista]['location']
                except Exception:
                    location = ''

                if flag:print("working on station %s " % station)
                
                #-----------SAC and MiniSeed both work here-----------
                tfiles = glob.glob(os.path.join(tdir[jj],'*'+station+'*'+location+'*'))
                if len(tfiles)==0:
                    print(str(station)+' does not have sac file at '+str(tdir[jj]));continue

                #----loop through each channel----
                for tfile in tfiles:
                    sacfile = os.path.basename(tfile)
                    try:
                        source = obspy.read(tfile)
                    except Exception as inst:
                        print(inst)
                        continue
                    comp = source[0].stats.channel
                    
                    if flag:
                        print("working on sacfile %s" % sacfile)

                    #---------make an inventory---------
                    inv1=noise_module.stats2inv(source[0].stats)
                    
                    if prepro:
                        t0=time.time()
                        source = noise_module.preprocess_raw(source,inv1,fft_para)
                        t1=time.time()                    
                        if len(source)==0:continue
                        if flag:print("prepro takes %f s" % (t1-t0))

                    # cut daily-long data into smaller segments
                    source_params,dataS_t,dataS,dataS_stats = noise_module.cut_trace_make_statis(fft_para,source,flag)
                    N = dataS.shape[0]

                    # do normalization if needed
                    source_white = noise_module.noise_processing(fft_para,dataS,flag)
                    Nfft = source_white.shape[1]
                    if flag:print('N and Nfft are %d %d' % (N,Nfft))

                    # save FFTs into HDF5 format
                    crap=np.zeros(shape=(N,Nfft//2),dtype=np.complex64)
                    fft_h5 = os.path.join(FFTDIR,network+'.'+station+'.'+location+'.h5')

                    if not os.path.isfile(fft_h5):
                        with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                            pass # create pyasdf file 
            
                    with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                        parameters = noise_module.fft_parameters(fft_para,source_params, \
                            locs.iloc[0],comp,Nfft,dataS_t[:,0])
                        
                        savedate = '{0:04d}_{1:02d}_{2:02d}'.format(dataS_stats.starttime.year,\
                            dataS_stats.starttime.month,dataS_stats.starttime.day)
                        path = savedate
                        data_type = str(comp)
                        
                        fft_ds.add_stationxml(inv1)
                        crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                        fft_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
        
        t11=time.time()
        print('it takes '+str(t11-t10)+' s to process one station in step 1')

t01=time.time()
print('step1 takes '+str(t01-t00)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
