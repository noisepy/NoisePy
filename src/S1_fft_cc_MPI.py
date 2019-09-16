import gc
import sys
import time
import scipy
import obspy
import pyasdf
import datetime
import os, glob
import numpy as np
import pandas as pd
import noise_module
from mpi4py import MPI
from scipy.fftpack.helper import next_fast_len
import matplotlib.pyplot  as plt

# ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
This main script of NoisePy:
    1) reads the saved noise data in user-defined chunck as inc_hours, cut them into smaller
        length segments, do general pre-processing (trend, normalization) and then do FFT;
    2) outputs FFT data of each station in ASDF format if needed and load them in memory for
        later cross-correlation;
    3) performs cross-correlation for all station pairs in that time chunck and output the
        sub-stacked (if selected) into ASDF format;
    4) provides the option to deal with tidy SAC/mseed data at local machine. (Jul.8.2019)

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@fas.harvard.edu)
        
Note:
    1) to read SAC/mseed files, we assume the users have sorted the data by the time chunck they prefer (e.g., 1day) 
        and store them in folders named after the time chunck (e.g, 2010_10_1). modify L135 to find your local data 
    2) a script of S0B_sacMSEED_to_ASDF.py is provided to help clean messy SAC/MSEED data and transfer
        them into ASDF format. the script takes minor time compared to that for cross-correlation times. so we recommend 
        to use this script to deal with SAC/mseed data for better NoisePy performance. 
    3) when whitening is applied, we recommend to use 'raw' method for cross-correlation, because the whitening is 
        equavilent to coherency cross-correlation
'''

tt0=time.time()

########################################
#########PARAMETER SECTION##############
########################################

#------absolute path parameters-------
# rootpath = '/n/scratchssdlfs/denolle_lab/CCFs_Kanto'
rootpath  = '/Users/chengxin/Documents/NoisePy_example/Kanto'                      # root path for this data processing
CCFDIR    = os.path.join(rootpath,'CCF')                        # dir to store CC data
DATADIR   = os.path.join(rootpath,'CLEANED_DATA')                   # dir where noise data is located

#-------some control parameters--------
input_fmt   = 'asdf'            # string: 'asdf', 'sac','mseed' 
to_whiten   = False             # False (no whitening), or running-mean, one-bit normalization
time_norm   = False             # False (no time normalization), or running-mean, one-bit normalization
cc_method   = 'coherency'       # select between raw and coherency (deconv is not symmetric!)
save_fft    = False             # True to save fft data, or False
flag        = False             # print intermediate variables and computing time for debugging purpose
ncomp       = 3                 # 1 or 3 component data (needed to decide whether do rotation)

# station/instrument info for input_fmt=='sac' or 'mseed'
stationxml = False              # station.XML file exists or not
rm_resp    = 'no'               # 'no','inv','spectrum','RESP','polozeros'
respdir    = 'none'     

# pre-processing parameters 
cc_len    = 1800                # basic unit of data length for fft (sec)
step      = 450                 # overlapping between each cc_len (sec)
smooth_N  = 100                 # moving window length for time/freq domain normalization if selected (points)

# cross-correlation parameters
maxlag         = 200            # lags of cross-correlation to save (sec)
substack       = True           # sub-stack daily cross-correlation or not
substack_len   = cc_len         # Time unit in sections to stack over: need to be multiples of cc_len
smoothspect_N  = 10             # moving window length to smooth spectrum amplitude (points)

# load useful download info if start from ASDF
if input_fmt == 'asdf':
    dfile = os.path.join(DATADIR,'download_info.txt')
    down_info = eval(open(dfile).read())
    samp_freq = down_info['samp_freq']
    freqmin   = down_info['freqmin']
    freqmax   = down_info['freqmax']
    start_date = down_info['start_date']
    end_date   = down_info['end_date']
    inc_hours  = down_info['inc_hours']     # NOTE : WE NEED TO BE ABLE TO BE FLEXIBLE FROM DOWNLOADS.
else:   # sac or mseed format
    samp_freq = 20
    freqmin   = 0.05
    freqmax   = 4
    start_date = ["2010_12_06_0_0_0"]
    end_date   = ["2010_12_15_0_0_0"]
    inc_hours  = 6
dt = 1/samp_freq

# criteria for data selection
max_over_std = 10               # maximum threshold between the maximum absolute amplitude and the STD of the time series
max_kurtosis = 10               # max kurtosis allowed.

# maximum memory allowed per core in GB
MAX_MEM = 4.0

# make a dictionary to store all variables: also for later cc
fc_para={'samp_freq':samp_freq,'dt':dt,'cc_len':cc_len,'step':step,'freqmin':freqmin,'freqmax':freqmax,\
    'to_whiten':to_whiten,'time_norm':time_norm,'cc_method':cc_method,'smooth_N':smooth_N,'data_format':\
    input_fmt,'rootpath':rootpath,'CCFDIR':CCFDIR,'start_date':start_date[0],'end_date':end_date[0],\
    'inc_hours':inc_hours,'substack':substack,'substack_len':substack_len,'smoothspect_N':smoothspect_N,\
    'maxlag':maxlag,'max_over_std':max_over_std,'max_kurtosis':max_kurtosis,'MAX_MEM':MAX_MEM,'ncomp':ncomp,\
    'stationxml':stationxml,'rm_resp':rm_resp,'respdir':respdir}
# save fft metadata for future reference
fc_metadata  = os.path.join(CCFDIR,'fft_cc_data.txt')       

#######################################
###########PROCESSING SECTION##########
#######################################

#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if not os.path.isdir(CCFDIR):os.mkdir(CCFDIR)
    
    # save metadata 
    fout = open(fc_metadata,'w')
    fout.write(str(fc_para));fout.close()

    # set variables to broadcast
    if input_fmt == 'asdf':
        tdir = sorted(glob.glob(os.path.join(DATADIR,'*.h5')))
    else:
        tdir = sorted(glob.glob(os.path.join(DATADIR,'Event_2010_*')))
        if len(tdir)==0: raise ValueError('No data file in %s',DATADIR)
        # get nsta by loop through all event folder
        nsta = 0
        for ii in range(len(tdir)):
            tnsta = len(glob.glob(os.path.join(tdir[ii],'*'+input_fmt)))
            if nsta<tnsta:nsta=tnsta

    nchunck = len(tdir)
    splits  = nchunck
    if nchunck==0:
        raise IOError('Abort! no available seismic files for FFT')
else:
    if input_fmt == 'asdf':
        splits,tdir = [None for _ in range(2)]
    else: splits,tdir,nsta = [None for _ in range(3)]

# broadcast the variables
splits = comm.bcast(splits,root=0)
tdir  = comm.bcast(tdir,root=0)
if input_fmt != 'asdf': nsta = comm.bcast(nsta,root=0)

# MPI loop: loop through each user-defined time chunck
for ick in range (rank,splits,size):
    t10=time.time()   

    #############LOADING NOISE DATA AND DO FFT##################

    # check whether time chunck been processed or not
    if input_fmt == 'asdf':
        tmpfile = os.path.join(CCFDIR,tdir[ick].split('/')[-1].split('.')[0]+'.tmp')
    else: 
        tmpfile = os.path.join(CCFDIR,tdir[ick].split('/')[-1]+'.tmp')
    if os.path.isfile(tmpfile):continue
    
    # retrive station information
    if input_fmt == 'asdf':
        ds=pyasdf.ASDFDataSet(tdir[ick],mpi=False,mode='r') 
        sta_list = ds.waveforms.list()
        nsta=ncomp*len(sta_list)
        print('found %d stations in total'%nsta)
    else:
        sta_list = sorted(glob.glob(os.path.join(tdir[ick],'*'+input_fmt)))
    if (len(sta_list)==0):
        print('continue! no data in %s'%tdir[ick]);continue

    # crude estimation on memory needs (assume float32)
    nsec_chunck = inc_hours/24*86400
    nseg_chunck = int(np.floor((nsec_chunck-cc_len)/step))
    npts_chunck = int(nseg_chunck*cc_len*samp_freq)
    memory_size = nsta*npts_chunck*4/1024**3
    if memory_size > MAX_MEM:
        raise ValueError('Require %s G memory (%s GB provided)! Reduce inc_hours[set as %d now] to reduce memory needs!' % (memory_size,MAX_MEM,inc_hours))

    nnfft = int(next_fast_len(int(cc_len*samp_freq)))
    # open array to store fft data/info in memory
    fft_array = np.zeros((nsta,nseg_chunck*(nnfft//2)),dtype=np.complex64)
    fft_std   = np.zeros((nsta,nseg_chunck),dtype=np.float32)
    fft_flag  = np.zeros(nsta,dtype=np.int16)
    fft_time  = np.zeros((nsta,nseg_chunck),dtype=np.float64) 
    # station information (for every channel)
    station=[];network=[];channel=[];clon=[];clat=[];location=[];elevation=[]     

    # loop through all stations
    iii = 0
    for ista in range(len(sta_list)):
        tmps = sta_list[ista]

        if input_fmt == 'asdf':
            # get station and inventory
            try:
                inv1 = ds.waveforms[tmps]['StationXML']
            except Exception as e:
                print(e)
                # choose abort or continue here???
                raise ValueError('abort! no stationxml for %s in file %s'%(tmps,tdir[ick]))
            sta,net,lon,lat,elv,loc = noise_module.sta_info_from_inv(inv1)

            #------get day information: works better than just list the tags------
            all_tags = ds.waveforms[tmps].get_waveform_tags()
            if len(all_tags)==0:continue
            
        else: # get station information
            all_tags = [1]
            sta = tmps.split('/')[-1]

        #----loop through each stream----
        for itag in range(len(all_tags)):
            if flag:print("working on station %s and trace %s" % (sta,all_tags[itag]))

            # read waveform data
            if input_fmt == 'asdf':
                source = ds.waveforms[tmps][all_tags[itag]]
            else:
                source = obspy.read(tmps)
                inv1   = noise_module.stats2inv(source[0].stats,fc_para)
                sta,net,lon,lat,elv,loc = noise_module.sta_info_from_inv(inv1)

            comp = source[0].stats.channel
            if len(source)==0:continue

            # cut daily-long data into smaller segments (dataS always in 2D)
            #source_params,dataS_t,dataS = noise_module.cut_trace_make_statis(fc_para,source,flag)
            trace_stdS,dataS_t,dataS = noise_module.cut_trace_make_statis(fc_para,source)        # optimized version:3-4 times faster
            if not len(dataS): continue
            N = dataS.shape[0]

            # do normalization if needed
            source_white = noise_module.noise_processing(fc_para,dataS)
            Nfft = source_white.shape[1];Nfft2 = Nfft//2
            if flag:print('N and Nfft are %d (proposed %d),%d (proposed %d)' %(N,nseg_chunck,Nfft,nnfft))

            # keep track of station info
            station.append(sta);network.append(net);channel.append(comp),clon.append(lon)
            clat.append(lat);location.append(loc);elevation.append(elv)

            # load fft data in memory for cross-correlations
            data = source_white[:,:Nfft2]
            fft_array[iii] = data.reshape(data.size)
            fft_std[iii]   = trace_stdS
            fft_flag[iii]  = 1
            fft_time[iii]  = dataS_t
            iii+=1
            del trace_stdS,dataS_t,dataS,source_white,data
    
    if input_fmt == 'asdf': del ds

    # check whether array size is enough
    if iii!=nsta:
        print('it seems some stations miss data in download step, but it is OKAY!')

    #############PERFORM CROSS-CORRELATION##################

    # make cross-correlations 
    for iiS in range(iii):
        fft1 = fft_array[iiS]
        source_std = fft_std[iiS]
        sou_ind = np.where((source_std<fc_para['max_over_std'])&(source_std>0)&(np.isnan(source_std)==0))[0]
        if not fft_flag[iiS] or not len(sou_ind): continue
                
        t0=time.time()
        #-----------get the smoothed source spectrum for decon later----------
        sfft1 = noise_module.smooth_source_spect(fc_para,fft1)
        sfft1 = sfft1.reshape(N,Nfft2)
        t1=time.time()
        if flag: 
            print('smoothing source takes %6.4fs' % (t1-t0))

        #-----------now loop III for each receiver B----------
        for iiR in range(iiS,iii): # include auto correlation!! why not?
            if flag:print('receiver: %s %s' % (station[iiR],network[iiR]))
            if not fft_flag[iiR]: continue
                
            fft2 = fft_array[iiR];sfft2 = fft2.reshape(N,Nfft2)
            receiver_std = fft_std[iiR]

            #---------- check the existence of earthquakes ----------
            rec_ind = np.where((receiver_std<fc_para['max_over_std'])&(receiver_std>0)&(np.isnan(receiver_std)==0))[0]
            bb=np.intersect1d(sou_ind,rec_ind)
            if len(bb)==0:continue

            t2=time.time()
            corr,tcorr,ncorr=noise_module.correlate(sfft1[bb,:],sfft2[bb,:],fc_para,Nfft,fft_time[iiR][bb])
            t3=time.time()

            #---------------keep daily cross-correlation into a hdf5 file--------------
            if input_fmt == 'asdf':
                tname = tdir[ick].split('/')[-1]
            else: 
                tname = tdir[ick].split('/')[-1]+'.h5'
            cc_h5 = os.path.join(CCFDIR,tname)
            crap  = np.zeros(corr.shape,dtype=corr.dtype)

            with pyasdf.ASDFDataSet(cc_h5,mpi=False) as ccf_ds:
                coor = {'lonS':clon[iiS],'latS':clat[iiS],'lonR':clon[iiR],'latR':clat[iiR]}
                parameters = noise_module.cc_parameters(fc_para,coor,tcorr,ncorr)

                # source-receiver pair
                data_type = network[iiS]+'s'+station[iiS]+'s'+network[iiR]+'s'+station[iiR]
                path = channel[iiS]+'s'+channel[iiR]
                crap[:] = corr[:]
                ccf_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

            t4=time.time()
            if flag:print('read S %6.4fs, cc %6.4fs, write cc %6.4fs'% ((t1-t0),(t3-t2),(t4-t3)))
        
        del fft1,sfft1,fft2,sfft2,source_std,receiver_std

    # create a stamp to show time chunck being done
    ftmp = open(tmpfile,'w')
    ftmp.write('done')
    ftmp.close()

    fft_array=[];fft_std=[];fft_flag=[];fft_time=[]
    n = gc.collect();print('unreadable garbarge',n)

    t11 = time.time()
    print('it takes %6.2fs to process the chunck of %s' % (t11-t10,tdir[ick].split('/')[-1]))

tt1 = time.time()
print('it takes %6.2fs to process step 1 in total' % (tt1-tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()
