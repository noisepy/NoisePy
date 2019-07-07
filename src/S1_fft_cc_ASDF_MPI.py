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

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
This main script of NoisePy:
    1) read the saved noise data in user-defined chunck as inc_hours, cut them into smaller
        length segments, do general pre-processing (trend, normalization) and then do FFT;
    2) output FFT data of each station in ASDF format if needed and load them in memory for
        later cross-correlation;
    3) perform cross-correlation for all station pairs in that time chunck and output the
        sub-stacked (if selected) into ASDF format

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@fas.harvard.edu)
        
Note:
    1) tune the script to read SAC/miniSEED files
    2) implement max_kurtosis?
'''

tt0=time.time()

########################################
#########PARAMETER SECTION##############
########################################

#------absolute path parameters-------
rootpath  = '/Users/chengxin/Documents/Harvard/NoisePy/v4.0_July'                       # root path for this data processing
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
nsta       = down_info['inc_hours']

#-------some control parameters--------
input_fmt   = 'asdf'            # string: 'asdf', 'sac','mseed' 
to_whiten   = False             # False (no whitening), or running-mean, one-bit normalization
time_norm   = False             # False (no time normalization), or running-mean, one-bit normalization
cc_method   = 'deconv'          # select between raw, deconv and coherency
save_fft    = True              # True to save fft data, or False
flag        = True              # print intermediate variables and computing time for debugging purpose

# pre-processing parameters 
dt        = 1/samp_freq
cc_len    = 3600                # basic unit of data length for fft (s)
step      = 1800                # overlapping between each cc_len (s)
smooth_N  = 100                 # moving window length for time/freq domain normalization if selected

# cross-correlation parameters
maxlag         = 500            # lags of cross-correlation to save
substack       = True           # sub-stack daily cross-correlation or not
substack_len   = 4*cc_len       # Time unit in sectons to stack over: need to be integer times of cc_len
smoothspect_N  = 10             # moving window length to smooth spectrum amplitude

# criteria for data selection
max_over_std = 10               # maximum threshold between the maximum absolute amplitude and the STD of the time series
max_kurtosis = 10               # max kurtosis allowed.

# maximum memory allowed per core in GB
MAX_MEM = 4.0

# make a dictionary to store all variables: also for later cc
fc_para={'samp_freq':samp_freq,'dt':dt,'cc_len':cc_len,'step':step,'freqmin':freqmin,'freqmax':freqmax,\
    'to_whiten':to_whiten,'time_norm':time_norm,'cc_method':cc_method,'smooth_N':smooth_N,'data_format':\
    input_fmt,'rootpath':rootpath,'FFTDIR':FFTDIR,'start_date':start_date[0],'end_date':end_date[0],\
    'inc_hours':inc_hours,'substack':substack,'substack_len':substack_len,'smoothspect_N':smoothspect_N,\
    'maxlag':maxlag,'max_over_std':max_over_std,'max_kurtosis':max_kurtosis,'MAX_MEM':MAX_MEM}
# save fft metadata for future reference
fc_metadata  = os.path.join(rootpath,'fft_cc_data.txt')       


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
    fout.write(str(fc_para));fout.close()

    # set variables to broadcast
    tdir = sorted(glob.glob(os.path.join(data_dir,'*.h5')))
    nchunck = len(tdir)
    splits  = nchunck
    if nchunck==0:
        raise IOError('Abort! no available seismic files for FFT')
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
        if (len(sta_list)==0):
            print('continue! no data in %s'%tdir[ick]);continue

        # crude estimation on memory needs (assume float32)
        nsec_chunck = inc_hours/24*86400
        nseg_chunck = int(np.floor((nsec_chunck-cc_len)/step))+1
        npts_chunck = int(nseg_chunck*cc_len*samp_freq)
        memory_size = nsta*npts_chunck*4/1024**3
        if memory_size > MAX_MEM:
            raise ValueError('Require %s G memory (%s GB provided)! Reduce inc_hours as it cannot load %s h all once!' % (memory_size,MAX_MEM,inc_hours))

        nnfft = int(next_fast_len(int(cc_len*samp_freq+1)))
        # open array to store fft data/info in memory
        fft_array = np.zeros((nsta,nseg_chunck*nnfft//2),dtype=np.complex64)
        fft_std   = np.zeros((nsta,nseg_chunck),dtype=np.float32)
        fft_flag  = np.zeros(nsta,dtype=np.int16)
        fft_time  = np.zeros((nsta,nseg_chunck),dtype=np.float64) 
        # station information (for every channel)
        station=[];network=[];channel=[];clon=[];clat=[];location=[];elevation=[]     

        # loop through all stations
        iii = 0
        for ista in range(len(sta_list)):
            tmps = sta_list[ista]
            # get station and inventory
            try:
                inv1 = ds.waveforms[tmps]['StationXML']
            except Exception as e:
                print(e);raise ValueError('abort! no stationxml for %s in file %s'%(tmps,tdir[ick]))
            sta,net,lon,lat,elv,loc = noise_module.sta_info_from_inv(inv1)

            #------get day information: works better than just list the tags------
            all_tags = ds.waveforms[tmps].get_waveform_tags()
            if len(all_tags)==0:continue

            #----loop through each stream----
            for itag in range(len(all_tags)):
                if flag:print("working on station %s and trace %s" % (sta,all_tags[itag]))

                source = ds.waveforms[tmps][all_tags[itag]]
                comp = source[0].stats.channel
                if len(source)==0:continue
                station.append(sta);network.append(net);channel.append(comp),clon.append(lon)
                clat.append(lat);location.append(loc);elevation.append(elv)

                # cut daily-long data into smaller segments (dataS always in 2D)
                source_params,dataS_t,dataS = noise_module.cut_trace_make_statis(fc_para,source,flag)
                N = dataS.shape[0]

                # do normalization if needed
                source_white = noise_module.noise_processing(fc_para,dataS,flag)
                Nfft = source_white.shape[1];Nfft2 = Nfft//2
                if flag:print('N and Nfft are %d (proposed %d),%d (proposed %d)' %(N,nseg_chunck,Nfft,nnfft))

                if save_fft:
                    # save FFTs into HDF5 format
                    crap=np.zeros(shape=(N,Nfft2),dtype=np.complex64)
                    fft_h5=os.path.join(FFTDIR,tdir[ick].split('/')[-1])

                    with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                        parameters = noise_module.fft_parameters(fc_para,source_params,inv1,Nfft,dataS_t[:,0])
                        
                        path = '{0:s}_{1:s}_{2:s}_{3:s}'.format(net,sta,comp,str(loc))
                        if itag==0:
                            try: fft_ds.add_stationxml(inv1)
                            except Exception: pass

                        crap[:,:Nfft2]=source_white[:,:Nfft2]
                        fft_ds.add_auxiliary_data(data=crap, data_type='FFT', path=path, parameters=parameters)

                # load fft data in memory for cross-correlations
                data = source_white[:,:Nfft2]
                fft_array[iii] = data.reshape(data.size)
                fft_std[iii]   = source_params[:,1]
                fft_flag[iii]  = 1
                fft_time[iii]  = dataS_t[:,0]
                iii+=1
        del ds

        # check whether array size is enough
        if iii!=nsta:
            print('it seems some data is missing in download step, but it is okay!')

        # make cross-correlations 
        path_array=[]
        for iiS in range(iii-1):
            fft1 = fft_array[iiS]
            source_std = fft_std[iiS]
            if not fft_flag[iiS]: continue
                    
            t0=time.time()
            #-----------get the smoothed source spectrum for decon later----------
            sfft1 = noise_module.smooth_source_spect(fc_para,fft1)
            sfft1 = sfft1.reshape(N,Nfft2)
            t1=time.time()
            if flag: 
                print('smoothing source takes %6.4fs' % (t1-t0))

            #-----------now loop III for each receiver B----------
            for iiR in range(iiS+1,iii):
                if flag:print('receiver: %s %s' % (station[iiR],network[iiR]))
                if not fft_flag[iiR]: continue
                    
                fft2 = fft_array[iiR];sfft2 = fft2.reshape(N,Nfft2)
                receiver_std = fft_std[iiR]

                #---------- check the existence of earthquakes ----------
                rec_ind = np.where((receiver_std<fc_para['max_over_std'])&(receiver_std>0)&(np.isnan(receiver_std)==0))[0]
                sou_ind = np.where((source_std<fc_para['max_over_std'])&(source_std>0)&(np.isnan(source_std)==0))[0]
                bb=np.intersect1d(sou_ind,rec_ind)
                if len(bb)==0:continue

                t2=time.time()
                corr,tcorr,ncorr=noise_module.optimized_correlate(sfft1[bb,:],sfft2[bb,:],fc_para,Nfft,fft_time[iiR][bb])
                t3=time.time()

                #---------------keep daily cross-correlation into a hdf5 file--------------
                cc_h5 = os.path.join(CCFDIR,tdir[ick].split('/')[-1])
                crap  = np.zeros(corr.shape,dtype=corr.dtype)

                with pyasdf.ASDFDataSet(cc_h5,mpi=False) as ccf_ds:
                    coor = {'lonS':clon[iiS],'latS':clat[iiS],'lonR':clon[iiS],'latR':clat[iiR]}
                    parameters = noise_module.optimized_cc_parameters(fc_para,coor,tcorr,ncorr)

                    path = network[iiS]+'s'+station[iiS]+'s'+channel[iiS]+'s'+str(location[iiS])+\
                        's'+network[iiR]+'s'+station[iiR]+'s'+channel[iiR]+'s'+str(location[iiR])
                    crap[:] = corr[:]
                    ccf_ds.add_auxiliary_data(data=crap, data_type='CCF', path=path, parameters=parameters)
                t4=time.time()
                # keep a track of the path information used for later stacking 
                if path not in path_array:
                    path_array.append(path)

                t5=time.time()
                if flag:
                    print('read S %6.4fs, cc %6.4fs, write cc %6.4fs'% ((t1-t0),(t3-t2),(t4-t3)))

        fft_array=[];fft_std=[];fft_flag=[];fft_time=[]
        n = gc.collect();print('unreadable garbarge',n)

        # save the ASDF path info for later stacking use
        path_para = {'paths':path_array}
        pfile = os.path.join(rootpath,'CCF/paths_'+str(rank)+'.txt')
        fout  = open(pfile,'w')
        fout.write(str(path_para));fout.close()
    
    t11 = time.time()
    print('it takes %6.4s to process the chunck of %s' % (t11-t10,tdir[ick]))

tt1 = time.time()
print('it takes %6.4fs to process step 1' % (tt1-tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()
