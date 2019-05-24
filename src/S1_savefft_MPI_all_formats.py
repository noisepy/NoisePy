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
this script pre-processs the noise data for each single station using the parameters given below and then
stored the whitened and nomalized fft trace for each station in ASDF format. 
- C.Jiang, T.Clements, M.Denolle (Nov.09.2018)

updated to handle SAC, MiniSeed and ASDF formate inputs. mostly follow the same routine!  (Apr.18.2019)

add a sub-function to make time-domain and freq-domain normalization on 3 components simutaneously, which 
is expected to preserve the relative amplitude among them. (May.10.2019)
'''

t00=time.time()

#------absolute path parameters-------
rootpath  = '/Users/chengxin/Documents/Harvard/code_develop/NoisePy/real_data'
FFTDIR = os.path.join(rootpath,'FFT')
event = os.path.join(rootpath,'noise_data/Event_*')
resp_dir = os.path.join(rootpath,'new_processing')       #needed only when resp is set to something other than 'inv'

#------input file types: make sure it is asdf--------
input_asdf  = False
input_sac   = True
input_mseed = False

#----station.lst needed for sac/mseed data----
if not input_asdf:
    event     = os.path.join(rootpath,'noise_data/Event_*')
    locations = os.path.join(rootpath,'station.lst')

#-----some control parameters------
prepro      = False             #preprocess the data (correct time/downsampling/trim data/response removal)?
to_whiten   = False              #whiten the spectrum?
time_norm   = False              #normalize the data in time domain (remove EQ and ambiguity)?
flag        = False             #print intermediate variables and computing time for debugging purpose

#-----assume response has been removed in downloading process-----
checkt  = True                  # check for traces with points bewtween sample intervals
resp    = 'inv'                 # boolean to remove instrumental response
use_resp_dir = False            # whether to use downloaded inventory to remove response

pre_filt=[0.04,0.05,5,8]
downsamp_freq=10
dt=1/downsamp_freq
cc_len=3600
step=1800
freqmin=0.05  
freqmax=4
norm_type='running_mean'
whiten_type='running_mean'
method='deconv'
if time_norm:
    smooth_N=100 

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

#-------make sure only one input type is selected--------
if (not input_asdf) and (not input_sac) and (not input_mseed):
    raise ValueError('Have to choose a format!')
elif (input_asdf and input_sac) or (input_asdf and input_mseed):
    raise ValueError('Cannot set data format to ASDF and SAC/miniseed at once!')

if rank == 0:
    #-----check whether dir exist----
    if not os.path.isdir(FFTDIR):
        os.mkdir(FFTDIR)

    if input_asdf:
        #----define common variables----
        tdir = sorted(glob.glob(event))
        if len(tdir)==0:
            raise IOError('no available files for doing FFT')
        splits = len(tdir)
    
    else:
        locs = pd.read_csv(locations)
        nsta = len(locs)
        splits = nsta
        tdir = sorted(glob.glob(event))

        if len(tdir)==0 or nsta==0:
            raise IOError('Abort! no available seismic files for doing FFT')
else:
    if not input_asdf:
        splits,locs,tdir = [None for _ in range(3)]
    else:
        splits,tdir = [None for _ in range(2)]

#-----broadcast the variables------
if input_asdf:
    splits = comm.bcast(splits,root=0)
    tdir = comm.bcast(tdir,root=0)
    extra = splits % size
else:
    splits = comm.bcast(splits,root=0)
    tdir = comm.bcast(tdir,root=0)
    locs = comm.bcast(locs,root=0)
    extra = splits % size

#--------MPI: loop through each station--------
for ista in range (rank,splits+size-extra,size):
    #time.sleep(rank/1000)

    if ista<splits:
        t10=time.time()

        if input_asdf:    
            #---------think about reading mini-seed files here---------
            with pyasdf.ASDFDataSet(tdir[ista],mode='r') as ds:
                temp = ds.waveforms.list()
                station = temp[0].split('.')[1]
                network = temp[0].split('.')[0]

                if flag:
                    print("working on station %s " % station)

                #------get traces and station inventory------
                inv1 = ds.waveforms[temp[0]]['StationXML']
                
                #----to use dowonloaded inventory----
                if use_resp_dir:
                    inv_file = glob.glob(os.path.join(resp_dir,network+'.'+station+'.xml'))
                    if not os.path.isfile(inv_file):
                        raise IOError('cannot find inv files %s' % (inv_file))
                    inv1 = obspy.read_inventory(inv_file[0],format='StationXML')

                if (not inv1[0][0].latitude) or (not inv1[0][0].longitude):
                    raise ValueError('no station information in inventory! double check!')

                #---------construct a pd structure for fft_parameter functions later----------
                locs = pd.DataFrame([[inv1[0][0].latitude,inv1[0][0].longitude,inv1[0][0].elevation]],\
                    columns=['latitude','longitude','elevation'])
                
                #------get day information: works better than just list the tags------
                all_tags = ds.waveforms[temp[0]].get_waveform_tags()

                #-----continue if there is no data-----
                if len(all_tags)==0:
                    continue

                #----loop through each stream----
                for itag in range(len(all_tags)):
                                        
                    if flag:
                        print("working on trace " + all_tags[itag])

                    source = ds.waveforms[temp[0]][all_tags[itag]]
                    comp = source[0].stats.channel
                    
                    if prepro:
                        if all_tags[itag].split('_')[0] != 'raw':
                            #raise ValueError('it appears pre-processing has been performed!')
                            print('warning! it appears pre-processing has been performed!')

                        t0=time.time()
                        source = noise_module.preprocess_raw(source,inv1,downsamp_freq,checkt,pre_filt,resp,resp_dir)
                        t1=time.time()
                        if flag:
                            print("prepro takes %f s" % (t1-t0))
                    
                    #----------variables to define days with earthquakes----------
                    all_madS = noise_module.mad(source[0].data)
                    all_stdS = np.std(source[0].data)
                    if all_madS==0 or all_stdS==0 or np.isnan(all_madS) or np.isnan(all_stdS):
                        print("continue! madS or stdS equeals to 0 for %s" % source)
                        continue

                    trace_madS = []
                    trace_stdS = []
                    nonzeroS = []
                    nptsS = []
                    source_slice = obspy.Stream()

                    #--------break a continous recording into pieces----------
                    t0=time.time()
                    for ii,win in enumerate(source[0].slide(window_length=cc_len, step=step)):
                        win.detrend(type="constant")
                        win.detrend(type="linear")
                        trace_madS.append(np.max(np.abs(win.data))/all_madS)
                        trace_stdS.append(np.max(np.abs(win.data))/all_stdS)
                        nonzeroS.append(np.count_nonzero(win.data)/win.stats.npts)
                        nptsS.append(win.stats.npts)
                        win.taper(max_percentage=0.05,max_length=20)
                        source_slice.append(win)
                    del source
                    
                    t1=time.time()
                    if flag:
                        print("breaking records takes %f s"%(t1-t0))

                    if len(source_slice) == 0:
                        print("No traces for %s " % source)
                        continue

                    source_params= np.vstack([trace_madS,trace_stdS,nonzeroS]).T
                    del trace_madS, trace_stdS, nonzeroS

                    #---------seems un-necesary for data already pre-processed with same length (zero-padding)-------
                    N = len(source_slice)
                    NtS = np.max(nptsS)
                    dataS_t= np.zeros(shape=(N,2))
                    dataS = np.zeros(shape=(N,NtS),dtype=np.float32)
                    for ii,trace in enumerate(source_slice):
                        dataS_t[ii,0]= source_slice[ii].stats.starttime-obspy.UTCDateTime(1970,1,1)# convert to dataframe
                        dataS_t[ii,1]= source_slice[ii].stats.endtime -obspy.UTCDateTime(1970,1,1)# convert to dataframe
                        dataS[ii,0:nptsS[ii]] = trace.data
                        if ii==0:
                            dataS_stats=trace.stats

                    #------check the dimension of the dataS-------
                    if dataS.ndim == 1:
                        axis = 0
                    elif dataS.ndim == 2:
                        axis = 1

                    Nfft = int(next_fast_len(int(dataS.shape[axis])))

                    #------to normalize in time or not------
                    if time_norm:
                        t0=time.time()   

                        if norm_type == 'one_bit': 
                            white = np.sign(dataS)
                        elif norm_type == 'running_mean':
                            white = noise_module.moving_ave(dataS,int(1/freqmin/2))

                        t1=time.time()
                        if flag:
                            print("temporal normalization takes %f s"%(t1-t0))

                    #-----to whiten or not------
                    if to_whiten:
                        t0=time.time()
                        source_white = noise_module.whiten(white,dt,freqmin,freqmax)
                        t1=time.time()
                        if flag:
                            print("spectral whitening takes %f s"%(t1-t0))
                    else:
                        source_white = scipy.fftpack.fft(white, Nfft, axis=axis)

                    #-------------save FFTs as HDF5 files-----------------
                    crap=np.zeros(shape=(N,Nfft//2),dtype=np.complex64)
                    fft_h5 = os.path.join(FFTDIR,network+'.'+station+'.h5')

                    if not os.path.isfile(fft_h5):
                        with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                            pass # create pyasdf file 
            
                    with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                        parameters = noise_module.fft_parameters(dt,cc_len,source_params, \
                            locs.iloc[0],comp,Nfft,N)
                        
                        savedate = '{0:04d}_{1:02d}_{2:02d}'.format(dataS_stats.starttime.year,\
                            dataS_stats.starttime.month,dataS_stats.starttime.day)
                        path = savedate

                        data_type = str(comp)
                        if itag==0:
                            fft_ds.add_stationxml(inv1)
                        crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                        fft_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                    del fft_ds, crap, parameters, source_slice, source_white, dataS, dataS_stats, dataS_t, source_params     

        else:

            #----loop through each day on each core----
            for jj in range (len(tdir)):
                station = locs.iloc[ista]['station']
                network = locs.iloc[ista]['network']
                if flag:
                    print("working on station %s " % station)
                
                #-----------SAC and MiniSeed both works here-----------
                tfiles = glob.glob(os.path.join(tdir[jj],'*'+station+'*'))
                if len(tfiles)==0:
                    print(str(station)+' does not have sac file at '+str(tdir[jj]))
                    continue

                if method != 'raw':
                    #----loop through each channel----
                    for tfile in tfiles:

                        #--------------------------------------------------------------
                        #---what if this station has several segments in that day------
                        #--------------------------------------------------------------
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
                            source = noise_module.preprocess_raw(source,inv1,downsamp_freq,checkt,pre_filt,resp,resp_dir)
                            if len(source)==0:
                                continue
                            t1=time.time()
                            if flag:
                                print("prepro takes %f s" % (t1-t0))

                        #----------variables to define days with earthquakes----------
                        all_madS = noise_module.mad(source[0].data)
                        all_stdS = np.std(source[0].data)
                        if all_madS==0 or all_stdS==0:
                            print("continue! madS or stdS equeals to 0 for %s" %tfile)
                            continue

                        trace_madS = []
                        trace_stdS = []
                        nonzeroS = []
                        nptsS = []
                        source_slice = obspy.Stream()

                        #--------break a continous recording into pieces----------
                        t0=time.time()
                        for ii,win in enumerate(source[0].slide(window_length=cc_len, step=step)):
                            win.detrend(type="constant")
                            win.detrend(type="linear")
                            trace_madS.append(np.max(np.abs(win.data))/all_madS)
                            trace_stdS.append(np.max(np.abs(win.data))/all_stdS)
                            nonzeroS.append(np.count_nonzero(win.data)/win.stats.npts)
                            nptsS.append(win.stats.npts)
                            win.taper(max_percentage=0.05,max_length=20)
                            source_slice.append(win)
                        del source
                        t1=time.time()
                        if flag:
                            print("breaking records takes %f s"%(t1-t0))

                        if len(source_slice) == 0:
                            print("No traces for %s " % tfile)
                            continue

                        source_params= np.vstack([trace_madS,trace_stdS,nonzeroS]).T

                        #---------seems un-necesary for data already pre-processed with same length (zero-padding)-------
                        N = len(source_slice)
                        NtS = np.max(nptsS)
                        dataS = np.zeros(shape=(N,NtS),dtype=np.float32)
                        for ii,trace in enumerate(source_slice):
                            dataS[ii,0:nptsS[ii]] = trace.data
                            if ii==0:
                                dataS_stats=trace.stats

                        #---2 parameters----
                        axis = dataS.ndim-1
                        Nfft = int(next_fast_len(int(dataS.shape[axis])))

                        #------to normalize in time or not------
                        if time_norm:
                            t0=time.time()   

                            if norm_type == 'one_bit': 
                                white = np.sign(dataS)
                            elif norm_type == 'running_mean':
                                
                                #--------convert to 1D array for smoothing in time-domain---------
                                white = np.zeros(shape=dataS.shape,dtype=dataS.dtype)
                                for kkk in range(N):
                                    white[kkk,:] = dataS[kkk,:]/noise_module.moving_ave(np.abs(dataS[kkk,:]),smooth_N)

                            t1=time.time()
                            if flag:
                                print("temporal normalization takes %f s"%(t1-t0))
                        else:
                            white = dataS

                        #-----to whiten or not------
                        if to_whiten:
                            t0=time.time()

                            if whiten_type == 'one_bit':
                                source_white = noise_module.whiten(white,dt,freqmin,freqmax)
                            elif whiten_type == 'running_mean':
                                source_white = noise_module.whiten_smooth(white,dt,freqmin,freqmax,smooth_N)
                            t1=time.time()
                            if flag:
                                print("spectral whitening takes %f s"%(t1-t0))
                        else:
                            source_white = scipy.fftpack.fft(white, Nfft, axis=axis)

                        '''
                        temp = scipy.fftpack.ifft(source_white[1,:],Nfft)[:NtS]
                        plt.subplot(211)
                        plt.plot(dataS[1,:]/max(dataS[1,:]),'r-')
                        plt.plot(white[1,:]/max(white[1,:]),'b-')
                        plt.plot(temp/max(temp),'g-')
                        plt.subplot(212)
                        temp = scipy.fftpack.fft(white,Nfft,axis=axis)
                        freqVec = scipy.fftpack.fftfreq(Nfft, d=dt)[:Nfft // 2]
                        plt.plot(freqVec,np.abs(temp[1,:Nfft//2])/max(np.abs(temp[1,:Nfft//2])),'r-')
                        plt.plot(freqVec,np.abs(source_white[1,:Nfft//2])/max(np.abs(source_white[1,:Nfft//2])),'b-')
                        plt.show()
                        '''

                        #-------------save FFTs as ASDF files-----------------
                        crap=np.zeros(shape=(N,Nfft//2),dtype=np.complex64)
                        fft_h5 = os.path.join(FFTDIR,network+'.'+station+'.h5')

                        if not os.path.isfile(fft_h5):
                            with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                                pass # create pyasdf file 
                
                        with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                            parameters = noise_module.fft_parameters(dt,cc_len,source_params, \
                                locs.iloc[ista],comp,Nfft,N)
                            
                            savedate = '{0:04d}_{1:02d}_{2:02d}'.format(dataS_stats.starttime.year,\
                                dataS_stats.starttime.month,dataS_stats.starttime.day)
                            path = savedate

                            data_type = str(comp)
                            fft_ds.add_stationxml(inv1)
                            crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                            fft_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                        del fft_ds, crap, parameters, source_slice, source_white, dataS, dataS_stats, source_params 
        
        t11=time.time()
        print('it takes '+str(t11-t10)+' s to process one station in step 1')

t01=time.time()
print('step1 takes '+str(t01-t00)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
