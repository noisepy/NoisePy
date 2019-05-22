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
import h5py
import pandas as pd
from mpi4py import MPI

'''
this script pre-processs the noise data for each single station using the parameters given below 
and stored the whitened and nomalized fft trace for each station in a HDF5 file as *.h5.

implemented with MPI (Nov.09.2018)
by C.Jiang, T.Clements, M.Denolle

this script only handles SAC and MiniSeed formate inputs. If you have ASDF file, use the other
script of S1_savASDF_v2.2
'''

t00=time.time()

rootpath  = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
FFTDIR = os.path.join(rootpath,'FFT/resp')
locations = os.path.join(rootpath,'station.lst')
event = os.path.join(rootpath,'noise_data/Event_2010_*')
#--------think about how to simplify this----------
respdir = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/instrument/resp_4types/resp_spect_20Hz'

#-----boolen parameters------
prepro   =True              #preprocess the data?
to_whiten=False              #whiten the spectrum?
time_norm=False              #normalize in time?
asdf     =True
hdf5     =False
flag     =False             #print intermediate variables and computing time

checkt  = True              #check for traces with points bewtween sample intervals
resp    = 'spectrum'    

#----more common variables---
pre_filt=[0.04,0.05,6,8]
downsamp_freq=20
dt=1/downsamp_freq
cc_len=3600
step=1800
freqmin=0.05  
freqmax=6
norm_type='one_bit'


#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    #-----check whether dir exist----
    if not os.path.isdir(FFTDIR):
        os.mkdir(FFTDIR)

    #----define common variables----
    locs = pd.read_csv(locations)
    nsta = len(locs)
    splits = nsta
    tdir = sorted(glob.glob(event))

    if len(tdir)==0 or nsta==0:
        raise IOError('no available files for doing FFT')
else:
    splits,locs,tdir = [None for _ in range(3)]

#-----broadcast the variables------
splits = comm.bcast(splits,root=0)
locs = comm.bcast(locs,root=0)
tdir = comm.bcast(tdir,root=0)
extra = splits % size

#--------MPI: loop through each station--------
for ista in range (rank,splits+size-extra,size):
    #time.sleep(rank/1000)

    if ista<splits:
        t10=time.time()
        #----loop through each day on each core----
        for jj in range (len(tdir)):
            station = locs.iloc[ista]['station']
            network = locs.iloc[ista]['network']
            if flag:
                print("working on station %s " % station)
            
            #---------think about reading mini-seed files here---------
            tfiles = glob.glob(os.path.join(tdir[jj],'*'+station+'*'))
            if len(tfiles)==0:
                print(str(station)+' does not have sac file at '+str(tdir[jj]))
                continue

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
                    source = noise_module.preprocess_raw(source,downsamp_freq,checkt,pre_filt,resp,respdir)
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

                #-----to whiten or not------
                if to_whiten:
                    t0=time.time()
                    source_white = noise_module.whiten(dataS,dt,freqmin,freqmax)
                    t1=time.time()
                    if flag:
                        print("spectral whitening takes %f s"%(t1-t0))
                else:
                    source_white = scipy.fftpack.fft(dataS, Nfft, axis=axis)

                #------to normalize in time or not------
                if time_norm:
                    t0=time.time()   
                    white = np.real(scipy.fftpack.ifft(source_white, Nfft, axis=axis)) #/ Nt

                    if norm_type == 'one_bit': 
                        white = np.sign(white)
                    elif norm_type == 'running_mean':
                        white = noise_module.moving_ave(white,int(1 / freqmin / 2))
                    source_white = scipy.fftpack.fft(white, Nfft, axis=axis)

                    t1=time.time()
                    if flag:
                        print("temporal normalization takes %f s"%(t1-t0))

                #-------------save FFTs as HDF5 files-----------------
                crap=np.zeros(shape=(N,Nfft//2),dtype=np.complex64)
                fft_h5 = os.path.join(FFTDIR,network+'.'+station+'.h5')

                if asdf:
                    if not os.path.isfile(fft_h5):
                        with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                            pass # create pyasdf file 
            
                    with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                        parameters = noise_module.fft_parameters(dt,cc_len,dataS_stats,dataS_t,source_params, \
                            locs.iloc[ista],comp,Nfft,N)
                        
                        savedate = '{0:04d}_{1:02d}_{2:02d}'.format(dataS_stats.starttime.year,\
                            dataS_stats.starttime.month,dataS_stats.starttime.day)
                        path = savedate

                        data_type = str(comp)
                        fft_ds.add_stationxml(inv1)
                        crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                        fft_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                    del fft_ds, crap, parameters, source_slice, source_white, dataS, dataS_stats, dataS_t, source_params   
                        
                elif hdf5:
                    if not os.path.isfile(fft_h5):
                        with h5py.File(fft_h5,"w") as fft_ds:
                            pass # create pyasdf file 
            
                    with h5py.File(fft_h5,"a") as fft_ds:
                        parameters = noise_module.fft_parameters(dt,cc_len,dataS_stats,dataS_t,source_params, \
                            locs.iloc[ista],comp,Nfft,N)
                        
                        savedate = '{0:04d}_{1:02d}_{2:02d}'.format(dataS_stats.starttime.year,\
                            dataS_stats.starttime.month,dataS_stats.starttime.day)
                        path = '_'.join(['fft', network, station, comp, savedate])

                        crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                        tmp=fft_ds.create_dataset(path+".real",data=np.real(crap),shape=np.shape(crap),dtype=np.float32)
                        for key in parameters.keys():
                            tmp.attrs[key]=parameters[key]
                            #print("key",key,":",parameters[key])
                        tmp=fft_ds.create_dataset(path+".imag",data=np.imag(crap),shape=np.shape(crap),dtype=np.float32)

                    del fft_ds, crap, parameters, source_slice, source_white, dataS, dataS_stats, dataS_t, source_params                 

        t11=time.time()
        print('it takes '+str(t11-t10)+' s to process %s in step 1' % station)

t01=time.time()
print('step1 takes '+str(t01-t00)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
