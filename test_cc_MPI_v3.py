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
import itertools
from obspy.clients.fdsn import Client

from mpi4py import MPI


'''
this script reads from the h5 file for each station (containing all pre-processed and fft-ed traces) and then
computes the cross-correlations between each station-pair at an overlapping time window.

this version is implemented with MPI (Nov.09.2018)
'''


#------some useful absolute paths-------
FFTDIR = '/n/flashlfs/mdenolle/KANTO/DATA/FFT/no_norm'
CCFDIR = '/n/flashlfs/mdenolle/KANTO/DATA/CCF_deconv'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'

#FFTDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT1'
#CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF3'
tcomp  = ['EHZ','EHE','EHN','HNU','HNE','HNN']


#-----some control parameters------
data_type = 'FFT'
downsamp_freq=20
dt=1/downsamp_freq
freqmin=0.05
freqmax=4
cc_len=3600
step=1800
maxlag=800
method='deconv'
#method='raw'
#method='coherence'


#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------


#-------form a station pair to loop through-------
if rank ==0:
    tfiles = glob.glob(os.path.join(FFTDIR,'*.h5'))
    tfiles = sorted(tfiles)
    pairs  = list(itertools.combinations(tfiles,2))
    splits = len(pairs)
else:
    splits,pairs = [None for _ in range(2)]

#------split the common variables------
splits = comm.bcast(splits,root=0)
pairs  = comm.bcast(pairs,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        source,receiver = pairs[ii][0],pairs[ii][1]
        print('source is ' + source.split('/')[-1] + ' receiver is ' + receiver.split('/')[-1] + ' rank is ' + str(rank))

        fft_h5 = source
        fft_ds_s = pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r')
        fft_h5 = receiver
        fft_ds_r = pyasdf.ASDFDataSet(fft_h5, mpi=False, mode='r')
        
        #-------get common source information------
        net_sta_s = fft_ds_s.waveforms.list()[0]
        staS = net_sta_s.split('.')[1]
        netS = net_sta_s.split('.')[0]
        path_list_s = fft_ds_s.auxiliary_data[data_type].list()

        #------get common receiver information--------
        net_sta_r = fft_ds_r.waveforms.list()[0]
        staR = net_sta_r.split('.')[1]
        netR = net_sta_r.split('.')[0]
        path_list_r = fft_ds_r.auxiliary_data[data_type].list()


        #---------loop through each component of the source------
        for jj in range(len(path_list_s)):
            #print('begin source ' + path_list_s[jj])

            paths = path_list_s[jj]
            compS = fft_ds_s.auxiliary_data[data_type][paths].parameters['component']

            #-----------get the parameter of Nfft-----------
            Nfft = len(np.array(fft_ds_s.auxiliary_data[data_type][path_list_s[0]].data[0,:]))
            dataS_t = []
                
            fft1= np.add(np.array(fft_ds_s.auxiliary_data[data_type][paths].data[:,:Nfft//2-1]) \
                    , 1j* np.array(fft_ds_s.auxiliary_data[data_type][paths].data[:,Nfft//2:Nfft-1]))
            source_std = fft_ds_s.auxiliary_data[data_type][paths].parameters['std']
            date =fft_ds_s.auxiliary_data[data_type][paths].parameters['starttime'] 
            dataS_t=np.array(pd.to_datetime([datetime.utcfromtimestamp(s) for s in date]))
            del date

            #-------day information------
            tday  = paths[-10:]

            for compR in tcomp:
                #------ find the corresponding path for the receiver for that day ----------
                tpath = '_'.join(['fft',netR,staR,compR,tday])

                #-------if it exists-------        
                if tpath in path_list_r:
                    pathr = tpath
                    print(str(pathr))
                    dataR_t = []
                                
                    fft2=np.add(np.array(fft_ds_r.auxiliary_data[data_type][pathr].data[:,:Nfft//2-1]) \
                            , 1j* np.array(fft_ds_r.auxiliary_data[data_type][pathr].data[:,Nfft//2:Nfft-1]))
                    sampling_rate = fft_ds_r.auxiliary_data[data_type][pathr].parameters['sampling_rate']
                    receiver_std = fft_ds_r.auxiliary_data[data_type][pathr].parameters['std']
                    receiver_mad = fft_ds_r.auxiliary_data[data_type][pathr].parameters['mad']
                    date =fft_ds_r.auxiliary_data[data_type][pathr].parameters['starttime'] 
                    dataR_t=np.array(pd.to_datetime([datetime.utcfromtimestamp(s) for s in date]))
                    del date


                    #---------- check the existence of earthquakes ----------
                    rec_ind = np.where(receiver_std < 20)[0]
                    sou_ind = np.where(source_std < 20)[0]

                    #-----note that Hi-net and Mesonet have different starting times-----
                    #bb,indx1,indx2=np.intersect1d(dataS_t[sou_ind],dataR_t[rec_ind],return_indices=True)
                    bb,indx1,indx2=np.intersect1d(sou_ind,rec_ind,return_indices=True)
                    indx1=sou_ind[indx1]
                    indx2=rec_ind[indx2]
                    if (len(indx1)==0) | (len(indx2)==0):
                        continue

                    #t0=time.time()
                    #-----------do daily cross-correlations now-----------
                    corr,tcorr=noise_module.correlate(fft1[indx1,:Nfft//2-1],fft2[indx2,:Nfft//2-1], \
                            np.round(maxlag),dt,Nfft,method)
                    #t1=time.time()
                    #print('corr takes ' + str(t1-t0) + " s")

                    #---------------keep the daily cross-correlation into a hdf5 file--------------
                    tsource = os.path.join(CCFDIR,netS+"."+staS)
                    if os.path.exists(tsource)==False:
                        os.mkdir(tsource)
                    fft_h5 = os.path.join(tsource,netS +"." + staS + "." + netR + "." + staR + '.h5')
                    crap   = np.zeros(corr.shape)
                    if not os.path.isfile(fft_h5):
                        with pyasdf.ASDFDataSet(fft_h5,mpi=False) as ds:
                            pass 
                    else:
                        print([netS+"."+staS+"."+netR+"."+staR+'.h5', 'Already exists',obspy.UTCDateTime()])

                    with pyasdf.ASDFDataSet(fft_h5,mpi=False) as ccf_ds:
                        parameters = {'dt':dt, 'maxlag':maxlag, 'starttime':str(dataR_t[rec_ind]), 'method':str(method)}

                        #------save the time domain cross-correlation functions-----
                        path = '_'.join(['ccfs',str(method),netS,staS,netR,staR,compS,compR,tday])
                        new_data_type = compS+compR
                        crap = np.squeeze(corr)
                        ccf_ds.add_auxiliary_data(data=crap, data_type=new_data_type, path=path, parameters=parameters)

                        #------save the freq domain cross-correlation functions for future C3-----
                        #path = '_'.join(['ccfs.f',str(method),netS,staS,netR,staR,compS,compR,tday])
                        #crap = np.squeeze(scorr)
                        #ccf_ds.add_auxiliary_data(data=crap, data_type=new_data_type, path=path, parameters=parameters)

                    del ccf_ds, crap, parameters, path, fft2, dataR_t, receiver_std, receiver_mad,corr,tcorr
                    #t2=time.time()
                    #print('saving data took ' + str(t2-t1) + " s")


        del fft_ds_s, fft_ds_r, path_list_r, path_list_s, fft1, dataS_t, source_std

comm.barrier()
if rank == 0:
    sys.exit()
