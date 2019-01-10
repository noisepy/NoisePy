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
from obspy.io.sac.sactrace import SACTrace

from mpi4py import MPI


'''
this script reads from the h5 file for each station (containing all pre-processed and fft-ed traces) and then
computes the cross-correlations between each station-pair at an overlapping time window.

this version is implemented with MPI (Nov.09.2018)
'''

t0=time.time()

#------some useful absolute paths-------
#FFTDIR = '/n/flashlfs/mdenolle/KANTO/DATA/FFT'
#STACKDIR = '/n/flashlfs/mdenolle/KANTO/DATA/STACK_deconv'
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'

FFTDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT'
STACKDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF'
locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations.txt'
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
    locs = pd.read_csv(locations)
    sta  = list(locs.iloc[:]['station'])
    splits = len(pairs)
else:
    splits,pairs,locs,sta = [None for _ in range(4)]

#------split the common variables------
splits = comm.bcast(splits,root=0)
pairs  = comm.bcast(pairs,root=0)
locs   = comm.bcast(locs,root=0)
sta    = comm.bcast(sta,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        source,receiver = pairs[ii][0],pairs[ii][1]
        print('source '+source.split('/')[-1]+' receiver '+receiver.split('/')[-1]+' rank '+str(rank))

        fft_h5   = source
        fft_ds_s = pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r')
        fft_h5   = receiver
        fft_ds_r = pyasdf.ASDFDataSet(fft_h5, mpi=False, mode='r')
        
        #-------get source information------
        net_sta_s = fft_ds_s.waveforms.list()[0]
        staS = net_sta_s.split('.')[1]
        netS = net_sta_s.split('.')[0]

        tindx = sta.index(staS)
        slat  = locs.iloc[tindx]['latitude']
        slon  = locs.iloc[tindx]['longitude']
        path_list_s = fft_ds_s.auxiliary_data[data_type].list()

        #------get receiver information--------
        net_sta_r = fft_ds_r.waveforms.list()[0]
        staR = net_sta_r.split('.')[1]
        netR = net_sta_r.split('.')[0]

        tindx = sta.index(staR)
        rlat  = locs.iloc[tindx]['latitude']
        rlon  = locs.iloc[tindx]['longitude']
        path_list_r = fft_ds_r.auxiliary_data[data_type].list()

        #---------initialize index and stacking arrays for saving sac files---------
        indx = np.zeros(shape=(len(tcomp),len(tcomp)),dtype=np.int16)       # indx contains the comp information for each station pair
        tlen = int((2*maxlag)/dt+1)
        ncorr = np.zeros(shape=(indx.size,tlen),dtype=np.float32)
        #pcorr = np.zeros(shape=(indx.size,tlen),dtype=np.float32)

        #---------loop through each component of the source------
        for jj in range(len(path_list_s)):

            paths = path_list_s[jj]
            compS = fft_ds_s.auxiliary_data[data_type][paths].parameters['component']

            #-----------get the parameter of Nfft-----------
            #t0=time.time()
            Nfft = fft_ds_s.auxiliary_data[data_type][paths].parameters['nfft']
            Nseg = fft_ds_s.auxiliary_data[data_type][paths].parameters['nseg']
            
            dataS_t = []
            fft1 = np.zeros(shape=(Nseg,Nfft//2-1),dtype=np.complex64)
            fft1= fft_ds_s.auxiliary_data[data_type][paths].data[:,:Nfft//2-1]
            
            source_std = fft_ds_s.auxiliary_data[data_type][paths].parameters['std']
            #date =fft_ds_s.auxiliary_data[data_type][paths].parameters['starttime'] 
            #dataS_t=np.array(pd.to_datetime([datetime.utcfromtimestamp(s) for s in date]))
            #del date

            #t1=time.time()
            #print('reading source takes '+str(t1-t0)+' s')

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
                    fft2 = np.zeros(shape=(Nseg,Nfft//2-1),dtype=np.complex64)           
                    fft2= fft_ds_r.auxiliary_data[data_type][pathr].data[:,:Nfft//2-1]

                    receiver_std = fft_ds_r.auxiliary_data[data_type][pathr].parameters['std']
                    
                    #date =fft_ds_r.auxiliary_data[data_type][pathr].parameters['starttime'] 
                    #dataR_t=np.array(pd.to_datetime([datetime.utcfromtimestamp(s) for s in date]))
                    #del date

                    #---------- check the existence of earthquakes ----------
                    rec_ind = np.where(receiver_std < 10)[0]
                    sou_ind = np.where(source_std < 10)[0]

                    #-----note that Hi-net has a few mi-secs differences to Mesonet in terms starting time-----
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
                    #print('cross correlations take '+str(t1-t0)+' s')

                    #--------find the index to store data--------
                    indx[tcomp.index(compS)][tcomp.index(compR)]=1
                    nindx=tcomp.index(compS)*len(tcomp)+tcomp.index(compR)

                    #--------linear stackings---------
                    if corr.ndim==2:
                        y = np.mean(corr,axis=0)
                        #y2 = noise_module.butter_pass(y,freqmin,freqmax,dt,2)
                    elif corr.ndim==1:
                        y = corr
                    else:
                        continue
                    ncorr[nindx][:] = ncorr[nindx][:] + y

                    #-------pws stacking--------
                    #y2 = noise_module.pws(np.array(corr),2,1/dt,5.)
                    #y4 = noise_module.butter_pass(y2,freqmin,freqmax,dt,2)
                    #pcorr[nindx][:] = pcorr[nindx][:] + y2
 

        #-------ready to write into files---------
        indx_array = np.where(indx==1)

        #-------first dim is for source and second for receiver-----
        for jj in range(len(indx_array[0])):
            compS = tcomp[indx_array[0][jj]]
            compR = tcomp[indx_array[1][jj]]
            dir1 = '.'.join([netS,staS])
            dir2 = '_'.join([netS,staS,netR,staR,compS,compR])
            nindx = indx_array[0][jj]*len(tcomp)+indx_array[1][jj]

            #------------make sure the directory exists----------
            if os.path.exists(os.path.join(STACKDIR,dir1))==False:
                os.mkdir(os.path.join(STACKDIR,dir1))    

            #--------save two stacked traces as SAC files---------
            filename1 = os.path.join(STACKDIR,dir1,dir2+'_ls.SAC')
            #filename2 = os.path.join(STACKDIR,dir1,dir2+'_pws.SAC')

            sac1 = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,
                            delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=ncorr[nindx][:])
            #sac2 = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,
            #                delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=pcorr[nindx][:])

            sac1.write(filename1,byteorder='big')
            #sac2.write(filename2,byteorder='big')

        #del pcorr
        del fft_ds_s, fft_ds_r, path_list_r, path_list_s, fft1, fft2, dataS_t, dataR_t, source_std, receiver_std, ncorr

t1=time.time()
print('step2 takes '+str(t1-t0)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
