import os
import sys
import glob
import numpy as np
import scipy
import obspy
import matplotlib.pyplot as plt
import noise_module
import time
import pyasdf
import pandas as pd
from mpi4py import MPI


'''
this script uses the day as the outmost loop and then computes the cross-correlations between each station-pair at 
that day for overlapping time window.

implemented with MPI (Nov.09.2018)

this optimized version runs 5 times faster than the previous one by 1) pulling the prcess of making smoothed spectrum
of the source outside of the receiver loop, 2) take advantage the linear relationship of ifft to average the spectrum
first before doing ifft in cross-correlaiton functions and 3) sacrifice the disk memory (by 1.5 times) to improve the 
I/O speed (by 4 times)  (Jan,28,2019)

updates include 1) remove the input file of station list by looping through all stations according to H5 file list, 
2) make use of the inventory information for lon,lat instead of from the station list, 3) add new parameters to the 
output H5 files for later CC steps and 4) make same type of data_types and paths list (Feb,15,2019)
'''

ttt0=time.time()
#------some useful absolute paths-------
#FFTDIR = '/n/flashlfs/mdenolle/KANTO/DATA/FFT_v2'
#CCFDIR = '/n/flashlfs/mdenolle/KANTO/DATA/CCF_v2'
#STACKDIR = '/n/flashlfs/mdenolle/KANTO/DATA/STACK'
#locations = '/n/home13/chengxin/cases/KANTO/locations_small.txt'

FFTDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT'
CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF'

#-----some control parameters------
flag=False              #output intermediate variables and computing times
smooth_N=10             #window length for smoothing the spectrum amplitude
downsamp_freq=20
dt=1/downsamp_freq
cc_len=3600
step=1800
maxlag=1800
method='deconv'
start_date = '2010_01_01'
end_date   = '2010_02_29'

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------


#-------form a station pair to loop through-------
if rank ==0:
    sfiles = sorted(glob.glob(os.path.join(FFTDIR,'*.h5')))
    day = noise_module.get_event_list(start_date,end_date)
    splits = len(day)
else:
    splits,sfiles,day = [None for _ in range(3)]

#------split the common variables------
splits = comm.bcast(splits,root=0)
day    = comm.bcast(day,root=0)
sfiles   = comm.bcast(sfiles,root=0)
extra  = splits % size

for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        iday = day[ii]

        t10 = time.time()
        #------loop I of each source-----
        for isource in range(len(sfiles)-1):
            source = sfiles[isource]
            staS = source.split('/')[-1].split('.')[1]
            netS = source.split('/')[-1].split('.')[0]
            if flag:
                print('source: %s %s' % (staS,netS))

            with pyasdf.ASDFDataSet(source, mpi=False, mode='r') as fft_ds_s:

                #-------get lon and lat information from inventory--------
                temp = fft_ds_s.waveforms.list()
                invS = fft_ds_s.waveforms[temp[0]]['StationXML']
                lonS = invS[0][0].longitude
                latS = invS[0][0].latitude
                if flag:
                    print('source coordinates: %8.2f %8.2f' % (lonS,latS))

                #----loop II of each component for source A------
                data_types_s = fft_ds_s.auxiliary_data.list()
                
                for icompS in range(len(data_types_s)):
                    if flag:
                        print("reading source %s for day %s" % (staS,icompS))
                    data_type_s = data_types_s[icompS]
                    path_list_s = fft_ds_s.auxiliary_data[data_type_s].list()

                    #--------iday exists for source A------
                    if iday in path_list_s:
                        paths = iday

                        t1=time.time()
                        #-----------get the parameter of Nfft-----------
                        Nfft = fft_ds_s.auxiliary_data[data_type_s][paths].parameters['nfft']
                        Nseg = fft_ds_s.auxiliary_data[data_type_s][paths].parameters['nseg']
                        
                        fft1 = fft_ds_s.auxiliary_data[data_type_s][paths].data[:,:Nfft//2] 
                        source_std = fft_ds_s.auxiliary_data[data_type_s][paths].parameters['std']
                        
                        t2=time.time()
                        #-----------get the smoothed source spectrum for decon later----------
                        if method == 'deconv':
                            temp = noise_module.moving_ave(np.abs(fft1.reshape(fft1.size,)),smooth_N)
                            sfft1 = np.conj(fft1.reshape(fft1.size,))/temp**2
                        elif method == 'coherence':
                            temp = noise_module.moving_ave(np.abs(fft1.reshape(fft1.size,)),smooth_N)
                            sfft1 = np.conj(fft1.reshape(fft1.size,))/temp
                        elif method == 'raw':
                            sfft1 = fft1
                        sfft1 = sfft1.reshape(Nseg,Nfft//2)

                        t3=time.time()
                        if flag:
                            print('read S %6.4fs, smooth %6.4fs' % ((t2-t1), (t3-t2)))

                        #-----------now loop III of each receiver B----------
                        for ireceiver in range(isource,len(sfiles)):
                            receiver = sfiles[ireceiver]
                            staR = receiver.split('/')[-1].split('.')[1]
                            netR = receiver.split('/')[-1].split('.')[0]
                            if flag:
                                print('receiver: %s %s' % (staR,netR))
                            
                            with pyasdf.ASDFDataSet(receiver, mpi=False, mode='r') as fft_ds_r:

                                #-------get lon and lat information from inventory--------
                                temp = fft_ds_r.waveforms.list()
                                invR = fft_ds_r.waveforms[temp[0]]['StationXML']
                                lonR = invR[0][0].longitude
                                latR = invR[0][0].latitude
                                if flag:
                                    print('receiver coordinates: %8.2f %8.2f' % (lonR,latR))

                                #-----loop IV of each component for receiver B------
                                data_types_r = fft_ds_r.auxiliary_data.list()

                                for icompR in range(len(data_types_r)):
                                    data_type_r = data_types_r[icompR]
                                    path_list_r = fft_ds_r.auxiliary_data[data_type_r].list()

                                    #----if that day exists for receiver B----
                                    if iday in path_list_r:
                                        pathr = iday

                                        t4=time.time()
                                        fft2= fft_ds_r.auxiliary_data[data_type_r][pathr].data[:,:Nfft//2]
                                        receiver_std = fft_ds_r.auxiliary_data[data_type_r][pathr].parameters['std']
                                        t5=time.time()

                                        #---------- check the existence of earthquakes ----------
                                        rec_ind = np.where(receiver_std < 10)[0]
                                        sou_ind = np.where(source_std < 10)[0]

                                        #-----note that Hi-net has a few mi-secs differences to Mesonet in terms starting time-----
                                        bb,indx1,indx2=np.intersect1d(sou_ind,rec_ind,return_indices=True)
                                        indx1=sou_ind[indx1]
                                        indx2=rec_ind[indx2]
                                        if (len(indx1)==0) | (len(indx2)==0):
                                            continue

                                        t6=time.time()
                                        corr=noise_module.optimized_correlate1(sfft1[indx1,:],fft2[indx2,:],\
                                                np.round(maxlag),dt,Nfft,len(indx1),method)
                                        t7=time.time()

                                        #---------------keep daily cross-correlation into a hdf5 file--------------
                                        cc_aday_h5 = os.path.join(CCFDIR,iday+'.h5')
                                        crap   = np.zeros(corr.shape)

                                        if not os.path.isfile(cc_aday_h5):
                                            with pyasdf.ASDFDataSet(cc_aday_h5,mpi=False) as ds:
                                                pass 

                                        with pyasdf.ASDFDataSet(cc_aday_h5,mpi=False) as ccf_ds:
                                            parameters = noise_module.optimized_cc_parameters(dt,maxlag,str(method),lonS,latS,lonR,latR)

                                            #------save the time domain cross-correlation functions-----
                                            path = netR+'s'+staR+'s'+data_type_r
                                            new_data_type = netS+'s'+staS+'s'+data_type_s
                                            crap = corr
                                            ccf_ds.add_auxiliary_data(data=crap, data_type=new_data_type, path=path, parameters=parameters)

                                        t8=time.time()
                                        if flag:
                                            print('read R %6.4fs, cc %6.4fs, write cc %6.4fs'% ((t5-t4),(t7-t6),(t8-t7)))

        t11 = time.time()
        print('it takes %6.4fs to process one day in step 2' % (t11-t10))


ttt1=time.time()
print('all step 2 takes '+str(ttt1-ttt0))

comm.barrier()
if rank == 0:
    sys.exit()
