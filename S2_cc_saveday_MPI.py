import os
import gc
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

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
this script loop through the days by using MPI and compute cross-correlation functions for each station-pair at that
day when there are overlapping time windows. (Nov.09.2018)

optimized to run ~5 times faster by 1) making smoothed spectrum of the source outside of the receiver loop; 2) taking 
advantage of the linearality of ifft to average the spectrum first before doing ifft in cross-correlaiton functions, 
and 3) sacrifice storage (by 1.5 times) to improve the I/O speed (by 4 times). 
Thanks to Zhitu Ma for thoughtful discussions.  (Jan,28,2019)

new updates include 1) remove the need of input station.lst by listing available HDF5 files, 2) make use of the inventory
for lon, lat information, 3) add new parameters to HDF5 files needed for later CC steps and 4) make data_types and paths
in the same format (Feb.15.2019). 

add the functionality of auto-correlations (Feb.22.2019). Note that the auto-cc is normalizing each station to its Z comp.

modify the structure of ASDF files to make it more flexable for later stacking and matrix rotation (Mar.06.2019)

updated to allow breaking the FFT of all stations into several segments and load them one segment by one segment at one in
the right beginning, so that 1) the required memory for dealing with large number of station-pairs can be minimized and 
2) it saves the time of repeatly reading the FFT for each station-pair (Mar.20.2019)
'''

ttt0=time.time()

rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/pre_processing'
FFTDIR = os.path.join(rootpath,'FFT')
CCFDIR = os.path.join(rootpath,'CCF')

#-----some control parameters------
flag=False               #output intermediate variables and computing times
#auto_corr=False         #include single-station auto-correlations or not
smooth_N=10             #window length for smoothing the spectrum amplitude
num_seg=1
downsamp_freq=20
dt=1/downsamp_freq
cc_len=3600
step=1800
maxlag=500              #enlarge this number if to do C3
method='coherence'
start_date = '2010_12_06'
end_date   = '2010_12_25'
inc_days   = 1

#if auto_corr and method=='coherence':
#    raise ValueError('Please set method to decon: coherence cannot be applied when auto_corr is wanted!')

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

#-------form a station pair to loop through-------
if rank ==0:
    if not os.path.isdir(CCFDIR):
        os.mkdir(CCFDIR)

    sfiles = sorted(glob.glob(os.path.join(FFTDIR,'*.h5')))
    day = noise_module.get_event_list(start_date,end_date,inc_days)
    splits = len(day)

    if not sfiles:
        raise IOError('Abort! No FFT data in %s' % FFTDIR)
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

        tt0=time.time()
        #-----------get parameters of Nfft and Nseg--------------
        with pyasdf.ASDFDataSet(sfiles[0],mpi=False,mode='r') as ds:
            data_types = ds.auxiliary_data.list()
            paths      = ds.auxiliary_data[data_types[0]].list()
            Nfft = ds.auxiliary_data[data_types[0]][paths[0]].parameters['nfft']
            Nseg = ds.auxiliary_data[data_types[0]][paths[0]].parameters['nseg']
            ncomp = len(data_types)
            nsta  = len(sfiles)
            ntrace = ncomp*nsta

        #----double check the ncomp parameters by opening a few stations------
        for jj in range(1,5):
            with pyasdf.ASDFDataSet(sfiles[jj],mpi=False,mode='r') as ds:
                data_types = ds.auxiliary_data.list()
                if len(data_types) > ncomp:
                    ncomp = len(data_types)
                    print('first station of %s misses other components' % (sfiles[0]))

        #--------record station information--------
        cc_coor = np.zeros((nsta,2),dtype=np.float32)
        sta = []
        net = []

        #----loop through each data segment-----
        nhours = int(np.ceil(Nseg/num_seg))
        for iseg in range(num_seg):
            
            #---index for the data chunck---
            sindx1 = iseg*nhours
            if iseg==num_seg-1:
                nhours = Nseg-iseg*nhours
            sindx2 = sindx1+nhours

            if nhours==0 or nhours <0:
                raise ValueError('nhours<=0, please double check')

            if flag:
                print('working on %dth segments of the daily FFT'% iseg)

            #-------make a crutial estimate on memory needed for the FFT of all stations: defaultly using float32--------
            memory_size = ntrace*Nfft/2*nhours*8/1024/1024/1024
            if memory_size > 8:
                raise MemoryError('Memory exceeds 8 GB! No enough memory to load them all once!')

            print('initialize the array ~%3.1f GB for storing all cc data' % (memory_size))

            #---------------initialize the array-------------------
            cc_array = np.zeros((ntrace,nhours*Nfft//2),dtype=np.complex64)
            cc_std   = np.zeros((ntrace,nhours),dtype=np.float32)
            cc_flag  = np.zeros((ntrace),dtype=np.int16)

            ttr0 = time.time()
            #-----loop through all stations------
            for ifile in range(len(sfiles)):
                tfile = sfiles[ifile]

                with pyasdf.ASDFDataSet(tfile,mpi=False,mode='r') as ds:
                    data_types = ds.auxiliary_data.list()

                    #-----load station informaiton here------
                    if iseg==0:
                        temp = ds.waveforms.list()
                        invS = ds.waveforms[temp[0]]['StationXML']
                        sta.append(temp[0].split('.')[1])
                        net.append(temp[0].split('.')[0])
                        cc_coor[ifile][0]=invS[0][0].longitude
                        cc_coor[ifile][1]=invS[0][0].latitude

                    if len(data_types) < ncomp:
                        
                        #-----check whether mising some components-----
                        for icomp in data_types:
                            if icomp[-1]=='E':
                                iindx = 0
                            elif icomp[-1]=='N':
                                iindx = 1
                            else:
                                iindx = 2
                            tpaths = ds.auxiliary_data[icomp].list()

                            if iday in tpaths:
                                if flag:
                                    print('find %dth data chunck for station %s day %s' % (iseg,tfile.split('/')[-1],iday))
                                indx = ifile*ncomp+iindx

                                #-----check bound----
                                if indx > ntrace:
                                    raise ValueError('index out of bound')
                                
                                dsize = ds.auxiliary_data[icomp][iday].data.size
                                if dsize == Nseg*Nfft//2:
                                    cc_flag[indx] = 1
                                    data  = ds.auxiliary_data[icomp][iday].data[sindx1:sindx2,:]
                                    cc_array[indx][:]= data.reshape(data.size)
                                    std   = ds.auxiliary_data[icomp][iday].parameters['std']
                                    cc_std[indx][:]  = std[sindx1:sindx2]
                    
                    else:

                        #-----E-N-U/Z orders when all components are available-----
                        for jj in range(len(data_types)):
                            icomp = data_types[jj]
                            tpaths = ds.auxiliary_data[icomp].list()
                            if iday in tpaths:
                                if flag:
                                    print('find %dth data chunck for station %s day %s' % (iseg,tfile.split('/')[-1],iday))
                                indx = ifile*ncomp+jj
                                
                                #-----check bound----
                                if indx > ntrace:
                                    raise ValueError('index out of bound')

                                dsize = ds.auxiliary_data[icomp][iday].data.size
                                if dsize == Nseg*Nfft//2:
                                    data  = ds.auxiliary_data[icomp][iday].data[sindx1:sindx2,:]
                                    cc_array[indx][:]= data.reshape(data.size)
                                    std   = ds.auxiliary_data[icomp][iday].parameters['std']
                                    cc_std[indx][:]  = std[sindx1:sindx2]
                                    cc_flag[indx] = 1

            ttr1 = time.time()
            print('loading all FFT takes %6.4fs' % (ttr1-ttr0))

            #-------loop I of each source------
            for isource in range(nsta-1):

                #---station info---
                staS = sta[isource]
                netS = net[isource]
                lonS = cc_coor[isource][0]
                latS = cc_coor[isource][1]

                if flag:
                    print('source: %s %s' % (staS,netS))

                #-----loop II of each component------
                for icompS in range(ncomp):
                    cc_indxS = isource*ncomp+icompS

                    #---no data for icomp---
                    if cc_flag[cc_indxS]==0:
                        #print('no data for %dth comp of %s' %(icompS,staS))
                        continue
                            
                    fft1 = cc_array[cc_indxS][:]
                    source_std = cc_std[cc_indxS][:]
                    sou_ind = np.where(source_std < 10)[0]
                    
                    t0=time.time()
                    #-----------get the smoothed source spectrum for decon later----------
                    if method == 'deconv':

                        #-----normalize single-station cc to z component-----
                        temp = noise_module.moving_ave(np.abs(fft1),smooth_N)

                        #--------think about how to avoid temp==0-----------
                        try:
                            sfft1 = np.conj(fft1)/temp**2
                        except ValueError:
                            raise ValueError('smoothed spectrum has zero values')

                    elif method == 'coherence':
                        temp = noise_module.moving_ave(np.abs(fft1),smooth_N)
                        try:
                            sfft1 = np.conj(fft1)/temp
                        except ValueError:
                            raise ValueError('smoothed spectrum has zero values')

                    elif method == 'raw':
                        sfft1 = np.conj(fft1)
                    
                    sfft1 = sfft1.reshape(nhours,Nfft//2)

                    t1=time.time()
                    if flag:
                        print('smooth %6.4fs' % (t1-t0))

                    #-----------now loop III for each receiver B----------
                    for ireceiver in range(isource+1,nsta):

                        #---station info---
                        staR = sta[ireceiver]
                        netR = net[ireceiver]
                        lonR = cc_coor[ireceiver][0]
                        latR = cc_coor[ireceiver][1]

                        if flag:
                            print('receiver: %s %s' % (staR,netR))

                        #--------loop IV of each component-------
                        for icompR in range(ncomp):
                            cc_indxR = ireceiver*ncomp+icompR

                            #---no data for icomp---
                            if cc_flag[cc_indxR]==0:
                                #print('no data for %dth comp of %s' %(icompR,staR))
                                continue
                            
                            t2 = time.time()
                            fft2 = cc_array[cc_indxR][:]
                            fft2 = fft2.reshape(nhours,Nfft//2)
                            receiver_std = cc_std[cc_indxR][:]

                            #---------- check the existence of earthquakes ----------
                            rec_ind = np.where(receiver_std < 10)[0]

                            #-----note that Hi-net has a few mi-secs differences to Mesonet in terms starting time-----
                            bb=np.intersect1d(sou_ind,rec_ind)
                            if len(bb)==0:
                                continue

                            t3=time.time()
                            corr=noise_module.optimized_correlate1(sfft1[bb,:],fft2[bb,:],\
                                    np.round(maxlag),dt,Nfft,len(bb),method)
                            t4=time.time()

                            #print('finished %s %s comp %s %s'%(staS,staR,icompS,icompR))

                            #---------------keep daily cross-correlation into a hdf5 file--------------
                            cc_aday_h5 = os.path.join(CCFDIR,iday+'.h5')
                            crap   = np.zeros(corr.shape,dtype=np.float32)

                            if not os.path.isfile(cc_aday_h5):
                                with pyasdf.ASDFDataSet(cc_aday_h5,mpi=False) as ccf_ds:
                                    pass 

                            with pyasdf.ASDFDataSet(cc_aday_h5,mpi=False) as ccf_ds:
                                parameters = noise_module.optimized_cc_parameters(dt,maxlag,str(method),len(bb),lonS,latS,lonR,latR)

                                #-----------make a universal change to component-----------
                                if icompR==0:
                                    compR = 'E'
                                elif icompR==1:
                                    compR = 'N'
                                elif icompR==2:
                                    compR = 'Z' 

                                if icompS==0:
                                    compS = 'E'
                                elif icompS==1:
                                    compS = 'N'
                                elif icompS==2:
                                    compS = 'Z' 

                                #------save the time domain cross-correlation functions-----
                                path = netR+'s'+staR+'s'+compR+str(iseg)
                                data_type = netS+'s'+staS+'s'+compS

                                crap[:] = corr[:]
                                ccf_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                            t5=time.time()
                            if flag:
                                print('read R %6.4fs, cc %6.4fs, write cc %6.4fs'% ((t3-t2),(t4-t3),(t5-t4)))

            cc_array=[];cc_std=[];cc_flag=[]
            n = gc.collect()
            print('unreadable garbarge',n)

            ttr2 = time.time()
            print('it takes %6.4fs to process %dth segment of data' %((ttr2-ttr1),iseg))

        tt1 = time.time()
        print('it takes %6.4fs to process day %s [%d segment] in step 2' % (tt1-tt0,iday,num_seg))


ttt1=time.time()
print('all step 2 takes %6.4fs'%(ttt1-ttt0))

comm.barrier()
if rank == 0:
    sys.exit()
