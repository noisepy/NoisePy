import os
import sys
import glob
import time
import scipy
import pyasdf
import numpy as np
import pandas as pd
import noise_module
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.fftpack.helper import next_fast_len

'''
this scripts takes the ASDF file outputed by script S2 (cc function for day x)
and get the spectrum of the coda part of the cc and store them in a
new HDF5 files for later stacking in script S4
Chengxin Jiang (Feb.15.2019)

use 1.5 km/s as the min vs to start the waveform window with 1000 s long. note 
that the maxlag should be modified if longer coda window is needed

use rfft in the future to keep longer c3 functions
'''

#----------some common variables here----------
#CCFDIR = '/n/flashlfs/mdenolle/KANTO/DATA/CCF_deconv'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'

CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF'
C3DIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_C3'

save_cc = True         #---whether to save c1 array into a numpy data-----
flag  = False
vmin  = 1.0
wcoda = 1000
maxlag = 1800
downsamp_freq=20
dt=1/downsamp_freq
Nfft  = int(next_fast_len(int(wcoda/dt)+1))
tt    = np.arange(-Nfft//2, Nfft//2+1)*dt
ind   = np.where(np.abs(tt) <= wcoda//2)[0]


#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    #----check the directory of STACK----
    if os.path.exists(C3DIR)==False:
        os.mkdir(C3DIR)

    #-----other variables to share-----
    daily_ccfs = glob.glob(os.path.join(CCFDIR,'*.h5'))
    splits = len(daily_ccfs)
else:
    daily_ccfs,splits = [None for _ in range(2)]

daily_ccfs   = comm.bcast(daily_ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size

#--------MPI loop through each day---------
for ii in range(rank,splits+size-extra,size):

    if ii<splits:
        t00=time.time()
        dayfile = daily_ccfs[ii]

        if flag:
            print('work on day %s' % dayfile)

        #------------dayfile contains everything needed----------
        with pyasdf.ASDFDataSet(dayfile,mpi=False,mode='r') as ds:
            data_types = ds.auxiliary_data.list()

            #------form a full list of station and station pairs following the order in hdf5 file------
            all_list = data_types[0:3]+ds.auxiliary_data[data_types[0]].list()
            pairs = []
            for ii in range(len(data_types)):
                tpaths = ds.auxiliary_data[data_types[ii]].list()
                for jj in range(len(tpaths)):
                    pairs.append((data_types[ii],tpaths[jj]))
            npairs = len(pairs)

            #-------make a crutial estimate on the memory needed to load the daily CC files: defaultly taken as float32--------
            ccfs_size  = 2*int(maxlag*downsamp_freq)+1
            memory_size = npairs*4*ccfs_size/1024/1024/1024
            if memory_size > 20:
                raise MemoryError('Memory exceeds 20 GB! No enough memory to load them all once!')

            #------cc_array holds all ccfs and npairs tracks the number of pairs for each source-----
            if flag:
                print('initialize the array for storing all cc data')
            cc_array = np.zeros((npairs,ccfs_size),dtype=np.float32)
            cc_dist  = np.zeros(npairs,dtype=np.float32)
            c3_win   = np.zeros(4,dtype=np.int16)
            k=0

            #-------load everything here to avoid repeating read HDF5 files------
            for ii in range(len(data_types)):
                data_type=data_types[ii]
                tpaths = ds.auxiliary_data[data_type].list()

                for jj in range(len(tpaths)):
                    tpath = tpaths[jj]
                    cc_array[k][:]= ds.auxiliary_data[data_type][tpath].data[:]
                    cc_dist[k]    = ds.auxiliary_data[data_type][tpath].parameters['dist']
                    
                    if flag:
                        print('station pair %s %s is %7.1f km apart' % (data_type,tpath,cc_dist[k]))
                    k+=1
            
            #-----get the common window for C3-------
            dist = max(cc_dist)
            c3_win[:] = noise_module.get_coda_window(dist,vmin,maxlag,dt,wcoda)  

            if k != npairs:
                raise ValueError('the size of cc_array is not right [%d vs %d]' % (k,npairs))

            if save_cc:
                np.save(dayfile,cc_array)

            #-----loop each station pair-----
            for ii in range(npairs):
                t0=time.time()
                sta1,sta2 = pairs[ii][0],pairs[ii][1]

                #-----use index in all_list to identify the order of virtual source-----
                indx1 = all_list.index(sta1)
                indx2 = all_list.index(sta2)

                if flag:
                    print('doing C3 for %dth station pair: [%s %s]' % (ii,sta1,sta2))

                #----------------------------------------------------------------
                #------this condition can be easily modified if you want to -----
                #----------other components of the C3 function as well-----------
                #----------------------------------------------------------------
                if sta1[-1] != 'Z' or sta2[-1] != 'Z':
                    print('Only do Z component here!!!')
                    continue

                #-----------initialize some variables-----------
                cc_P = np.zeros(Nfft,dtype=np.complex64)
                cc_N = cc_P
                cc_final = cc_P

                tpairs = 0
                #------loop through all virtual sources------
                for indx in range(len(all_list)):
                    t01=time.time()
                    virtualS = all_list[indx]

                    #---------condition when virtual source is either station 1 or station 2-----------
                    if virtualS.split('s')[1] == sta1.split('s')[1] or virtualS.split('s')[1] == sta2.split('s')[1]:
                        print('Moving to next virtual source!')
                        continue

                    #------for situation of virtualS A -> (sta1 C,sta2 F)------
                    if indx < indx1:
                        k1=pairs.index((virtualS,sta1))
                        k2=pairs.index((virtualS,sta2))
                        S1_data = cc_array[k1][:]
                        S2_data = cc_array[k2][:]
                        
                        if flag:    
                            print('situation 1: virtualS %s sta1 %s sta2 %s; index %d %d' % (virtualS,sta1,sta2,k1,k2))
                    
                    #------for situation of E->(C,F)-------
                    elif indx < indx2:
                        k1=pairs.index((sta1,virtualS))
                        k2=pairs.index((virtualS,sta2))
                        S1_data = cc_array[k1][:]
                        S1_data = S1_data[::-1]
                        S2_data = cc_array[k2][:]
                    
                        if flag:    
                            print('situation 2: virtualS %s sta1 %s sta2 %s; index %d %d' % (virtualS,sta1,sta2,k1,k2))

                    #------for situation of G->(C,F)-------
                    else:
                        k1=pairs.index((sta1,virtualS))
                        k2=pairs.index((sta2,virtualS))
                        S1_data = cc_array[k1][:]
                        S1_data = S1_data[::-1]
                        S2_data = cc_array[k2][:]
                        S2_data = S2_data[::-1]
                        
                        if flag:    
                            print('situation 3: virtualS %s sta1 %s sta2 %s; index %d %d' % (virtualS,sta1,sta2,k1,k2))
            
                    #--------cast all processing into C3-process function-------
                    ccp,ccn=noise_module.C3_process(S1_data,S2_data,Nfft,c3_win)
                    cc_P+=ccp
                    cc_N+=ccn
                    tpairs+=1
                    
                    t02=time.time()
                    if flag:
                        print('moving to next virtual source')
                        print('a virtual source takes %f s to compute' %(t02-t01))

                #-------stack the contribution from all virtual sources------
                cc_P = cc_P/tpairs
                cc_N = cc_N/tpairs
                cc_final = 0.5*cc_P + 0.5*cc_N
                cc_final = np.real(np.fft.ifftshift(scipy.fftpack.ifft(cc_final, Nfft,axis=0)))
                corr = cc_final[ind]

                if flag:
                    print('start to ouput to HDF5 file')

                #------ready to write into HDF5 files-------
                c3_h5 = os.path.join(C3DIR,dayfile.split('/')[-1])
                crap  = np.zeros(corr.shape)

                if not os.path.isfile(c3_h5):
                    with pyasdf.ASDFDataSet(c3_h5,mpi=False,mode='w') as ds:
                        pass 

                with pyasdf.ASDFDataSet(c3_h5,mpi=False,mode='a') as ccf_ds:
                    parameters = {'dt':dt, 'maxlag':maxlag, 'wcoda':wcoda, 'vmin':vmin}

                    #------save the time domain cross-correlation functions-----
                    path = sta2
                    data_type = sta1
                    crap = corr
                    ccf_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                t1=time.time()
                print('one station pair taks %f s to compute' %(t1-t0))

        t10=time.time()
        print('one days takes %f s to compute' % (t10-t00))

comm.barrier()
if rank == 0:
    sys.exit()