import os
import glob
import sys
import obspy
import time
import noise_module
import numpy as np
import pandas as pd
import itertools
from obspy.io.sac.sactrace import SACTrace
import pyasdf
from mpi4py import MPI

#----------some common variables here----------
#CCFDIR = '/n/flashlfs/mdenolle/KANTO/DATA/CCF_deconv'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'
#STACKDIR = '/n/flashlfs/mdenolle/KANTO/DATA/STACK_deconv'
#STACKDIR = '/n/regal/denolle_lab/cjiang/STACK'
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'

t0=time.time()

CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_opt'
locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations_small.txt'
STACKDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/STACK_opt'
maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq
comp1 = ['EHE','EHN','EHZ']
comp2 = ['HNE','HNN','HNU']

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    #----check the directory of STACK----
    if os.path.exists(STACKDIR)==False:
        os.mkdir(STACKDIR)

    #-----other variables to share-----
    locs = pd.read_csv(locations)
    sta  = list(locs.iloc[:]['station'])
    pairs = list(itertools.combinations(sta,2))
    ccfs = glob.glob(os.path.join(CCFDIR,'*.h5'))
    splits = len(pairs)
else:
    locs,sta,ccfs,splits=[None for _ in range(4)]

locs   = comm.bcast(locs,root=0)
pairs  = comm.bcast(pairs,root=0)
ccfs   = comm.bcast(ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size

#-----loop I: source stations------
for ii in range(rank,splits+size-extra,size):
    
    if ii<splits:

        source,receiver = pairs[ii][0],pairs[ii][1]
        tcorr = np.arange(-maxlag-dt,maxlag,dt)
        ncorr = np.zeros((9,tcorr.size),dtype=np.float32)
        nflag = np.zeros(9,dtype=np.int16)

        #-------just loop through each day-----
        for iday in range(len(ccfs)):

            #-----a flag to find comp info for S+R------
            if iday==0:

                if not os.path.exists(os.path.join(STACKDIR,source)):
                    os.mkdir(os.path.join(STACKDIR,source))

                #----source information-----
                indx = sta.index(source)
                slat = locs.iloc[indx]['latitude']
                slon = locs.iloc[indx]['longitude']
                netS = locs.iloc[indx]['network']
                if netS == 'E' or netS == 'OK':
                    compS = comp2
                else:
                    compS = comp1

                #-----receiver information------
                indx = sta.index(receiver)
                rlat = locs.iloc[indx]['latitude']
                rlon = locs.iloc[indx]['longitude']
                netR = locs.iloc[indx]['network']
                if netR == 'E' or netR == 'OK':
                    compR = comp2
                else:
                    compR = comp1

            fft_h5 = ccfs[iday]
            ds = pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r')

            #-------find the data types for source A--------
            data_types = ds.auxiliary_data.list()
            
            for icompS in range(len(compS)):
                sfile = netS+'s'+source+'s'+compS[icompS]
                
                if sfile in data_types:
                    indxS = data_types.index(sfile)
                    dtype  = data_types[indxS]
                    path_list = ds.auxiliary_data[dtype].list()

                    for icompR in range(len(compR)):
                        rfile = netR+'_'+receiver+'_'+compR[icompR]

                        if rfile in path_list:
                            indxR = path_list.index(rfile)
                            tindx  = icompS*3+icompR
                            ncorr[tindx,:] += ds.auxiliary_data[dtype][path_list[indxR]].data[:]
                            nflag[tindx] += 1
 
        #------------------save two stacked traces into SAC files-----------------
        for icompS in range(len(compS)):
            for icompR in range(len(compR)):
                if nflag[icompS*3+icompR] >0:
                    temp = netS+'.'+source+'_'+netR+'.'+receiver+'_'+compS[icompS]+'_'+compR[icompR]+'.SAC'
                    filename = os.path.join(STACKDIR,source,temp)
                    sac = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,\
                        delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=ncorr[icompS*3+icompR,:])
                    sac.write(filename,byteorder='big')

t1=time.time()
print('s3 takes '+str(t1-t0)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
