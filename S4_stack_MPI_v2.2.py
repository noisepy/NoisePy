import os
import glob
import sys
import obspy
import time
import noise_module
import numpy as np
import pandas as pd
from obspy.io.sac.sactrace import SACTrace
import pyasdf
from mpi4py import MPI

t0=time.time()
#----------some common variables here----------
#CCFDIR = '/n/flashlfs/mdenolle/KANTO/DATA/CCF_deconv'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'
#STACKDIR = '/n/flashlfs/mdenolle/KANTO/DATA/STACK_deconv'
#STACKDIR = '/n/regal/denolle_lab/cjiang/STACK'
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'

CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_C3'
locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations_small.txt'
STACKDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/STACK_C3'

flag = False
maxlag = 500
downsamp_freq=20
dt=1/downsamp_freq
comp1 = ['EHE','EHN','EHZ']
comp2 = ['HNE','HNN','HNU']
#comp1 = ['BHE','BHN','BHZ']
#comp2 = ['BHE','BHN','BHZ']

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

    for ista in sta:
        if not os.path.exists(os.path.join(STACKDIR,ista)):
            os.mkdir(os.path.join(STACKDIR,ista))

    #-------make station pairs based on list--------        
    pairs= noise_module.get_station_pairs(sta)
    ccfs = glob.glob(os.path.join(CCFDIR,'*.h5'))
    splits = len(pairs)
else:
    locs,pairs,ccfs,splits=[None for _ in range(4)]

locs   = comm.bcast(locs,root=0)
pairs  = comm.bcast(pairs,root=0)
ccfs   = comm.bcast(ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size

#-----loop I: source stations------
for ii in range(rank,splits+size-extra,size):
    
    if ii<splits:

        source,receiver = pairs[ii][0],pairs[ii][1]
        sta  = list(locs.iloc[:]['station'])
        ncorr = np.zeros((9,int(2*maxlag/dt)+1),dtype=np.float32)
        nflag = np.zeros(9,dtype=np.int16)

        #-------just loop through each day-----
        for iday in range(len(ccfs)):
            if flag:
                print("work on source %s receiver %s at day %s" % (source, receiver,ccfs[iday]))

            #-----a flag to find comp info for S+R------
            if iday==0:

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

                if flag:
                    print("first day to get source and receiver information")

            fft_h5 = ccfs[iday]
            with pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r') as ds:

                #-------find the data types for source A--------
                data_types = ds.auxiliary_data.list()
                
                for icompS in range(len(compS)):
                    sfile = netS+'s'+source+'s'+compS[icompS]
                    if flag:
                        print("work on source %s" % sfile)
                    
                    if sfile in data_types:
                        indxS = data_types.index(sfile)
                        dtype  = data_types[indxS]
                        path_list = ds.auxiliary_data[dtype].list()

                        for icompR in range(len(compR)):
                            rfile = netR+'s'+receiver+'s'+compR[icompR]

                            if flag:
                                print("work on receiver %s" % rfile)
                            
                            if rfile in path_list:
                                indxR = path_list.index(rfile)
                                tindx  = icompS*3+icompR
                                ncorr[tindx,:] += ds.auxiliary_data[dtype][path_list[indxR]].data[:]
                                nflag[tindx] += 1

                                if flag:
                                    print("stacked for day %s" % rfile)
    
        #------------------save two stacked traces into SAC files-----------------
        for icompS in range(len(compS)):
            for icompR in range(len(compR)):
                if nflag[icompS*3+icompR] >0:
                    temp = netS+'.'+source+'_'+netR+'.'+receiver+'_'+compS[icompS]+'_'+compR[icompR]+'.SAC'
                    filename = os.path.join(STACKDIR,source,temp)
                    sac = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,\
                        delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=ncorr[icompS*3+icompR,:])
                    sac.write(filename,byteorder='big')
                    if flag:
                        print("wrote to %s" % temp)

t1=time.time()
print('S3 takes '+str(t1-t0)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
