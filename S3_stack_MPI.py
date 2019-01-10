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

#----------some common variables here----------
#CCFDIR = '/n/flashlfs/mdenolle/KANTO/DATA/CCF_deconv'
#CCFDIR = '/n/regal/denolle_lab/cjiang/CCF'
#STACKDIR = '/n/flashlfs/mdenolle/KANTO/DATA/STACK_deconv'
#STACKDIR = '/n/regal/denolle_lab/cjiang/STACK'
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'

t0=time.time()

CCFDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF1'
STACKDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/STACK1'
locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations_small.txt'

#---------filter paramters------
freqmin=0.05
freqmax=4

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
    ccfs = glob.glob(os.path.join(CCFDIR,'*/*.h5'))
    splits = len(ccfs)
else:
    locs,sta,ccfs,splits=[None for _ in range(4)]

locs = comm.bcast(locs,root=0)
sta  = comm.bcast(sta,root=0)
ccfs = comm.bcast(ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra = splits % size

#-----loop I: source stations------
for ii in range(rank,splits+size-extra,size):
    
    if ii<splits:
        fft_h5 = ccfs[ii]
        print(['working on '+fft_h5])
        ds = pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r')

        #-----extract source information-----
        ssta = fft_h5.split('.')[2]
        indx = sta.index(ssta)
        slat = locs.iloc[indx]['latitude']
        slon = locs.iloc[indx]['longitude']
        del indx

        source = fft_h5.split('/')[-2]
        if os.path.exists(os.path.join(STACKDIR,source))==False:
            os.mkdir(os.path.join(STACKDIR,source))        
        
        #-----extract receiver information-----
        tsta = fft_h5.split('.')[-2]
        indx = sta.index(tsta)
        rlat = locs.iloc[indx]['latitude']
        rlon = locs.iloc[indx]['longitude']
        print(['source '+ssta+' receiver '+tsta])

        #-------loop II: cross components-------
        data_types = ds.auxiliary_data.list()
        for kk in range(len(data_types)):
            dtype = data_types[kk]

            path_list = ds.auxiliary_data[dtype].list()
            
            #----------some common variables for the stacked trace--------
            maxlag = ds.auxiliary_data[dtype][path_list[0]].parameters['maxlag']
            dt = ds.auxiliary_data[dtype][path_list[0]].parameters['dt']
            tcorr = np.arange(-maxlag,maxlag+dt,dt)
            ncorr = np.zeros(tcorr.shape)
            #pcorr = np.zeros(tcorr.shape)

            #-------loop III: ccfs at different days--------
            for tt in range(len(path_list)):
                tpath = path_list[tt]
                tdata = ds.auxiliary_data[dtype][tpath].data[:]
                if tdata.ndim==2:
                    y = np.mean(tdata,axis=0)
                    #y2 = noise_module.butter_pass(y,freqmin,freqmax,dt,2)
                elif tdata.ndim==1:
                    y = tdata
                else:
                    continue
                ncorr = ncorr + y

                #-------pws stacking--------
                #y2 = noise_module.pws(np.array(tdata),2,1/dt,5.)
                #y4 = noise_module.butter_pass(y2,freqmin,freqmax,dt,2)
                #pcorr = pcorr + y2
 
            #------------------save two stacked traces into SAC files-----------------
            filename1 = os.path.join(STACKDIR,source,path_list[0][0:len(tpath)-11]+'_ls.SAC')
            #filename2 = os.path.join(STACKDIR,source,path_list[0][0:len(tpath)-11]+'_pws.SAC')
            sac1 = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,
                            delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=ncorr)
            #sac2 = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,
            #                delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=pcorr)
            sac1.write(filename1,byteorder='big')
            #sac2.write(filename2,byteorder='big')
            
            del sac1,ncorr

        del ds

t1=time.time()
print('s3 takes '+str(t1-t0)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
