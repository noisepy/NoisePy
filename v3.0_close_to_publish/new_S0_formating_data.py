import os
import sys
import glob
from datetime import datetime
import numpy as np
import scipy
from scipy.fftpack.helper import next_fast_len
from obspy.signal.filter import bandpass
import obspy
import matplotlib.pyplot as plt
import noise_module
import time
import pyasdf
import pandas as pd
from mpi4py import MPI


'''
this script pre-processs the noise data for each single station using the parameters given below 
and stored the whitened and nomalized fft trace for each station in a HDF5 file as *.h5.

this version is implemented with MPI (Nov.09.2018)
by C.Jiang, T.Clements, M.Denolle
'''

t00=time.time()
#------form the absolute paths-------
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'
#FFTDIR = '/n/flashlfs/mdenolle/KANTO/DATA/FFT/'
#FFTDIR = '/n/regal/denolle_lab/cjiang/FFT'
#event = '/n/flashlfs/mdenolle/KANTO/DATA/????/Event_????_???'
#resp_dir = '/n/flashlfs/mdenolle/KANTO/DATA/resp'


#------form the absolute paths-------
rootpath  = '/mnt/data1/JAKARTA/'
DATA1 = os.path.join(rootpath,'DATA2')
locations = os.path.join(rootpath,'locations.txt')
event = os.path.join(rootpath,'DATA1/20??_???/*/')
#2013_322/*/JKA02/JKA02miniSEED/*
resp_dir = os.path.join(rootpath,'resp')
comp=['CHZ','CHN','CHE']

#-----some control parameters------
prepro=True    # do you need to reprocess the data?
to_whiten=False   # do you want to whiten the spectrum?
rm_resp_spectrum=False       #remove response using spectrum?
rm_resp_inv=False           #remove response using inventory
flag=True                  #print intermediate variables and computing time
time_norm=False

down_sample=True
ftype='mseed'   # file data type
pre_filt=[0.001,0.01,20,25]
downsamp_freq=50
dt=1/downsamp_freq
cc_len=300
step=60
freqmin=0.05   # minimum frequency to whiten in
freqmax=20   # maximum frequency to whiten in



#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------


if rank == 0:
    #----define common variables----
    locs = pd.read_csv(locations)
    nsta = len(locs)
    splits = nsta
    tdir = sorted(glob.glob(event))
else:
    splits,locs,tdir = [None for _ in range(3)]


#-----broadcast the variables------
splits = comm.bcast(splits,root=0)
locs = comm.bcast(locs,root=0)
tdir = comm.bcast(tdir,root=0)
extra = splits % size

#--------MPI: loop through each station--------
for ista in range (rank,splits+size-extra,size):
    if ista<splits:
        t10=time.time()
        #----loop through each day on each core----

        station = locs.iloc[ista]['station']
        network = locs.iloc[ista]['network']
        if flag:
            print("working on station %s " % station) 

        #----loop through each channel----

        for icomp in comp:

            # OPEN THE ASDF FILE HERE
            data_h5 = os.path.join(DATA1,network+'.'+station+"."+icomp+'.h5')
            print(data_h5)
            if not os.path.isfile(data_h5):
                with pyasdf.ASDFDataSet(data_h5,mpi=False,compression=None) as ds:
                    pass # create pyasdf file 
        
            
            #bigsource1=obspy.Stream()
            for jj in range (len(tdir)):
                                
                # read all of the data within directory, merge them all, then pre-process
                tfiles = sorted(glob.glob(tdir[jj]+"/"+station+"/"+station+"miniSEED/*"+icomp))
                if len(tfiles)==0:
                    print(str(station)+' does not have sac file at '+str(tdir[jj]))
                    continue

                # loop through every 10 files
                for ig in range((len(tfiles)%10)*2):
                    bigsource=obspy.Stream()
                    print(ig)
                    for tfile in tfiles[10*ig+5:(ig+1)*11+5]:
                        sacfile = os.path.basename(tfile)
                        try:
                            source1 = obspy.read(tfile)
                            bigsource.append(source1[0])
                            print(source1)
                        except Exception as inst:
                            print(type(inst))
                            continue
                        if flag:
                            print("working on sacfile %s" % sacfile)

                    if len(bigsource)==0:
                        continue
                    print(bigsource)

                        #---------make an inventory---------
                    inv1=noise_module.stats2inv(bigsource[0].stats,locs=locs)

                        #------------Pre-Processing-----------
                        #source = obspy.Stream()
                    bigsource = bigsource.merge(method=1,fill_value='interpolate')#[0]
                    bigsource=noise_module.preprocess_raw(bigsource,downsamp_freq,clean_time=True,pre_filt=None,resp=None,respdir=None)
                    print(bigsource)
                        
                    with pyasdf.ASDFDataSet(data_h5,mpi=False,compression=None) as fft_ds:
                        fft_ds.add_stationxml(inv1)
                        for ii in range(len(bigsource)):
                            print((bigsource[ii]))
                            if isinstance(bigsource[ii],np.float32):
                                continue
                            fft_ds.add_waveforms(bigsource[ii], tag='waveforms')
                            print(bigsource[ii].stats.starttime)
#                            plt.plot
                        print(fft_ds)
                    del bigsource
            del tfiles
        t11=time.time()
        print('it takes '+str(t11-t10)+' s to process one station in step 1')

t01=time.time()
print('step1 takes '+str(t01-t00)+' s')

comm.barrier()

if rank == 0:
    sys.exit()