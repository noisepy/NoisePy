import os
import glob
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

'''
the main purpose of this module is to assemble short functions
to plot the intermediate files for Noise_Python package
'''

#def compare_c2_c3_waveforms(c2file,c3file,maxlag,c2_maxlag,dt):
'''
use data type from c3file to plot the waveform for c2 and c3
note that the length of c3file is shorter than c2file
c2file: HDF5 file for normal cross-correlation function
c3file: HDF5 file for C3 function
maxlag: maximum time lag for C3
c2_maxlag: maxinum time lag for C1
dt: time increment
'''

c2file = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF/2010_01_11.h5'
c3file = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_C3/2010_01_11.h5'
maxlag = 1000
c2_maxlag = 1800
dt = 0.05

#-------time axis-------
tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt
tt_c2 = np.arange(-c2_maxlag/dt, c2_maxlag/dt+1)*dt
ind   = np.where(abs(tt_c2)<=-tt[0])[0]
c3_waveform = np.zeros(tt.shape,dtype=np.float32)
c2_waveform = np.zeros(tt_c2.shape,dtype=np.float32)

#-------make station pairs--------
ds_c2 = pyasdf.ASDFDataSet(c2file,mode='r')
ds_c3 = pyasdf.ASDFDataSet(c3file,mode='r')

#------loop through all c3 data_types-------
data_type_c3 = ds_c3.auxiliary_data.list()
for ii in range(len(data_type_c3)):
    path_c3 = ds_c3.auxiliary_data[data_type_c3[ii]].list()
    for jj in range(len(path_c3)):
        print(data_type_c3[ii],path_c3[jj])

        sta1 = data_type_c3[ii].split('s')[1]
        sta2 = path_c3[jj].split('s')[1]
        c3_waveform = ds_c3.auxiliary_data[data_type_c3[ii]][path_c3[jj]].data[:]
        c2_waveform = ds_c2.auxiliary_data[data_type_c3[ii]][path_c3[jj]].data[:]
        c1_waveform = c2_waveform[ind]
        
        plt.subplot(211)
        plt.plot(c3_waveform)
        plt.subplot(212)
        plt.plot(c1_waveform)
        plt.legend(sta1+'_'+sta2,loc='upper right')
        plt.show()


def plot_spectrum(sfile,iday,icomp):
    '''
    this script plots the noise spectrum for the idayth on icomp (results from step1)
    '''


def plot_c1_waveform(sfile,sta1,sta2,comp1,comp2):            
    '''
    this script plots the cross-correlation functions for the station pair of sta1-sta2
    and component 1 and component 2
    '''


import pandas as pd


freqmin=0.15
freqmax=0.5
maxlag = 180
c2_maxlag = 1800
dt = 0.02
locations = '/mnt/data1/JAKARTA/locations.txt'
locs = pd.read_csv(locations)
sta  = list(locs.iloc[:]['station'])

#-------time axis-------
tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt
tt_c2 = np.arange(-c2_maxlag/dt, c2_maxlag/dt+1)*dt
ind   = np.where(abs(tt_c2)<=-tt[0])[0]
c3_waveform = np.zeros(tt.shape,dtype=np.float32)
c2_waveform = np.zeros(tt_c2.shape,dtype=np.float32)

CCFDIR = '/mnt/data1/JAKARTA/CCF'
Cfiles = sorted(glob.glob(os.path.join(CCFDIR,'*.h5')))
for c2file in Cfiles:
    print(c2file)
    ds_c2 = pyasdf.ASDFDataSet(c2file,mode='r')
    data_type_c2 = ds_c2.auxiliary_data.list()
    print(ds_c2)
    for ii in range(len(data_type_c2)):
        path_c2 = ds_c2.auxiliary_data[data_type_c2[ii]].list()
        for jj in range(len(path_c2)):
            sta1 = data_type_c2[ii].split('s')[1]
            sta2 = path_c2[jj].split('s')[1]
            cmp1 = data_type_c2[ii].split('s')[2]
            cmp2 = path_c2[jj].split('s')[2]
            print(sta1,sta2,cmp1,cmp2)
            c2_waveform = bandpass(ds_c2.auxiliary_data[data_type_c2[ii]][path_c2[jj]].data[:],freqmin,freqmax,1/dt, corners=4, zerophase=True)[ind]
            dist=ds_c2.auxiliary_data[data_type_c2[ii]][path_c2[jj]].parameters['dist']
            #c1_waveform = c2_waveform[ind]
            #plt.subplot(211)
            #plt.plot(c3_waveform)
            #plt.subplot(212)
            plt.plot(tt,c2_waveform/np.max(np.abs(c2_waveform))+dist)
            plt.title(sta1+" "+cmp1)
            plt.legend(sta2,loc='upper right')
        plt.show()