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


def plot_moveout(sfile,net1,sta1,comp1):            
    '''
    this script plots the cross-correlation functions for the station pair of sta1-sta2
    and component 1 and component 2
    '''
    
    freqmin=0.1
    freqmax=0.5
    maxlag = 100
    dt = 0.05
    tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt

    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()
    isource = net1+'s'+sta1+'s'+comp1
    if isource in slist:
        sta1  = isource.split('s')[1]
        comp1 = isource.split('s')[2]
        rlist = ds.auxiliary_data[isource].list()
        mdist = 0

        for ireceiver in rlist:
            sta2 = ireceiver.split('s')[1]
            comp2 = ireceiver.split('s')[2]
            dist = ds.auxiliary_data[isource][ireceiver].parameters['dist']
            data = ds.auxiliary_data[isource][ireceiver].data[:]/max(ds.auxiliary_data[isource][ireceiver].data[:])
            data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            npts = len(data)
            indx0 = npts//2
            tindx = int(maxlag/dt)
            plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+dist,'b-',linewidth=0.8)
            plt.title(sta1+" "+comp1)
            plt.xlabel('time (s)')
            plt.ylabel('offset (km)')
            plt.text(maxlag*0.9,dist,sta2,fontsize=6)

            if mdist < dist:
                mdist = dist
        plt.plot([0,0],[0,mdist],'r--',linewidth=0.8)
        plt.show()


def plot_cc_2lags(sfile,net1,sta1,comp1,net2,sta2,comp2):
    '''
    this script plots the two lags of the cross-correlation functions to compare
    the symmetry of the waveforms
    '''
    freqmin=0.1
    freqmax=0.3
    maxlag = 100
    dt = 0.05
    tt = np.arange(0, maxlag/dt+1)*dt

    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()
    isource = net1+'s'+sta1+'s'+comp1
    ireceiver = net2+'s'+sta2+'s'+comp2

    if isource in slist:
        rlist = ds.auxiliary_data[isource].list()
        if ireceiver in rlist:
            dist = ds.auxiliary_data[isource][ireceiver].parameters['dist']
            data = ds.auxiliary_data[isource][ireceiver].data[:]
            data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            npts = len(data)
            indx0 = npts//2
            tindx = int(maxlag/dt)
            plt.plot(tt,data[indx0:indx0+tindx+1],'r-',linewidth=2)
            plt.plot(tt,np.flip(data[indx0-tindx:indx0+1]),'g-',linewidth=2)
            plt.title('%s_%s_%s_%s %6.1f km' % (sta1,comp1,sta2,comp2,dist))
            plt.legend(['positive','negative'],loc='upper right')
            plt.show()