import os
import glob
import obspy
import pyasdf
import numpy as np
import noise_module
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

'''
this script is an updated version of plot_modules.py to plot the CCFs in 2D
'''

def plot_moveout_stack(sdir,freqmin,freqmax,ccomp,maxlag=None,tag=None):
    '''
    plot the moveout of the CCFs located in sdir. waveforms are to be filtered at freqmin-freqmax

    usage: plot_moveout('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.AYHM',0.1,0.3,'ZZ')
    '''

    #---all station pairs----
    afiles = sorted(glob.glob(os.path.join(sdir,'*.h5')))
    nsta = len(afiles)
    if not maxlag:
        maxlag = 100
    if not tag:
        tag = 'Allstacked'

    #-----some basic parameters------
    ds = pyasdf.ASDFDataSet(afiles[0],mode='r')
    try:
        delta = ds.auxiliary_data[tag][ccomp].parameters['dt']
        lag   = ds.auxiliary_data[tag][ccomp].parameters['lag']
    except Exception as error:
        print(error)
        print('cannot find delta and lag')
    del ds

    #--------index for the data---------
    indx1 = int((lag-maxlag)/delta)
    indx2 = int((lag+maxlag)/delta)
    t     = np.arange(0,indx2-indx1+1,800)*delta-maxlag

    #------initialize the array--------
    data = np.zeros((nsta,indx2-indx1+1),dtype=np.float32)
    dist = np.zeros(nsta,dtype=np.float32)
    Ntau  = int(np.ceil(np.min([1/freqmin,1/freqmax])/delta)) + 15
    Mdate = 12
    NSV   = 2

    #-----loop through all files------
    for ii in range(nsta):
        with pyasdf.ASDFDataSet(afiles[ii],mpi=False,mode='r') as ds:
            slist = ds.auxiliary_data.list()
            if tag in slist:
                rlist = ds.auxiliary_data[tag].list()
                if ccomp in rlist:
                    try:
                        tdata = ds.auxiliary_data[tag][ccomp].data[indx1:indx2+1]
                    except Exception:
                        continue

                    #---------filter the data---------
                    data[ii,:] = bandpass(tdata,freqmin,freqmax,int(1/delta),corners=4, zerophase=True)
                    data[ii,:] = data[ii,:]/max(data[ii,:])
                    #----------keep a track of distance info---------
                    dist[ii]   = ds.auxiliary_data[tag][ccomp].parameters['dist']

    #----sort the data using dist-----
    new_orders = np.argsort(dist)

    #-------plotting the move-out-------
    fig,ax = plt.subplots(2,sharex=True)
    ax[0].matshow(data[new_orders][:],cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
    new = noise_module.NCF_denoising(data[new_orders][:],np.min([Mdate,data.shape[0]]),Ntau,NSV)

    ax[1].matshow(new,cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
    ax[0].set_title('Filterd Cross-Correlations %s' % (sdir.split('/')[-1]))
    ax[1].set_title('Denoised Cross-Correlations')
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel('Distance [km]')
    ax[1].set_xticks(t)
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Distance [km]')
    plt.show() 


def plot_cc_stack(sfile,freqmin,freqmax,ccomp,maxlag=None):
    '''
    plot sub-stacked CCFs for station pair of sfile. waveforms are to be filtered at freqmin-freqmax

    usage: plot_moveout('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.ABHM/E.ABHM_E.KKHM.h5',0.5,1,'ZZ')
    '''

    #---all station pairs----
    if not maxlag:
        maxlag = 100

    #-----some basic parameters------
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()
    rlist = ds.auxiliary_data[slist[0]].list()
    if ccomp in rlist:
        delta = ds.auxiliary_data[slist[0]][ccomp].parameters['dt']
        lag   = ds.auxiliary_data[slist[0]][ccomp].parameters['lag']
    else:
        raise ValueError('ccomp %s is not in the file'%ccomp)

    #--------index for the data---------
    indx1 = int((lag-maxlag)/delta)
    indx2 = int((lag+maxlag)/delta)
    t     = np.arange(0,indx2-indx1+1,400)*delta-maxlag

    #------initialize the array--------
    data = np.zeros((len(slist),indx2-indx1+1),dtype=np.float32)
    Ntau  = int(np.ceil(np.min([1/freqmin,1/freqmax])/delta)) + 15
    Mdate = 12
    NSV   = 2

    #-----loop through all files------
    for ii in range(len(slist)):
        rlist = ds.auxiliary_data[slist[ii]].list()
        if ccomp in rlist:
            try:
                tdata = ds.auxiliary_data[slist[ii]][ccomp].data[indx1:indx2+1]
            except Exception:
                continue
            data[ii,:] = bandpass(tdata,freqmin,freqmax,int(1/delta),corners=4, zerophase=True)
            #data[ii,:] = data[ii,:]/max(data[ii,:])

    fig,ax = plt.subplots(2,sharex=True)
    ax[0].matshow(data/data.max(),cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
    new = noise_module.NCF_denoising(data,np.min([Mdate,data.shape[0]]),Ntau,NSV)

    ax[1].matshow(new/new.max(),cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
    ax[0].set_title('Filterd Cross-Correlations %s' % (sfile.split('/')[-1]))
    ax[1].set_title('Denoised Cross-Correlations')
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel('Day index')
    ax[1].set_xticks(t)
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Day index')
    plt.show() 


def plot_freq_time_stack(sfile,freqmin,freqmax,ccomp,maxlag=None):
    '''
    plot the dispersive CCFs in a very narrow frequency band at freqmin-freqmax range

    usage: plot_freq_time_stack('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.ABHM/E.ABHM_E.KKHM.h5',0.5,1,'ZZ')
    '''

    #---all station pairs----
    if not maxlag:
        maxlag = 100

    #-----some basic parameters------
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()
    rlist = ds.auxiliary_data[slist[0]].list()
    if ccomp in rlist:
        delta = ds.auxiliary_data[slist[0]][ccomp].parameters['dt']
        lag   = ds.auxiliary_data[slist[0]][ccomp].parameters['lag']
    else:
        raise ValueError('ccomp %s is not in the file'%ccomp)

    #--------index for the data---------
    indx1 = int((lag-maxlag)/delta)
    indx2 = int((lag+maxlag)/delta)
    t     = np.arange(0,indx2-indx1+1,400)*delta-maxlag

    #------initialize the array--------
    data = np.zeros((len(slist),indx2-indx1+1),dtype=np.float32)
    Ntau  = int(np.ceil(np.min([1/freqmin,1/freqmax])/delta)) + 15
    Mdate = 12
    NSV   = 2

    #-----loop through all files------
    for ii in range(len(slist)):
        rlist = ds.auxiliary_data[slist[ii]].list()
        if ccomp in rlist:
            try:
                tdata = ds.auxiliary_data[slist[ii]][ccomp].data[indx1:indx2+1]
            except Exception:
                continue
            data[ii,:] = bandpass(tdata,freqmin,freqmax,int(1/delta),corners=4, zerophase=True)
            #data[ii,:] = data[ii,:]/max(data[ii,:])

    fig,ax = plt.subplots(2,sharex=True)
    ax[0].matshow(data/data.max(),cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
    new = noise_module.NCF_denoising(data,np.min([Mdate,data.shape[0]]),Ntau,NSV)

    ax[1].matshow(new/new.max(),cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
    ax[0].set_title('Filterd Cross-Correlations %s' % (sfile.split('/')[-1]))
    ax[1].set_title('Denoised Cross-Correlations')
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel('Day index')
    ax[1].set_xticks(t)
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Day index')
    plt.show() 