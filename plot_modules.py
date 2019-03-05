import os
import glob
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

'''
the main purpose of this module is to assemble some simple functions
to plot different waveform types from intermediate files outputed by 
the Noise_Python package

Chengxin Jiang (Mar.04.2019)

it includes:
1). plot_spectrum(sfile,iday,icomp) to plot the FFT files for each station
2). plot_moveout(sfile,freqmin,freqmax,net1=None,sta1=None,comp1=None) to plot
    the move-out of the ccfs for each source
3). plot_cc_2lags(sfile,net1,sta1,comp1,net2,sta2,comp2) to plot both lags
    of the ccfs to compare the symmetry
4). compare_c2_c3_waveforms(c2file,c3file,maxlag,c2_maxlag,dt)
'''

def plot_spectrum(sfile,iday,icomp):
    '''
    this script plots the noise spectrum for the idayth on icomp (results from step1)
    and compare it with the waveforms in time-domain
    '''
    dt = 0.05
    sta = sfile.split('/')[-1].split('.')[1]
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    comp = ds.auxiliary_data.list()
    
    #--check whether it exists----
    if icomp in comp:
        tlist = ds.auxiliary_data[icomp].list()
        if iday in tlist:
            spect = ds.auxiliary_data[icomp][iday].data[:]

            #---look at hourly----
            if spect.ndim==2:
                nfft = spect.shape[1]
                for ii in range(spect.shape[0]):
                    waveform = np.real(np.fft.irfft(spect[ii])[0:nfft])
                    freq  = np.fft.fftfreq(nfft*2,dt)[0:nfft]
                    plt.subplot(211)
                    plt.loglog(freq,np.abs(spect[ii]),'k-')
                    plt.title('station %s %s @ %s' % (sta,icomp,iday))
                    plt.subplot(212)
                    plt.plot(np.arange(0,nfft)*dt,waveform,'k-')
                    plt.plot([0,nfft*dt],[0,0],'r--',linewidth=1)
                    plt.show()

            #----look at stacked daily----
            else:
                nfft  = len(spect)
                waveform = np.real(np.fft.irfft(spect)[0:nfft])
                freq  = np.fft.fftfreq(nfft*2,dt)[0:nfft]
                plt.subplot(211)
                plt.loglog(freq,np.abs(spect),'k-')
                plt.title('station %s %s @ %s' % (sta,icomp,iday))
                plt.subplot(212)
                plt.plot(np.arange(0,nfft)*dt,waveform,'k-')
                plt.plot([0,nfft*dt],[0,0],'r--',linewidth=1)
                plt.show()


def plot_moveout(sfile,freqmin,freqmax,net1=None,sta1=None,comp1=None):            
    '''
    this script plots the cross-correlation functions for the station pair of sta1-sta2
    and component 1 and component 2 filtered at freq bands of freqmin-freqmax. if no 
    station is provided, it plots the move-out for each virtual source

    usage: plot_moveout('2010_12_18.h5',0.1,0.3) or
           plot_moveout('2010_12_18.h5',0.1,0.3,'E','HNOM','HNU')
    '''
    
    #---basic parameters----
    #freqmin=0.2
    #freqmax=0.5
    maxlag = 100
    dt = 0.05
    tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt

    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()
    #-------to plot one station moveout------
    if net1 and sta1 and comp1:
        isource = net1+'s'+sta1+'s'+comp1
        if isource in slist:
            sta1  = isource.split('s')[1]
            comp1 = isource.split('s')[2]
            rlist = ds.auxiliary_data[isource].list()
            mdist = 0
            plt.figure(figsize=(9,6))

            #---loop through all receivers-----
            for ireceiver in rlist:
                sta2 = ireceiver.split('s')[1]
                comp2 = ireceiver.split('s')[2]
                #--plot all 3 cross-component---
                if comp2[2] == 'E':
                    color = 'r-'
                elif comp2[2] == 'N':
                    color = 'g-'
                else:
                    color = 'b-'

                #---------read parameters and copy data-----------
                dist = ds.auxiliary_data[isource][ireceiver].parameters['dist']
                data = ds.auxiliary_data[isource][ireceiver].data[:]
                data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                data = data*2.5/max(data)
                npts = len(data)
                indx0 = npts//2
                tindx = int(maxlag/dt)
                plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+dist,color,linewidth=0.8)
                plt.title('%s %s filtered @%4.1f-%4.1f Hz' % (sta1,comp1,freqmin,freqmax))
                plt.xlabel('time (s)')
                plt.ylabel('offset (km)')
                plt.text(maxlag*0.9,dist,sta2,fontsize=6)

                #----use to plot o times------
                if mdist < dist:
                    mdist = dist
            plt.plot([0,0],[0,mdist],'r--',linewidth=0.8)
            plt.legend(['E','N','Z'],loc='upper right')
            plt.show()
    else:

        #----loop through all sources------
        for isource in slist:
            sta1  = isource.split('s')[1]
            comp1 = isource.split('s')[2]
            rlist = ds.auxiliary_data[isource].list()
            mdist = 0
            plt.figure(figsize=(9,6))

            #---for all receivers-----
            for ireceiver in rlist:
                sta2 = ireceiver.split('s')[1]
                comp2 = ireceiver.split('s')[2]

                #---plot all 3 cross-component----
                if comp2[2] == 'E':
                    color = 'r-'
                elif comp2[2] == 'N':
                    color = 'g-'
                else:
                    color = 'b-'

                #------------get parameters and copy data-------------
                dist = ds.auxiliary_data[isource][ireceiver].parameters['dist']
                data = ds.auxiliary_data[isource][ireceiver].data[:]
                data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                data = data*2.5/max(data)
                npts = len(data)
                indx0 = npts//2
                tindx = int(maxlag/dt)
                plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+dist,color,linewidth=0.8)
                plt.title('%s %s filtered @%4.1f-%4.1f Hz' % (sta1,comp1,freqmin,freqmax))
                plt.xlabel('time (s)')
                plt.ylabel('offset (km)')
                plt.text(maxlag*0.9,dist,sta2,fontsize=6)

                if mdist < dist:
                        mdist = dist
            plt.plot([0,0],[0,mdist],'r--',linewidth=0.8)
            plt.legend(['E','N','Z'],loc='upper right')
            plt.show()


def plot_cc_2lags(sfile,freqmin,freqmax,net1,sta1,comp1,net2=None,sta2=None,comp2=None):
    '''
    this script plots the two lags of the cross-correlation functions to compare
    the symmetry of the waveforms

    usage: plot_cc_2lags('2010_12_20.h5',0.1,0.3,'E','AYHM','HNU') or
           plot_cc_2lags('2010_12_20.h5',0.1,0.3,'E','AYHM','HNU','E','YKKM','HNU')
    '''
    #---basic parameters---
    #freqmin=0.1
    #freqmax=0.3
    maxlag = 100
    dt = 0.05
    tt = np.arange(0, maxlag/dt+1)*dt

    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()
    isource = net1+'s'+sta1+'s'+comp1

    if isource in slist:
        #-----when receiver info is provided-----
        rlist = ds.auxiliary_data[isource].list()
        if net2 and sta2 and comp2:
            ireceiver = net2+'s'+sta2+'s'+comp2
            if ireceiver in rlist:
                #--------read dist info to infer travel time window---------
                dist = ds.auxiliary_data[isource][ireceiver].parameters['dist']
                data = ds.auxiliary_data[isource][ireceiver].data[:]
                data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                npts = len(data)
                indx0 = npts//2
                tindx = int(maxlag/dt)
                #-------positive lag and flip negative lags----------
                plt.plot(tt,data[indx0:indx0+tindx+1],'r-',linewidth=1)
                plt.plot(tt,np.flip(data[indx0-tindx:indx0+1]),'g-',linewidth=1)
                plt.title('%s_%s_%s_%s %6.1f km @%4.1f-%4.1f Hz' % (sta1,comp1,sta2,comp2,dist,freqmin,freqmax))
                plt.legend(['positive','negative'],loc='upper right')
                plt.xlabel('time [s]')   
                plt.show()
        else:
            for ir in range(len(rlist)//3):
                #----plot all 3 components-----
                ireceiver = rlist[ir*3]
                sta2 = ireceiver.split('s')[1]
                dist = ds.auxiliary_data[isource][ireceiver].parameters['dist']
                data1 = ds.auxiliary_data[isource][ireceiver].data[:]
                data1 = bandpass(data1,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                ireceiver = rlist[3*ir+1]
                data2 = ds.auxiliary_data[isource][ireceiver].data[:]
                data2 = bandpass(data2,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                ireceiver = rlist[3*ir+2]
                data3 = ds.auxiliary_data[isource][ireceiver].data[:]
                data3 = bandpass(data3,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                npts = len(data1)
                indx0 = npts//2
                tindx = int(maxlag/dt)
                #-------plot positive lag and flip negative lags----------
                plt.subplot(311)
                plt.plot(tt,data1[indx0:indx0+tindx+1]/max(data1),'r-',linewidth=1)
                plt.plot(tt,np.flip(data1[indx0-tindx:indx0+1])/max(data1),'g-',linewidth=1)
                plt.title('%s_%s_%s_[ENZ] %6.1f km @%4.1f-%4.1f Hz' % (sta1,comp1,sta2,dist,freqmin,freqmax))
                plt.legend(['positive','negative'],loc='upper right')
                #----3 cross components---
                plt.subplot(312)
                plt.plot(tt,data2[indx0:indx0+tindx+1]/max(data2),'r-',linewidth=1)
                plt.plot(tt,np.flip(data2[indx0-tindx:indx0+1])/max(data2),'g-',linewidth=1)
                plt.subplot(313)
                plt.plot(tt,data3[indx0:indx0+tindx+1]/max(data3),'r-',linewidth=1)
                plt.plot(tt,np.flip(data3[indx0-tindx:indx0+1])/max(data3),'g-',linewidth=1)
                plt.xlabel('time [s]')            
                plt.show()


def plot_cc_withtime(ccfdir,freqmin,freqmax,net1,sta1,comp1,net2=None,sta2=None,comp2=None):
    '''
    plot the filtered cross-correlation functions between station-pair sta1-sta2
    for all of the available days stored in ccfdir

    '''
    #---basic parameters----
    maxlag = 100
    dt = 0.05
    tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt

    afiles = sorted(glob.glob(os.path.join(ccfdir,'*.h5')))
    source = net1+'s'+sta1+'s'+comp1

    #----when recevier is known----
    if net2 and sta2 and comp2:
        recever= net2+'s'+sta2+'s'+comp2

        #---loop through each day------
        for ii in range(len(afiles)):
            
            #--maximum 50 days one plot----
            if ii%50==0:
                plt.figure(figsize=(9,6))

            with pyasdf.ASDFDataSet(afiles[ii],mode='r') as ds:
                dist = ds.auxiliary_data[source][recever].parameters['dist']
                iday = afiles[ii].split('/')[-1].split('.')[0]
                data = ds.auxiliary_data[source][recever].data[:]
                data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                #--normalize the data----
                data = data/max(data)
                npts = len(data)
                #----make index----
                indx0 = npts//2
                tindx = int(maxlag/dt)
                plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+ii*2,'k-',linewidth=0.5)
                plt.text(maxlag*0.9,ii*2,iday,fontsize=6)

        plt.grid(True)
        plt.plot([0,0],[0,ii*2],'r--',linewidth=1.5)
        plt.title('%s_%s_%s_%s dist %6.1f @%4.1f-%4.1f Hz' % (sta1,comp1,sta2,comp2,dist,freqmin,freqmax))
        plt.xlabel('time [s]')
        plt.ylabel('days')
        plt.show()
    
    else:
        #------when receiver info is not known---------
        with pyasdf.ASDFDataSet(afiles[0],mode='r') as ds:
            rlist = ds.auxiliary_data[source].list()E
            dist = ds.auxiliary_data[source][rlist[0]].parameters['dist']
        
        for recever in rlist:
            net2 = recever.split('s')[0]
            sta2 = recever.split('s')[1]
            comp2= recever.split('s')[2]

            #---loop through each day------
            for ii in range(len(afiles)):
                
                #--maximum 50 days one plot----
                if ii%50==0:
                    plt.figure(figsize=(9,6))

                with pyasdf.ASDFDataSet(afiles[ii],mode='r') as ds:
                    iday = afiles[ii].split('/')[-1].split('.')[0]
                    data = ds.auxiliary_data[source][recever].data[:]
                    data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                    #--normalize the data----
                    data = data/max(data)
                    npts = len(data)
                    #----make index----
                    indx0 = npts//2
                    tindx = int(maxlag/dt)
                           
                    plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+ii*2,'k-',linewidth=0.5)
                    plt.text(maxlag*0.9,ii*2,iday,fontsize=6)
            plt.grid(True)
            plt.plot([0,0],[0,ii*2],'r--',linewidth=1.5)
            plt.title('%s_%s_%s_%s dist %6.1f @%4.1f-%4.1f Hz' % (sta1,comp1,sta2,comp2,dist,freqmin,freqmax))
            plt.xlabel('time [s]')
            plt.ylabel('days')
            plt.show()


def compare_c2_c3_waveforms(c2file,c3file,maxlag,c2_maxlag,dt):
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