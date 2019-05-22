import os
import glob
import obspy
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
4). plot_cc_withtime(ccfdir,stackdir,freqmin,freqmax,net1,sta1,comp1,net2=None,sta2=None,comp2=None)
    to plot the ccfs of one station-pair at different days
5). plot_ZH_pmotion(sfile,freqmin,freqmax,net1,sta1,net2=None,sta2=None) to show the R-Z cross components
    as well as the particle motion to identify body-wave components
6). plot_multi_freq
7). compare_c2_c3_waveforms(c2file,c3file,maxlag,c2_maxlag,dt)
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
            std   = ds.auxiliary_data[icomp][iday].parameters['std']

            #---look at hourly----
            if spect.ndim==2:
                nfft = spect.shape[1]*2
                freq  = np.fft.fftfreq(nfft,dt)[0:nfft//2]
                for ii in range(spect.shape[0]):
                    waveform = np.real(np.fft.irfft(spect[ii])[0:nfft])
                    plt.subplot(211)
                    plt.loglog(freq,np.abs(spect[ii]),'k-')
                    plt.title('station %s %s @ %s; std %4.1f' % (sta,icomp,iday,std[ii]))
                    plt.subplot(212)
                    plt.plot(np.arange(0,nfft-2)*dt,waveform,'k-')
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


def plot_spectrum2(sfile,iday,icomp,freqmin,freqmax):
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
                plt.title('station %s %s @ %s' % (sta,icomp,iday))
                for ii in range(spect.shape[0]):
                    waveform = np.real(np.fft.irfft(spect[ii])[0:nfft])
                    waveform = bandpass(waveform,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                    waveform = waveform*0.5/max(waveform)
                    plt.plot(np.arange(0,nfft)*dt,waveform+ii,'k-')
                    #plt.plot([0,nfft*dt],[ii,ii],'r--',linewidth=0.2)
                plt.show()


def plot_moveout(sfile,freqmin,freqmax,net1=None,sta1=None,comp1=None):            
    '''
    this script plots the cross-correlation functions for the station pair of sta1-sta2
    and component 1 and component 2 filtered at freq bands of freqmin-freqmax. if no 
    station is provided, it plots the move-out for each virtual source

    usage: plot_moveout('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/CCF/2010_12_18.h5',0.1,0.3) or
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
            rlist = ds.auxiliary_data[isource].list()
            mdist = 0
            plt.figure(figsize=(9,6))

            #---loop through all receivers-----
            for ireceiver in rlist:
                sta2 = ireceiver.split('s')[1]
                comp2 = ireceiver.split('s')[2]
                #--plot all 3 cross-component---
                if comp2 == 'E':
                    color = 'r-'
                elif comp2 == 'N':
                    color = 'g-'
                else:
                    color = 'b-'

                #---------read parameters and copy data-----------
                dist = ds.auxiliary_data[isource][ireceiver].parameters['dist']
                data = ds.auxiliary_data[isource][ireceiver].data[:]
                data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                data = data/max(data)
                npts = len(data)
                indx0 = npts//2
                tindx = int(maxlag/dt)
                if tindx>indx0:
                    raise ValueError('tindx larger than indx0')
                plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+dist,color,linewidth=0.8)
                plt.title('%s %s filtered @%4.1f-%4.1f Hz' % (sta1,comp1,freqmin,freqmax))
                plt.xlabel('time (s)')
                plt.ylabel('offset (km)')
                plt.text(maxlag*0.9,dist+0.5,sta2,fontsize=6)

                #----use to plot o times------
                if mdist < dist:
                    mdist = dist
            plt.plot([0,0],[0,mdist],'k--',linewidth=2)
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
                if comp2 == 'E':
                    color = 'r-'
                elif comp2 == 'N':
                    color = 'g-'
                else:
                    color = 'b-'

                #------------get parameters and copy data-------------
                dist = ds.auxiliary_data[isource][ireceiver].parameters['dist']
                data = ds.auxiliary_data[isource][ireceiver].data[:]
                data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                data = data/max(data)
                npts = len(data)
                indx0 = npts//2
                tindx = int(maxlag/dt)
                if tindx>indx0:
                    raise ValueError('tindx larger than indx0')
                plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+dist,color,linewidth=0.8)
                plt.title('%s %s filtered @%4.1f-%4.1f Hz' % (sta1,comp1,freqmin,freqmax))
                plt.xlabel('time (s)')
                plt.ylabel('offset (km)')
                plt.text(maxlag*0.9,dist+0.5,sta2,fontsize=6)

                if mdist < dist:
                        mdist = dist
            plt.plot([0,0],[0,mdist],'k--',linewidth=2)
            plt.legend(['E','N','Z'],loc='upper right')
            plt.show()


def plot_moveout_stack(sdir,freqmin,freqmax,ccomp,maxlag=None,tag=None):            
    '''
    this script plots the cross-correlation functions for the station pair of sta1-sta2
    and component 1 and component 2 filtered at freq bands of freqmin-freqmax. if no 
    station is provided, it plots the move-out for each virtual source

    usage: plot_moveout('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.AYHM',0.1,0.3,'ZZ') 
    '''
    
    #---basic parameters----
    afiles = sorted(glob.glob(os.path.join(sdir,'*.h5')))
    if not maxlag:
        maxlag = 100
    if not tag:
        tag = 'Allstacked'

    #-----------get all tags-------------
    with pyasdf.ASDFDataSet(afiles[0],mode='r') as ds:
        tags = ds.auxiliary_data.list()
        dt = ds.auxiliary_data['Allstacked']['ZZ'].parameters['dt']
        tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt

    mdist=0
    #----all stacked days-----
    if tag in tags:
        plt.figure(figsize=(9,6))

        #-------all receivers--------
        for ii in range(len(afiles)):
            receiver = afiles[ii].split('_')[-1]
            with pyasdf.ASDFDataSet(afiles[ii],mode='r') as ds:
                slist = ds.auxiliary_data.list()
                
                #----check that day exist---
                if tag in slist:
                    
                    #---------read parameters and copy data-----------
                    dist = ds.auxiliary_data[tag][ccomp].parameters['dist']
                    data = ds.auxiliary_data[tag][ccomp].data[:]
                    data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                    data = data/max(data)
                    npts = len(data)
                    indx0 = npts//2
                    tindx = int(maxlag/dt)
                    if tindx>indx0:
                        raise ValueError('tindx larger than indx0')
                    plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+dist,'k',linewidth=0.8)
                    plt.title('%s %s %s filtered @%4.1f-%4.1f Hz' % (sdir.split('/')[-1],tag,ccomp,freqmin,freqmax))
                    plt.xlabel('time (s)')
                    plt.ylabel('offset (km)')
                    plt.text(maxlag*0.9,dist+0.5,receiver,fontsize=6)

                    #----use to plot o times------
                    if mdist < dist:
                        mdist = dist
        plt.plot([0,0],[0,mdist],'r--',linewidth=1)
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


def plot_cc_withtime(ccfdir,freqmin,freqmax,net1,sta1,comp1,net2=None,sta2=None,comp2=None,stackdir=None):
    '''
    plot the filtered cross-correlation functions between station-pair sta1-sta2
    for all of the available days stored in ccfdir

    example: plot_cc_withtime('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/CCF',1,5,'E','AYHM','Z') or 
    plot_cc_withtime('Mesonet_BW/CCF',1,5,'E','AYHM','Z','Mesonet_BW/STACK/AYHM','E','BKKM','Z')
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
            if ii%60==0:
                plt.figure(figsize=(9,6))

            with pyasdf.ASDFDataSet(afiles[ii],mode='r') as ds:
                #print(afiles[ii],source,recever)
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
        if stackdir:
            stackfile = os.path.join(stackdir,net1+'.'+sta1+'_'+net2+'.'+sta2+'.h5')
            #----plot stacked one----
            with pyasdf.ASDFDataSet(stackfile,mode='r') as stack_ds:
                ccomp = comp1[-1]+comp2[-1]
                tags = 'Allstacked'
                data = stack_ds.auxiliary_data[tags][ccomp].data[:]
                data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                data = data/max(data)
                indx0= len(data)//2
                plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+ii*2+2,'b-',linewidth=1)
                plt.plot(tt,data[indx0-tindx:indx0+tindx+1]-2,'b-',linewidth=1)
        #---------highlight zero time------------
        plt.plot([0,0],[0,ii*2],'r--',linewidth=1.5)
        plt.title('%s_%s_%s_%s dist %6.1f @%4.1f-%4.1f Hz' % (sta1,comp1,sta2,comp2,dist,freqmin,freqmax))
        plt.xlabel('time [s]')
        plt.ylabel('days')
        plt.show()
    
    else:
        #------when receiver info is not known---------
        with pyasdf.ASDFDataSet(afiles[0],mode='r') as ds:
            rlist = ds.auxiliary_data[source].list()
        
        for recever in rlist:
            net2 = recever.split('s')[0]
            sta2 = recever.split('s')[1]
            comp2= recever.split('s')[2]

            #---loop through each day------
            for ii in range(len(afiles)):
                
                #--maximum 50 days one plot----
                if ii%60==0:
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

            if stackdir:
                stackfile = os.path.join(stackdir,net1+'.'+sta1+'_'+net2+'.'+sta2+'.h5')
                #----plot stacked one----
                with pyasdf.ASDFDataSet(stackfile,mode='r') as stack_ds:
                    ccomp = comp1[-1]+comp2[-1]
                    tags = 'Allstacked'
                    data = stack_ds.auxiliary_data[tags][ccomp].data[:]
                    data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                    data = data/max(data)
                    indx0= len(data)//2
                    plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+ii*2+2,'b-',linewidth=1)
                    plt.plot(tt,data[indx0-tindx:indx0+tindx+1]-2,'b-',linewidth=1)
            #---------highlight zero time------------
            plt.grid(True)
            plt.plot([0,0],[-1,ii*2+3],'r--',linewidth=1.5)
            plt.title('%s_%s_%s_%s, dist:%6.1fkm @%4.1f-%4.1f Hz' % (sta1,comp1,sta2,comp2,dist,freqmin,freqmax))
            plt.xlabel('time [s]')
            plt.ylabel('days')
            plt.show()


def plot_cc_withtime_stack(ccffile,freqmin,freqmax,ccomp,maxlag=None):
    '''
    plot the filtered cross-correlation functions between station-pair sta1-sta2
    for all of the available days stored in ccfdir

    example: plot_cc_withtime('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.AYHM/E.AYHM_E.ENZM.h5',1,5,'ZZ',50) 
    '''
    #---basic parameters----
    if not maxlag:
        maxlag = 100

    #-----check whether file exists------
    if os.path.isfile(ccffile):
        with pyasdf.ASDFDataSet(ccffile,mode='r') as ds:
            dt = ds.auxiliary_data['Allstacked']['ZZ'].parameters['dt']
            tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt
            dist = ds.auxiliary_data['Allstacked']['ZZ'].parameters['dist']

            #-----loop through each day-----
            slist = ds.auxiliary_data.list()

            for ii in range(len(slist)-1):
                if ii%60==0:
                    plt.figure(figsize=(9,6))

                iday = slist[ii]

                #-----in case there is no data on that day-----
                try:
                    data = ds.auxiliary_data[iday][ccomp].data[:]
                    data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
                    
                    #--normalize the data----
                    data = data/max(data)

                except Exception:
                    data = np.zeros(tt.shape,dtype=np.float32)
                
                npts = len(data)                
                #----make index----
                indx0 = npts//2
                tindx = int(maxlag/dt)
                if ii==0:
                    color = 'b-'
                else:
                    color = 'k-'
                plt.plot(tt,data[indx0-tindx:indx0+tindx+1]+ii*2,color,linewidth=0.5)
                plt.text(maxlag*0.7,ii*2,iday,fontsize=6)

            plt.grid(True)

            #---------highlight zero time------------
            plt.plot([0,0],[0,ii*2],'r--',linewidth=1.5)
            plt.title('%s %s, dist:%6.1fkm @%4.1f-%4.1f Hz' % (ccffile.split('/')[-1],ccomp,dist,freqmin,freqmax))
            plt.xlabel('time [s]')
            plt.ylabel('days')
            plt.show()
    

def plot_ZH_pmotion(sfile,freqmin,freqmax,net1,sta1,net2=None,sta2=None):
    '''
    plot the 4 component of ccfs where Rayleigh wave might dominant, and use the particle
    motion to identify the surface wave component

    example: plot_ZH_pmotion('2010_12_16.h5',0.5,1,'E','ENZM')
    '''
    #---basic parameters----
    maxlag = 100
    dt = 0.05
    tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt
    chan = ['ZZ','ZR','RZ','RR']

    #------set path and type-------------
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()

    if net2 and sta2:
        isource = net1+'s'+sta1+'s'+net2+'s'+sta2
        if isource in slist:
            dist = ds.auxiliary_data[isource][chan[0]].parameters['dist']
            #-----------component 1--------------
            data = ds.auxiliary_data[isource][chan[0]].data[:]
            data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            #--normalize the data----
            data = data/max(data)
            npts = len(data)
            indx0 = npts//2
            tindx = int(maxlag/dt)
            plt.figure(figsize=(14,8))
            plt.subplot(421)
            plt.plot(tt,data[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
            plt.text(maxlag*0.9,0.8,'ZZ',fontsize=10)
            plt.title('%s, dist:%4.1fkm @ %4.1f-%4.1f Hz' % (isource,dist,freqmin,freqmax))

            #---------component 2------------
            data1 = ds.auxiliary_data[isource][chan[1]].data[:]
            data1 = bandpass(data1,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            data1 = data1/max(data1)
            plt.subplot(423)
            plt.plot(tt,data1[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
            plt.text(maxlag*0.9,0.8,'ZR',fontsize=10)

            #---------component 3---------------
            data2 = ds.auxiliary_data[isource][chan[2]].data[:]
            data2 = bandpass(data2,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            data2 = data2/max(data2)
            plt.subplot(425)
            plt.plot(tt,data2[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
            plt.text(maxlag*0.9,0.8,'RZ',fontsize=10)

            #--------component 4-------------
            data3 = ds.auxiliary_data[isource][chan[3]].data[:]
            data3 = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            data3 = data3/max(data3)
            plt.subplot(427)
            plt.plot(tt,data3[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
            plt.text(maxlag*0.9,0.8,'RR',fontsize=10)

            #---------particle motion---------
            tindx = int(20/dt)
            plt.subplot(243)
            plt.plot(data1[indx0:indx0+tindx+1],data[indx0:indx0+tindx+1],'k-',linewidth=0.5)
            plt.xlabel('ZR');plt.ylabel('ZZ')

            plt.subplot(244)
            plt.plot(data2[indx0:indx0+tindx+1],data[indx0:indx0+tindx+1],'k-',linewidth=0.5)
            plt.xlabel('RZ');plt.ylabel('ZZ')

            plt.subplot(247)
            plt.plot(data3[indx0:indx0+tindx+1],data2[indx0:indx0+tindx+1],'k-',linewidth=0.5)
            plt.xlabel('RR');plt.ylabel('RZ')
            
            plt.subplot(248)
            plt.plot(data3[indx0:indx0+tindx+1],data1[indx0:indx0+tindx+1],'k-',linewidth=0.5)
            plt.xlabel('RR');plt.ylabel('ZR')
            plt.show()
        else:
            raise ValueError('station pair not exist in %s' % sfile)
    
    else:
        #------loop through each station-pair--------
        for isource in slist:
            dist = ds.auxiliary_data[isource][chan[0]].parameters['dist']
            #-----------component 1--------------
            data = ds.auxiliary_data[isource][chan[0]].data[:]
            data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            #--normalize the data----
            data = data/max(data)
            npts = len(data)
            indx0 = npts//2
            tindx = int(maxlag/dt)
            plt.figure(figsize=(14,8))
            plt.subplot(421)
            plt.plot(tt,data[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
            plt.text(maxlag*0.9,0.8,'ZZ',fontsize=10)
            plt.title('%s, dist:%4.1fkm @ %4.1f-%4.1f Hz' % (isource,dist,freqmin,freqmax))

            #---------component 2------------
            data1 = ds.auxiliary_data[isource][chan[1]].data[:]
            data1 = bandpass(data1,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            data1 = data1/max(data1)
            plt.subplot(423)
            plt.plot(tt,data1[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
            plt.text(maxlag*0.9,0.8,'ZR',fontsize=10)

            #---------component 3---------------
            data2 = ds.auxiliary_data[isource][chan[2]].data[:]
            data2 = bandpass(data2,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            data2 = data2/max(data2)
            plt.subplot(425)
            plt.plot(tt,data2[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
            plt.text(maxlag*0.9,0.8,'RZ',fontsize=10)

            #--------component 4-------------
            data3 = ds.auxiliary_data[isource][chan[3]].data[:]
            data3 = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            data3 = data3/max(data3)
            plt.subplot(427)
            plt.plot(tt,data3[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
            plt.text(maxlag*0.9,0.8,'RR',fontsize=10)

            #---------particle motion---------
            tindx = int(20/dt)
            plt.subplot(243)
            plt.plot(data1[indx0:indx0+tindx+1],data[indx0:indx0+tindx+1],'k-',linewidth=0.5)
            plt.xlabel('ZR');plt.ylabel('ZZ')

            plt.subplot(244)
            plt.plot(data2[indx0:indx0+tindx+1],data[indx0:indx0+tindx+1],'k-',linewidth=0.5)
            plt.xlabel('RZ');plt.ylabel('ZZ')

            plt.subplot(247)
            plt.plot(data3[indx0:indx0+tindx+1],data2[indx0:indx0+tindx+1],'k-',linewidth=0.5)
            plt.xlabel('RR');plt.ylabel('RZ')
            
            plt.subplot(248)
            plt.plot(data3[indx0:indx0+tindx+1],data1[indx0:indx0+tindx+1],'k-',linewidth=0.5)
            plt.xlabel('RR');plt.ylabel('ZR')
            plt.show()

def plot_ZH_pmotion_stack(sfile,freqmin,freqmax,t0,t1,maxlag,tags=None):
    '''
    plot the 4 component of ccfs where Rayleigh wave might dominant, and use the particle
    motion to identify the surface wave component

    example: plot_ZH_pmotion('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.AYHM/E.AYHM_E.BKKM.h5',0.5,1,-15,-5,50)
    '''
    #---basic parameters----
    #maxlag = 100
    #dt = 0.05
    chan = ['ZZ','ZR','RZ','RR']

    #------set path and type-------------
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()
    dt = ds.auxiliary_data[slist[0]][chan[0]].parameters['dt']
    tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt

    if not tags:
        tags = "Allstacked"
    
    if tags not in slist:
        raise ValueError('tags %s not in the file %s' % (tags,sfile))
        
    dist = ds.auxiliary_data[tags][chan[0]].parameters['dist']
    #-----------component 1--------------
    data = ds.auxiliary_data[tags][chan[0]].data[:]
    data = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
    #--normalize the data----
    data = data/max(data)
    npts = len(data)
    indx0 = npts//2
    tindx = int(maxlag/dt)
    plt.figure(figsize=(14,8))
    plt.subplot(421)
    plt.plot(tt,data[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
    plt.text(maxlag*0.9,0.8,'ZZ',fontsize=10)
    plt.title('%s, dist:%4.1fkm @ %4.1f-%4.1f Hz' % (tags,dist,freqmin,freqmax))

    #---------component 2------------
    data1 = ds.auxiliary_data[tags][chan[1]].data[:]
    data1 = bandpass(data1,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
    data1 = data1/max(data1)
    plt.subplot(423)
    plt.plot(tt,data1[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
    plt.text(maxlag*0.9,0.8,'ZR',fontsize=10)

    #---------component 3---------------
    data2 = ds.auxiliary_data[tags][chan[2]].data[:]
    data2 = bandpass(data2,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
    data2 = data2/max(data2)
    plt.subplot(425)
    plt.plot(tt,data2[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
    plt.text(maxlag*0.9,0.8,'RZ',fontsize=10)

    #--------component 4-------------
    data3 = ds.auxiliary_data[tags][chan[3]].data[:]
    data3 = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
    data3 = data3/max(data3)
    plt.subplot(427)
    plt.plot(tt,data3[indx0-tindx:indx0+tindx+1],'k-',linewidth=0.5)
    plt.text(maxlag*0.9,0.8,'RR',fontsize=10)

    #---------particle motion---------
    tindx1 = int(t0/dt)
    tindx2 = int(t1/dt)
    indx1 = indx0+tindx1
    indx2 = indx0+tindx2+1
    t = np.linspace(indx1*dt,indx2*dt,indx2-indx1)

    plt.subplot(243)
    #plt.plot(data1[indx1:indx2],data[indx1:indx2],'k-',linewidth=0.5)
    plt.scatter(data1[indx1:indx2],data[indx1:indx2],c=t,s=5,cmap='jet')
    plt.xlabel('ZR');plt.ylabel('ZZ')

    plt.subplot(244)
    #plt.plot(data2[indx1:indx2],data[indx1:indx2],'k-',linewidth=0.5)
    plt.scatter(data2[indx1:indx2],data[indx1:indx2],c=t,s=5,cmap='jet')
    plt.xlabel('RZ');plt.ylabel('ZZ')
    #fig.colorbar(im,ax=ax)

    plt.subplot(247)
    #plt.plot(data3[indx1:indx2],data2[indx1:indx2],'k-',linewidth=0.5)
    plt.scatter(data3[indx1:indx2],data2[indx1:indx2],c=t,s=5,cmap='jet')
    plt.xlabel('RR');plt.ylabel('RZ')
    
    plt.subplot(248)
    #plt.plot(data3[indx1:indx2],data1[indx1:indx2],'k-',linewidth=0.5)
    plt.scatter(data3[indx1:indx2],data1[indx1:indx2],c=t,s=5,cmap='jet')
    plt.xlabel('RR');plt.ylabel('ZR')
    #fig.colorbar(im,ax=ax)
    plt.show()


def plot_multi_freq(sfile,freqmin,freqmax,nfreq,net1,sta1,comp1,net2,sta2,comp2):
    '''
    plot the stacked ccfs for sta1-sta2 at multi-frequency bands between freqmin-freqmax. 
    this may be useful to show the dispersive property of the surface waves, which could 
    be used together with plot_ZH_pmotion to identify surface wave components

    example:
    '''
    #---basic parameters----
    maxlag = 100

    #------set path and type-------------
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()

    #-----check tags information------
    tags = net1+'s'+sta1+'s'+comp1
    path = net2+'s'+sta2+'s'+comp2

    if tags not in slist:
        raise ValueError('tags %s not in the sfile %s' % (tags,sfile))
    
    #--------read the data and parameters------------
    parameters = ds.auxiliary_data[tags][path].parameters
    corr = ds.auxiliary_data[tags][path].data[:]
    sampling_rate = int(1/parameters['dt'])
    npts = int(2*sampling_rate*parameters['lag'])
    indx = npts//2
    tt = np.arange(-maxlag*sampling_rate, maxlag*sampling_rate+1)/sampling_rate
    
    #------make frequency information-----
    freq = np.zeros(nfreq,dtype=np.float32)
    step = (np.log(freqmin)-np.log(freqmax))/(nfreq-1)
    
    for ii in range(nfreq):
        freq[ii]=np.exp(np.log(freqmin)-ii*step)

    indx0 = maxlag*sampling_rate
    #----loop through each freq-----
    for ii in range(1,nfreq-1):
        if ii==1:
            plt.figure(figsize=(9,6))

        f1 = freq[ii-1]
        f2 = freq[ii+1]
        ncorr = bandpass(corr,f1,f2,sampling_rate,corners=4,zerophase=True)
        ncorr = ncorr/max(ncorr)

        #------plot the signals-------
        plt.plot(tt,ncorr[indx-indx0:indx+indx0+1]+ii,'k-',linewidth=0.6)
        ttext = '{0:4.2f}-{1:4.2f} Hz'.format(f1,f2)
        plt.text(maxlag*0.9,ii+0.3,ttext,fontsize=6)
        if ii==1:
            plt.title('%s %s and %s %s cross component' % (sta1,comp1,sta2,comp2))
            plt.xlabel('time [s]')
            plt.ylabel('waveform #')
    plt.grid(True)
    plt.show()


def plot_multi_freq_stack(sfile,freqmin,freqmax,nfreq,ccomp,tags=None):
    '''
    plot the stacked ccfs for sta1-sta2 at multi-frequency bands between freqmin-freqmax. 
    this may be useful to show the dispersive property of the surface waves, which could 
    be used together with plot_ZH_pmotion to identify surface wave components

    example: plot_multi_freq_stack('/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK/E.AYHM/E.AYHM_E.BKKM.h5',0.08,6,15,'RR')
    '''
    #---basic parameters----
    maxlag = 100

    #------set path and type-------------
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    slist = ds.auxiliary_data.list()

    #-----check tags information------
    if not tags:
        tags = "Allstacked"
    
    if tags not in slist:
        raise ValueError('tags %s not in the file %s' % (tags,sfile))
    
    #--------read the data and parameters------------
    parameters = ds.auxiliary_data[tags][ccomp].parameters
    corr = ds.auxiliary_data[tags][ccomp].data[:]
    sampling_rate = int(1/parameters['dt'])
    npts = int(2*sampling_rate*parameters['lag'])
    indx = npts//2
    tt = np.arange(-maxlag*sampling_rate, maxlag*sampling_rate+1)/sampling_rate
    
    #------make frequency information-----
    freq = np.zeros(nfreq,dtype=np.float32)
    step = (np.log(freqmin)-np.log(freqmax))/(nfreq-1)
    
    for ii in range(nfreq):
        freq[ii]=np.exp(np.log(freqmin)-ii*step)

    indx0 = maxlag*sampling_rate
    #----loop through each freq-----
    for ii in range(1,nfreq-1):
        if ii==1:
            plt.figure(figsize=(9,6))

        f1 = freq[ii-1]
        f2 = freq[ii+1]
        ncorr = bandpass(corr,f1,f2,sampling_rate,corners=4,zerophase=True)
        ncorr = ncorr/max(ncorr)

        #------plot the signals-------
        plt.plot(tt,ncorr[indx-indx0:indx+indx0+1]+ii,'k-',linewidth=0.6)
        ttext = '{0:4.2f}-{1:4.2f} Hz'.format(f1,f2)
        plt.text(maxlag*0.9,ii+0.3,ttext,fontsize=6)
        if ii==1:
            plt.title('%s at %s component' % (sfile.split('/')[-1],ccomp))
            plt.xlabel('time [s]')
            plt.ylabel('waveform #')
    plt.grid(True)
    plt.show()
    del ds


def plot_SNR_stackday(sfile1,sfile2,sfile3,sfile4,sfile5,ccomp,lag):
    '''
    extract the paramters of SNR for each stacked trace to get a statistical sense of
    how SNR varies with stacking days for one station-pair

    example: plot_SNR_distance('stack_1day/E.AYHM/E.AYHM_E.ENZM.h5','stack_2day/E.AYHM/E.AYHM_E.ENZM.h5',\
    'stack_3day/E.AYHM/E.AYHM_E.ENZM.h5','stack_5day/E.AYHM/E.AYHM_E.ENZM.h5','stack_10day/E.AYHM/E.AYHM_E.ENZM.h5','ZZ',-1)
    '''
    ds1 = pyasdf.ASDFDataSet(sfile1,mode='r')
    ds2 = pyasdf.ASDFDataSet(sfile2,mode='r')
    ds3 = pyasdf.ASDFDataSet(sfile3,mode='r')
    ds4 = pyasdf.ASDFDataSet(sfile4,mode='r')
    ds5 = pyasdf.ASDFDataSet(sfile5,mode='r')
    if lag==0:
        snr_para = 'ssnr'
    elif lag==-1:
        snr_para = 'nsnr'
    else:
        snr_para = 'psnr'

    slists = ds1.auxiliary_data.list()
    freq   = ds1.auxiliary_data[slists[0]][ccomp].parameters['freq']
    for data_type in slists[1:2]:
        snr = ds1.auxiliary_data[data_type][ccomp].parameters[snr_para]
        plt.scatter(freq,snr,marker='^',s=30)
    del ds1
    
    slists = ds2.auxiliary_data.list()
    freq   = ds2.auxiliary_data[slists[0]][ccomp].parameters['freq']
    for data_type in slists[1:2]:
        snr = ds2.auxiliary_data[data_type][ccomp].parameters[snr_para]
        plt.scatter(freq,snr,marker='o',s=30)
    del ds2

    slists = ds3.auxiliary_data.list()
    freq   = ds3.auxiliary_data[slists[0]][ccomp].parameters['freq']
    for data_type in slists[1:2]:
        snr = ds3.auxiliary_data[data_type][ccomp].parameters[snr_para]
        plt.scatter(freq,snr,marker='s',s=30)
    del ds3
    
    slists = ds4.auxiliary_data.list()
    freq   = ds4.auxiliary_data[slists[0]][ccomp].parameters['freq']
    for data_type in slists[1:2]:
        snr = ds4.auxiliary_data[data_type][ccomp].parameters[snr_para]
        plt.scatter(freq,snr,marker='*',s=30)
    del ds4

    slists = ds5.auxiliary_data.list()
    freq   = ds5.auxiliary_data[slists[0]][ccomp].parameters['freq']
    for data_type in slists[1:2]:
        snr = ds5.auxiliary_data[data_type][ccomp].parameters[snr_para]
        plt.scatter(freq,snr,marker='x',s=30)
    snr = ds5.auxiliary_data[slists[0]][ccomp].parameters[snr_para]
    plt.scatter(freq,snr,marker='D',s=30)
    del ds5
    plt.legend(['1day','2day','3day','5day','10day','20day'],loc='upper right')
    
    plt.title([sfile1.split('/')[-1],ccomp,snr_para])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('SNR')
    plt.show()


def plot_SNR_distance(sdir,ccomp,lag):
    '''
    show the statistic of how SNR varies with distance for the stacked cross correlations
    '''
    #------all station pairs-------
    sfiles = glob.glob(os.path.join(sdir,'*.h5'))
    tags = 'Allstacked'
    if lag==0:
        snr_para = 'ssnr'
    elif lag==-1:
        snr_para = 'nsnr'
    else:
        snr_para = 'psnr'
    
    #-----read in freq--------
    ds = pyasdf.ASDFDataSet(sfiles[0],mode='r')
    freq = ds.auxiliary_data[tags][ccomp].parameters['freq']
    print(freq)
    del ds

    #---loop through each station-pair----
    for sfile in sfiles:
        with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
            dist = ds.auxiliary_data[tags][ccomp].parameters['dist']
            snr = ds.auxiliary_data[tags][ccomp].parameters[snr_para]
            plt.subplot(331)
            plt.scatter(dist,snr[1],marker='o',s=25,c=0)
            plt.subplot(332)
            plt.scatter(dist,snr[2],marker='o',s=25,c=0)
            plt.subplot(333)
            plt.scatter(dist,snr[4],marker='o',s=25,c=0)
            plt.subplot(334)
            plt.scatter(dist,snr[6],marker='o',s=25,c=0)
            plt.subplot(335)
            plt.scatter(dist,snr[7],marker='o',s=25,c=0)
            plt.subplot(336)
            plt.scatter(dist,snr[8],marker='o',s=25,c=0)
            plt.subplot(337)
            plt.scatter(dist,snr[9],marker='o',s=25,c=0)
            plt.subplot(338)
            plt.scatter(dist,snr[10],marker='o',s=25,c=0)
            plt.subplot(339)
            plt.scatter(dist,snr[11],marker='o',s=25,c=0)
    plt.subplot(331)
    plt.ylabel('SNR')
    plt.subplot(334)
    plt.ylabel('SNR')
    plt.subplot(337)
    plt.ylabel('SNR')
    plt.xlabel('distance [km]')
    plt.subplot(338)
    plt.xlabel('distance [km]')
    plt.subplot(339)
    plt.xlabel('distance [km]')
    plt.subplot(332)
    plt.title([sdir.split('/')[-1],dist,'km'])
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
