from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
import noise_module
import numpy as np 
import pyasdf
import scipy
import glob
import os

#----the path for the data---
'''
this is not important at this moment

rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
STACKDIR = os.path.join(rootpath,'STACK')
sta = glob.glob(os.path.join(STACKDIR,'E.*'))
nsta = len(sta)
'''

#----some common variables-----
epsilon = 0.1
nbtrial = 50
tmin = -30
tmax = -15
fmin = 0.5
fmax = 1
comp = 'TT'
maxlag = 100

#----for plotting-----
Mdate = 12
NSV   = 2

h5file = '/Users/chengxin/Documents/Harvard/Kanto/data/STACK/E.AYHM/E.AYHM_E.ENZM.h5'

#-------open ASDF file to read data-----------
with pyasdf.ASDFDataSet(h5file,mode='r') as ds:
    slist = ds.auxiliary_data.list()

    #------loop through the reference waveforms------
    if slist[0]== 'Allstacked':

        #------useful parameters from ASDF file------
        rlist = ds.auxiliary_data[slist[0]].list()
        if comp not in rlist:
            raise IOError('no component in the data %s' %(h5file.split('/')[-1]))

        delta = ds.auxiliary_data[slist[0]][comp].parameters['dt']
        lag   = ds.auxiliary_data[slist[0]][comp].parameters['lag']

        #--------index for the data---------
        indx1 = int((lag-maxlag)/delta)
        indx2 = int((lag+maxlag)/delta)
        t     = np.arange(0,indx2-indx1+1,400)*delta-maxlag
        Ntau  = int(np.ceil(np.min([1/fmin,1/fmax])/delta)) + 15

        #----------plot waveforms-----------
        ndays = len(slist)
        data  = np.zeros((ndays,indx2-indx1+1),dtype=np.float32)
        for ii in range(ndays):
            trlist = ds.auxiliary_data[slist[ii]].list()
            if comp in rlist:
                tdata = ds.auxiliary_data[slist[ii]][comp].data[indx1:indx2+1]
                data[ii,:] = bandpass(tdata,fmin,fmax,int(1/delta),corners=4, zerophase=True)
                data[ii,:] = data[ii,:]/max(data[ii,:])

        fig,ax = plt.subplots(2,sharex=True)
        ax[0].matshow(data,cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
        new = noise_module.NCF_denoising(data,np.min([Mdate,data.shape[0]]),Ntau,NSV)

        ax[1].matshow(new/new.max(),cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
        ax[0].set_title('Filterd Cross-Correlations')
        ax[1].set_title('Denoised Cross-Correlations')
        ax[0].xaxis.set_visible(False)
        ax[1].set_xticks(t)
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].set_xlabel('Time [s]')
        plt.show()
        #outfname = directory + '/' + 'Fig_dv_' + virt + '.pdf'
        #fig.savefig(outfname, format='pdf', dpi=400)
        #plt.close(fig)

        #-------parameters for doing stretching-------
        tvec = np.arange(tmin,tmax,delta)
        ref  = data[0,:]
        window = np.arange(int(tmin/delta),int(tmax/delta))+int(maxlag/delta)

        #--------parameters to store dv/v and cc--------
        dv1 = np.zeros(ndays,dtype=np.float32)
        cc = np.zeros(ndays,dtype=np.float32)
        cdp = np.zeros(ndays,dtype=np.float32)
        error1 = np.zeros(ndays,dtype=np.float32)
        dv2 = np.zeros(ndays,dtype=np.float32)
        error2 = np.zeros(ndays,dtype=np.float32)
        moving_window_length = 3*int(1/fmin)
        slide_step = 0.4*moving_window_length

        #------loop through the reference waveforms------
        for ii in range(1,ndays):
            cur = data[ii,:]

            #----plug in the stretching function-------
            [dv1[ii], cc[ii], cdp[ii], error1[ii]] = noise_module.Stretching_current(ref, cur, tvec, -epsilon, epsilon, nbtrial, window, fmin, fmax, tmin, tmax)
            [dv2[ii], error2[ii]] = noise_module.mwcs_dvv(ref, cur, moving_window_length, slide_step, int(1/delta), window, fmin, fmax, tmin)

        #----plot the results------
        plt.subplot(311)
        plt.title(h5file.split('/')[-1])
        plt.plot(dv1,'r-');plt.plot(dv2,'b-')
        plt.ylabel('dv/v [%]')
        plt.subplot(312)
        plt.plot(cc,'r-');plt.plot(cdp,'b-')
        plt.ylabel('cc')
        plt.subplot(313)
        plt.plot(error1,'r-');plt.plot(error2,'b-')
        plt.xlabel('days');plt.ylabel('errors [%]')
        plt.show()
