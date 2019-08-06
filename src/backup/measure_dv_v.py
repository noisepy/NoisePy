from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
import noise_module
import numpy as np 
import matplotlib
import pyasdf
import scipy
import glob
import sys
import os

'''
a compilation of 6 different methods to estimate dv/v values for stacked waveforms 
stored in ASDF files. the 6 methods are 1) stretch, 2) MWCS and 3) DTW. 

update to add the comparision of the resulted dv/v from positive lag, negative lag 
and the averaged waveforms for the strecthing part
'''

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

font = {'family':'normal','weight':'bold','size':10}
matplotlib.rc('font', **font)

#----the path for the data---
rootpath = '/Users/chengxin/Documents/Harvard/code_develop/NoisePy/real_data'
STACKDIR = os.path.join(rootpath,'STACK')
sta = glob.glob(os.path.join(STACKDIR,'N.*'))

#----some common variables-----
epsilon = 0.01              # limit for the dv/v range (*100 to get range in %)
nbtrial = 50                # number of increment of dt [-epsilon,epsilon] for the streching method
tmin = np.abs(50)           # this is the absolute values meaning both lags are used in MWCS
tmax = np.abs(90)           # same as tmin as absolute values here
fmin = 0.3
fmax = 1
comp = 'ZZ'
maxlag  = 100                # maximum window to measure dv/v
stretch = True               # flag for the stretching method
mwcs    = True               # flag for the MWCS method
allstation = True           # make measurement to all stacked data or not
wfilter    = True            # make measurement to the clean waveforms from wiener filter
onelag     = False           # make measurement based on both lags data
start_date = '2010_01_01'    # assume a continuous recording from start to end date with increment of stack-days
end_date   = '2010_01_02'
stack_days = 1


#----parameters for NCF denoising-----
Mdate = 12
NSV   = 2

if not allstation:
    h5files = ['/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK1/E.ABHM/E.ABHM_E.OHSM.h5']
    nsta = len(h5files)
else:
    h5files = sta
    nsta = len(sta)

for ista in range(nsta):
    h5file = h5files[ista]

    #--------assume continous recordings for each stacked segments---------
    tlist = noise_module.get_event_list(start_date,end_date,stack_days)
    tags_allstack = ['Allstacked']
    for ii in range(len(tlist)-1):
        tags_allstack.append('F'+tlist[ii].replace('_','')+'T'+tlist[ii+1].replace('_',''))
    nstacks = len(tags_allstack)

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

            #-------------prepare the data matrix-----------------
            data  = np.zeros((nstacks,indx2-indx1+1),dtype=np.float32)
            flag  = np.zeros(nstacks,dtype=np.int16)

            #------loop through each stacked segment------
            for iseg in range(nstacks):
                ttag = tags_allstack[iseg]

                if ttag in slist:
                    trlist = ds.auxiliary_data[ttag].list()
                    if comp in trlist:
                        
                        tdata = ds.auxiliary_data[ttag][comp].data[indx1:indx2+1]
                        data[iseg,:] = bandpass(tdata,fmin,fmax,int(1/delta),corners=4, zerophase=True)
                        data[iseg,:] = data[iseg,:]/max(data[iseg,:])
                    else:
                        flag[iseg]   = 1
                else:
                    flag[iseg] = 1

            fig,ax = plt.subplots(2,sharex=True)
            ax[0].matshow(data,cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
            new = noise_module.NCF_denoising(data,np.min([Mdate,data.shape[0]]),Ntau,NSV)

            ax[1].matshow(new,cmap='seismic',extent=[-maxlag,maxlag,data.shape[0],1],aspect='auto')
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
            window = np.arange(int(tmin/delta),int(tmax/delta))+int(maxlag/delta)
            if wfilter:
                ref  = new[0,:]
            else:
                ref  = data[0,:]

            #--------parameters to store dv/v and cc--------
            if stretch:
                
                if onelag:
                    dv1 = np.zeros(nstacks,dtype=np.float32)
                    cc = np.zeros(nstacks,dtype=np.float32)
                    cdp = np.zeros(nstacks,dtype=np.float32)
                    error1 = np.zeros(nstacks,dtype=np.float32)
                else:
                    dv1 = np.zeros((nstacks,3),dtype=np.float32)
                    cc = np.zeros((nstacks,3),dtype=np.float32)
                    cdp = np.zeros((nstacks,3),dtype=np.float32)
                    error1 = np.zeros((nstacks,3),dtype=np.float32)
            
            if mwcs:
                dv2 = np.zeros(nstacks,dtype=np.float32)
                error2 = np.zeros(nstacks,dtype=np.float32)
                moving_window_length = 3*int(1/fmin)
                slide_step = 0.2*moving_window_length

                #-----check whether window is long enough-----
                if moving_window_length > 2*(tmax-tmin):
                    raise IOError('the time window is too small for making MWCS')

            #------loop through the reference waveforms------
            for ii in range(1,nstacks):
                if wfilter:
                    cur = new[ii,:]
                else:
                    cur = data[ii,:]
                
                #---for the missing days---
                if not flag[ii]:

                #----plug in the stretching function-------
                    if stretch:

                        #----just on one lag----
                        if onelag:
                            [dv1[ii], cc[ii], cdp[ii], error1[ii]] = noise_module.Stretching_current\
                                (ref, cur, tvec, epsilon, nbtrial, window, fmin, fmax, tmin, tmax)
                        else:
                            [dv1[ii][0], cc[ii][0], cdp[ii][0], error1[ii][0]] = noise_module.Stretching_current\
                                (ref, cur, tvec, epsilon, nbtrial, window, fmin, fmax, tmin, tmax)
                            [dv1[ii][1], cc[ii][1], cdp[ii][1], error1[ii][1]] = noise_module.Stretching_current\
                                (np.flip(ref,axis=0), np.flip(cur,axis=0), tvec, epsilon, nbtrial, window, fmin, fmax, tmin, tmax)
                            [dv1[ii][2], cc[ii][2], cdp[ii][2], error1[ii][2]] = noise_module.Stretching_current\
                                (0.5*(ref+np.flip(ref,axis=0)),0.5*(cur+np.flip(cur,axis=0)), tvec, epsilon, nbtrial, window, fmin, fmax, tmin, tmax)
                    
                    if mwcs:
                        [dv2[ii], error2[ii]] = noise_module.mwcs_dvv1(ref, cur, moving_window_length, slide_step, int(1/delta), window, fmin, fmax, tmin)

                #------set the segments without data to be nan-------
                else:
                    if stretch:
                        if onelag:
                            dv1[ii]=np.nan;cc[ii]=np.nan;cdp[ii]=np.nan;error1[ii]=np.nan
                        else:
                            dv1[ii][0]=np.nan;cc[ii][0]=np.nan;cdp[ii][0]=np.nan;error1[ii][0]=np.nan
                            dv1[ii][1]=np.nan;cc[ii][1]=np.nan;cdp[ii][1]=np.nan;error1[ii][1]=np.nan
                            dv1[ii][2]=np.nan;cc[ii][2]=np.nan;cdp[ii][2]=np.nan;error1[ii][2]=np.nan
                    if mwcs:
                        dv2[ii]=np.nan;error2[ii]=np.nan

            #----plot the results------
            if stretch:
                tt = np.arange(1,nstacks,stack_days)
                if onelag:
                    plt.figure()
                    plt.subplot(211)
                    plt.title(h5file.split('/')[-1])
                    plt.plot(tt,dv1[1:]);plt.errorbar(tt,dv1[1:],yerr=error1[1:]*0.2)
                    plt.ylabel('dv/v [%]');plt.xlabel('days')
                    plt.subplot(212)
                    plt.plot(cc[1:],'r-');plt.plot(cdp[1:],'b-')
                    plt.ylabel('cc');plt.xlabel('days')
                    plt.show()
                else:
                    plt.figure()
                    plt.subplot(311)
                    plt.title(h5file.split('/')[-1])
                    plt.plot(tt,dv1[1:,0],'r-')#;plt.errorbar(tt,dv1[1:,0],yerr=error1[1:,0]*0.1)
                    plt.plot(tt,dv1[1:,1],'g-')#;plt.errorbar(tt,dv1[1:,1],yerr=error1[1:,1]*0.1)
                    plt.plot(tt,dv1[1:,2],'b-')#;plt.errorbar(tt,dv1[1:,2],yerr=error1[1:,2]*0.1)
                    plt.ylabel('dv/v [%]');plt.xlabel('days')
                    plt.legend(['P Lag','N Lag','Average'],loc='upper right')
                    plt.subplot(312)
                    plt.plot(cc[1:,0],'r-')
                    plt.plot(cc[1:,1],'g-')
                    plt.plot(cc[1:,2],'b-')
                    plt.ylabel('cc');plt.xlabel('days')
                    plt.subplot(313)
                    plt.plot(error1[1:,0],'r-')
                    plt.plot(error1[1:,1],'g-')
                    plt.plot(error1[1:,2],'b-')
                    plt.ylabel('error [%]');plt.xlabel('days')

            if mwcs:
                if not stretch:
                    plt.figure()
                    plt.subplot(211)
                    plt.title(h5file.split('/')[-1])
                    plt.plot(dv2[1:],'b-')
                    plt.ylabel('dv/v [%]')
                    plt.subplot(212)
                    plt.plot(error2[1:],'b-')
                    plt.xlabel('days');plt.ylabel('errors [%]')
                else:
                    if onelag:
                        plt.figure()
                        plt.subplot(211)
                        plt.title(h5file.split('/')[-1])
                        plt.plot(dv2[1:],'b-');plt.plot(dv1[1:],'r-')
                        plt.ylabel('dv/v [%]')
                        plt.legend(['mwcs','stretch'],loc='upper right')
                        plt.subplot(212)
                        plt.plot(error2[1:],'b-');plt.plot(error1[1:],'r-')
                        plt.xlabel('days');plt.ylabel('errors [%]')
                    else:
                        plt.figure()
                        plt.subplot(211)
                        plt.title(h5file.split('/')[-1])
                        plt.plot(dv2[1:],'b-');plt.plot(dv1[1:,2],'r-')
                        plt.legend(['mwcs','stretch-ave'],loc='upper right')
                        plt.ylabel('dv/v [%]')
                        plt.subplot(212)
                        plt.plot(error2[1:],'b-');plt.plot(error1[1:,2],'r-')
                        plt.xlabel('days');plt.ylabel('errors [%]')           
            plt.show()
