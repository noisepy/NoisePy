import os
import glob
import scipy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import butter, hilbert
from obspy.signal.filter import highpass
from scipy.fftpack import fft,next_fast_len

'''
check the convergence rate of the stacked CCFs based on the correlation coefficient
between the sub-stacked waveforms and the all-stacked waveform

--used to find out how long data needed to be stacked for stable CCFs
'''

def pws(arr,sampling_rate,power=2,pws_timegate=5.):
    """
    Performs phase-weighted stack on array of time series. 
    Modified on the noise function from Tim Climents.

    Follows methods of Schimmel and Paulssen, 1997. 
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    :type arr: numpy.ndarray
    :param arr: N length array of time series data 
    :type power: float
    :param power: exponent for phase stack
    :type sampling_rate: float 
    :param sampling_rate: sampling rate of time series 
    :type pws_timegate: float 
    :param pws_timegate: number of seconds to smooth phase stack
    :Returns: Phase weighted stack of time series data
    :rtype: numpy.ndarray  
    """

    if arr.ndim == 1:
        return arr
    N,M = arr.shape
    analytic = hilbert(arr,axis=1, N=next_fast_len(M))[:,:M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j*phase),axis=0)
    phase_stack = np.abs(phase_stack)**(power)

    # smoothing 
    #timegate_samples = int(pws_timegate * sampling_rate)
    #phase_stack = moving_ave(phase_stack,timegate_samples)
    weighted = np.multiply(arr,phase_stack)
    return np.mean(weighted,axis=0)


# input files
sfiles  = sorted(glob.glob('/Volumes/Chengxin/KANTO/STACK/pws_stack/E.AYHM/*.h5'))
outpath = '/Users/chengxin/Documents/NoisePy_example/Kanto/figures/convergence_rate/pws'
if not os.path.isdir(outpath): os.mkdir(outpath)

# common parameters
ccomp = 'ZZ'
lag   = 200
fmin  = 0.05
stack = 'pws'

# loop through each station-pair
for sfile in sfiles:
    # useful parameters from each asdf file 
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    alist = ds.auxiliary_data.list()
    try:
        dt    = ds.auxiliary_data[alist[0]][ccomp].parameters['dt']
        dist  = ds.auxiliary_data[alist[0]][ccomp].parameters['dist']
    except Exception:
        print('continue! no %s component exist'%ccomp)
        continue

    # stacked data and filter it
    sdata = ds.auxiliary_data[alist[0]][ccomp].data[:]
    sdata = highpass(sdata,fmin,int(1/dt),corners=4, zerophase=True)

    # time domain variables
    nwin  = len(alist[1:])
    npts  = sdata.size
    tvec  = np.arange(-npts//2,npts//2+1)*dt
    indx  = np.where(np.abs(tvec)<=lag)[0]
    npts  = len(indx)
    ndata = np.zeros(shape=(nwin,npts),dtype=np.float32)
    corr  = np.zeros(nwin,dtype=np.float32)

    #################################
    ####### load data matrix ########
    #################################
    for ii,ilist in enumerate(alist[1:]):
        try:
            tdata     = ds.auxiliary_data[ilist][ccomp].data[:]
            ndata[ii] = highpass(tdata[indx],fmin,int(1/dt),corners=4, zerophase=True)
            corr[ii]  = np.corrcoef(sdata[indx],ndata[ii])[0,1]
        except Exception:
            continue 

    # remove bad/empty waveforms
    tindx = np.where(corr>0)[0]
    ndata = ndata[tindx]
    nwin  = len(ndata)
    print('old nwin and new nwin are %d and %d'%(len(alist[1:]),nwin))

    # make new time domain variables
    nsdata = np.zeros(shape=(nwin,npts),dtype=np.float32)
    corr   = np.zeros(nwin,dtype=np.float32)

    # freq domain variables
    nfft  = int(next_fast_len(npts))
    nfreq = scipy.fftpack.fftfreq(nfft,d=dt)[:nfft//2]
    spect = np.zeros(shape=(nwin,nfft//2),dtype=np.complex64)
    findx = np.where((nfreq>=0.1) & (nfreq<=5))[0]

    ###############################
    #### do consective stacking ###
    ###############################
    for ii in range(nwin):
        if ii==0:
            nsdata[ii] = ndata[ii]
            spect[ii]  = fft(nsdata[ii],nfft,axis=0)[:nfft//2]
            spect[ii] /= np.max(np.abs(spect[ii]),axis=0)
            corr[ii]   = np.corrcoef(sdata[indx],nsdata[ii])[0,1]
            nsdata[ii] /= np.max(nsdata[ii],axis=0)
        else:
            if stack == 'linear':
                nsdata[ii] = np.mean(ndata[:ii],axis=0)
            else:
                nsdata[ii] = pws(ndata[:ii],int(1/dt))
            spect[ii]  = fft(nsdata[ii],nfft,axis=0)[:nfft//2]
            spect[ii] /= np.max(np.abs(spect[ii]),axis=0)
            corr[ii]   = np.corrcoef(sdata[indx],nsdata[ii])[0,1]
            nsdata[ii] /= np.max(nsdata[ii],axis=0)

    #################################
    ######## do plotting now ########
    #################################
    fig,ax = plt.subplots(3,sharex=False)
    ax[0].plot(corr,'r-')
    ax[0].set_ylabel('correlation coeff')
    ax[0].set_xlabel('number of segments')
    ax[0].xaxis.set_ticks_position('bottom')
    ax[0].set_title('%s %5.2f km'%(sfile.split('/')[-1],dist))
    ax[1].matshow(nsdata,cmap='seismic',extent=[-lag,lag+dt,nwin,0],aspect='auto')
    ax[1].set_xlabel('time [s]')
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].plot(tvec[indx],(sdata[indx]*5000+nwin//2),'k-',linewidth=0.5)
    ax[2].matshow(np.abs(spect[:,findx]),cmap='seismic',extent=[0.1,5,nwin,0],aspect='auto')
    ax[2].set_xlabel('freq [Hz]')
    ax[2].set_xscale('log')
    ax[2].xaxis.set_ticks_position('bottom')
    fig.tight_layout()

    tmp = sfile.split('/')[-1]
    outfname = outpath+'/{0:s}.pdf'.format(tmp[0:-3])
    fig.savefig(outfname, format='pdf', dpi=400)
    plt.close()