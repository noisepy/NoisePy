import scipy
import pycwt
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack.helper import next_fast_len

'''
use synthetic data to verify the cwt function
'''

# make synthetic data
################################
syn = True
if syn:

    # time series 
    dt = 0.1
    t  = np.arange(0,50,dt)
    npts = len(t)
    pi2 = np.pi*2

    # add noise
    amp_noise = 0.25
    # freq bands
    fmin = 0.1;fmax = 2

    # main freq 
    a1 = 1;f1 = 0.2
    a2 = 0.8;f2 = 0.5
    a3 = 1.2;f3 = 1
    syn_data = a1*(np.cos(pi2*f1*t)+amp_noise*np.random.rand(len(t)))+a2*(np.cos(pi2*f2*t)+amp_noise*np.random.rand(len(t)))+a3*(np.cos(pi2*f3*t)+amp_noise*np.random.rand(len(t)))

    # do fft 
    nfft = next_fast_len(npts)
    spec = scipy.fftpack.fft(syn_data,nfft)[:nfft//2]
    freqvec = scipy.fftpack.fftfreq(nfft,d=dt)[:nfft//2]
    plt.subplot(211)
    plt.plot(t,syn_data)
    plt.subplot(212)
    plt.plot(freqvec,np.abs(spec))
    plt.show()

    # continous wavelet transform
    dj=1/12
    s0=-1
    J=-1
    wvn='morlet'
    cwt, sj, freq, coi, _, _ = pycwt.cwt(syn_data, dt, dj, s0, J, wvn)

    # do filtering here
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: frequency out of limits!')
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt = cwt[freq_ind]
    freq = freq[freq_ind]
    period = 1/freq
    sj = sj[freq_ind]
    rcwt = np.real(cwt)
    pcwt = np.abs(cwt)**2

    fig,ax = plt.subplots(2,sharex=True)
    im1=ax[0].imshow(rcwt,cmap='jet',extent=[0,t[-1],np.log2(period[-1]),np.log2(period[0])],aspect='auto')
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('period [s]')
    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),np.ceil(np.log2(period.max())))
    ax[0].set_yticks(np.log2(Yticks))
    ax[0].set_yticklabels(Yticks)
    ax[0].xaxis.set_ticks_position('bottom')
    cbar=fig.colorbar(im1,ax=ax[0])

    im2=ax[1].imshow(pcwt,cmap='jet',extent=[0,t[-1],np.log2(period[-1]),np.log2(period[0])],aspect='auto')
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('period [s]')
    ax[1].set_yticks(np.log2(Yticks))
    ax[1].set_yticklabels(Yticks)
    ax[1].xaxis.set_ticks_position('bottom')
    cbar=fig.colorbar(im2,ax=ax[1])
    fig.tight_layout()
    fig.show()