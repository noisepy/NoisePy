import os
import copy
import scipy
import time
import obspy
import pyasdf
import numpy as np
from numba import jit
from scipy.fftpack import fft,ifft,next_fast_len
from scipy.signal import butter, hilbert, wiener
from scipy.linalg import svd
from obspy.signal.filter import bandpass,lowpass
from obspy.signal.util import _npts2nfft
from obspy.core.util.base import _get_function_from_entry_point


'''
assemby of modules to process noise data and make cross-correlations. it includes some functions from
the noise package by Tim Clements (https://github.com/tclements/noise)

by: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
    Marine Denolle (mdenolle@fas.harvard.edu)
'''


def load_pfiles(pfiles):
    '''
    read the dictionary containing all station-pair information for the cross-correlation data
    that is saved in ASDF format, and merge them into one sigle array for stacking purpose. 

    input pfiles: the file names containing all information
    output: an array of all station-pair information for the cross-correlations
    '''
    # get all unique path list
    paths_all = []
    for ii in range(len(pfiles)):
        with pyasdf.ASDFDataSet(pfiles[ii],mpi=False,mode='r') as pds:
            try:
                tpath = pds.auxiliary_data.list()
            except Exception:
                continue
        paths_all = list(set(paths_all+tpath))
    return paths_all

def get_coda_window(dist,vmin,maxlag,dt,wcoda):
    '''
    calculate the coda wave window for the ccfs based on
    the travel time of the balistic wave and select the 
    index for the time window
    '''
    #--------construct time axis----------
    tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt

    #--get time window--
    tbeg=int(dist/vmin)
    tend=tbeg+wcoda
    if tend>maxlag:
        raise ValueError('time window ends at maxlag, too short!')
    if tbeg>maxlag:
        raise ValueError('time window starts later than maxlag')
    
    #----convert to point index----
    ind1 = np.where(abs(tt)==tbeg)[0]
    ind2 = np.where(abs(tt)==tend)[0]

    if len(ind1)!=2 or len(ind2)!=2:
        raise ValueError('index for time axis is wrong')
    ind = [ind2[0],ind1[0],ind1[1],ind2[1]]

    return ind    


def C3_process(S1_data,S2_data,Nfft,win):
    '''
    performs all C3 processes including 1) cutting the time window for P-N parts;
    2) doing FFT for the two time-seris; 3) performing cross-correlations in freq;
    4) ifft to time domain
    '''
    #-----initialize the spectrum variables----
    ccp1 = np.zeros(Nfft,dtype=np.complex64)
    ccn1 = ccp1
    ccp2 = ccp1
    ccn2 = ccp1
    ccp  = ccp1
    ccn  = ccp1

    #------find the time window for sta1------
    S1_data_N = S1_data[win[0]:win[1]]
    S1_data_N = S1_data_N[::-1]
    S1_data_P = S1_data[win[2]:win[3]]
    S2_data_N = S2_data[win[0]:win[1]]
    S2_data_N = S2_data_N[::-1]
    S2_data_P = S2_data[win[2]:win[3]]

    #---------------do FFT-------------
    ccp1 = scipy.fftpack.fft(S1_data_P, Nfft)
    ccn1 = scipy.fftpack.fft(S1_data_N, Nfft)
    ccp2 = scipy.fftpack.fft(S2_data_P, Nfft)
    ccn2 = scipy.fftpack.fft(S2_data_N, Nfft)

    #------cross correlations--------
    ccp = np.conj(ccp1)*ccp2
    ccn = np.conj(ccn1)*ccn2

    return ccp,ccn

def get_SNR(corr,snr_parameters,parameters):
    '''
    estimate the SNR for the cross-correlation functions. the signal is defined
    as the maxinum in the time window of [dist/max_vel,dist/min_vel]. the noise
    is defined as the std of the trailing 100 s window. flag is to indicate to 
    estimate both lags of the cross-correlation funciton of just the positive

    corr: the noise cross-correlation functions
    snr_parameters: dictionary for some parameters to estimate S-N
    parameters: dictionary for parameters about the ccfs
    '''
    #---------common variables----------
    sampling_rate = int(1/parameters['dt'])
    npts = int(2*sampling_rate*parameters['lag'])
    indx = npts//2
    dist = parameters['dist']
    minvel = snr_parameters['minvel']
    maxvel = snr_parameters['maxvel']

    #-----index to window the signal part------
    indx_sig1 = int(dist/maxvel)*sampling_rate
    indx_sig2 = int(dist/minvel)*sampling_rate
    if maxvel > 5:
        indx_sig1 = 0

    #-------index to window the noise part---------
    indx_noise1 = indx_sig2
    indx_noise2 = indx_noise1+snr_parameters['noisewin']*sampling_rate

    #----prepare the filters----
    fb = snr_parameters['freqmin']
    fe = snr_parameters['freqmax']
    ns = snr_parameters['steps']
    freq = np.zeros(ns,dtype=np.float32)
    psnr = np.zeros(ns,dtype=np.float32)
    nsnr = np.zeros(ns,dtype=np.float32)
    ssnr = np.zeros(ns,dtype=np.float32)

    #--------prepare frequency info----------
    step = (np.log(fb)-np.log(fe))/(ns-1)
    for ii in range(ns):
        freq[ii]=np.exp(np.log(fe)+ii*step)

    for ii in range(1,ns-1):
        f2 = freq[ii-1]
        f1 = freq[ii+1]

        #-------------filter data before estimate SNR------------
        ncorr = bandpass(corr,f1,f2,sampling_rate,corners=4,zerophase=True)
        psignal = max(ncorr[indx+indx_sig1:indx+indx_sig2])
        nsignal = max(ncorr[indx-indx_sig2:indx-indx_sig1])
        ssignal = max((ncorr[indx+indx_sig1:indx+indx_sig2]+np.flip(ncorr[indx-indx_sig2:indx-indx_sig1]))/2)
        pnoise  = np.std(ncorr[indx+indx_noise1:indx+indx_noise2])
        nnoise  = np.std(ncorr[indx-indx_noise2:indx-indx_noise1])
        snoise  = np.std((ncorr[indx+indx_noise1:indx+indx_noise2]+np.flip(ncorr[indx-indx_noise2:indx-indx_noise1]))/2)
        
        #------in case there is no data-------
        if pnoise==0 or nnoise==0 or snoise==0:
            psnr[ii]=0
            nsnr[ii]=0
            ssnr[ii]=0
        else:
            psnr[ii] = psignal/pnoise
            nsnr[ii] = nsignal/nnoise
            ssnr[ii] = ssignal/snoise

    parameters['psnr'] = psnr[1:-1]
    parameters['nsnr'] = nsnr[1:-1]
    parameters['ssnr'] = nsnr[1:-1]
    parameters['freq'] = freq[1:-1]

    return parameters

def norm(arr):
    """ Demean and normalize a given input to unit std. """
    arr -= arr.mean(axis=1, keepdims=True)
    return (arr.T / arr.std(axis=-1)).T

def NCF_denoising(img_to_denoise,Mdate,Ntau,NSV):

	if img_to_denoise.ndim ==2:
		M,N = img_to_denoise.shape
		if NSV > np.min([M,N]):
			NSV = np.min([M,N])
		[U,S,V] = svd(img_to_denoise,full_matrices=False)
		S = scipy.linalg.diagsvd(S,S.shape[0],S.shape[0])
		Xwiener = np.zeros([M,N])
		for kk in range(NSV):
			SV = np.zeros(S.shape)
			SV[kk,kk] = S[kk,kk]
			X = U@SV@V
			Xwiener += wiener(X,[Mdate,Ntau])
			
		denoised_img = wiener(Xwiener,[Mdate,Ntau])
	elif img_to_denoise.ndim ==1:
		M = img_to_denoise.shape[0]
		NSV = np.min([M,NSV])
		denoised_img = wiener(img_to_denoise,Ntau)
		temp = np.trapz(np.abs(np.mean(denoised_img) - img_to_denoise))    
		denoised_img = wiener(img_to_denoise,Ntau,np.mean(temp))

	return denoised_img

def correlate(fft1_smoothed_abs,fft2,D,Nfft,dataS_t):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of 
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that 
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)

    PARAMETERS:
    ---------------------
    fft1_smoothed_abs: smoothed power spectral density of the FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    dataS_t: matrix of datetime object.

    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function
    '''
    #----load paramters----
    dt      = D['dt']
    freqmin = D['freqmin']
    freqmax = D['freqmax']
    maxlag  = D['maxlag']
    method  = D['cc_method']
    cc_len  = D['cc_len'] 
    substack= D['substack']                                                          
    substack_len  = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    nwin  = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size,)*fft2.reshape(fft2.size,)

    if method == "coherency":
        temp = moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)             
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    #--------------- remove outliers in frequency domain -------------------
    # [reduce the number of IFFT by pre-selecting good windows before substack]
    freq = scipy.fftpack.fftfreq(Nfft, d=dt)[:Nfft2]
    i1 = np.where( (freq>=freqmin) & (freq <= freqmax))[0]

    # this creates the residuals between each window and their median
    med = np.log10(np.median(corr[:,i1],axis=0))
    r   = np.log10(corr[:,i1]) - med
    ik  = np.zeros(nwin,dtype=np.int)
    # find time window of good data
    for i in range(nwin):
        if np.any( (r[i,:]>=med-10) & (r[i,:]<=med+10) ):ik[i]=i
    ik1 = np.nonzero(ik)
    ik=ik[ik1]

    if substack:
        if substack_len == cc_len:
            # choose to keep all fft data for a day
            s_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)   # stacked correlation
            ampmax = np.zeros(nwin,dtype=np.float32)
            n_corr = np.zeros(nwin,dtype=np.int16)                  # number of correlations for each substack
            t_corr = dataS_t                                        # timestamp
            crap   = np.zeros(Nfft,dtype=np.complex64)
            for i in range(len(ik)): 
                n_corr[ik[i]]= 1           
                crap[:Nfft2] = corr[ik[i],:]
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[ik[i],:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]
        
        else:     
            # get time information
            Ttotal = dataS_t[-1]-dataS_t[0]             # total duration of what we have now
            tstart = dataS_t[0]

            nstack = int(np.round(Ttotal/substack_len))
            ampmax = np.zeros(nstack,dtype=np.float32)
            s_corr = np.zeros(shape=(nstack,Nfft),dtype=np.float32)
            n_corr = np.zeros(nstack,dtype=np.int)
            t_corr = np.zeros(nstack,dtype=np.float)
            crap   = np.zeros(Nfft,dtype=np.complex64)                                              

            for istack in range(nstack):                                                                   
                # find the indexes of all of the windows that start or end within 
                itime = np.where( (dataS_t[ik] >= tstart) & (dataS_t[ik] < tstart+substack_len) )[0]  
                if len(ik[itime])==0:tstart+=substack_len;continue
                
                crap[:Nfft2] = np.mean(corr[ik[itime],:],axis=0)   # linear average of the correlation 
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[istack,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(ik[itime])           # number of windows stacks
                t_corr[istack] = tstart                   # save the time stamps
                tstart += substack_len
                #print('correlation done and stacked at time %s' % str(t_corr[istack]))
            
            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        n_corr = len(ik)
        s_corr = np.zeros(Nfft,dtype=np.float32)
        t_corr = dataS_t[0]
        crap   = np.zeros(Nfft,dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr[ik,:],axis=0)
        crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2],axis=0)
        crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
        s_corr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    # trim the CCFs in [-maxlag maxlag] 
    t = np.arange(-Nfft2+1, Nfft2)*dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim==1:
        s_corr = s_corr[ind]
    elif s_corr.ndim==2:
        s_corr = s_corr[:,ind]
    return s_corr,t_corr,n_corr

if __name__ == "__main__":
    pass
