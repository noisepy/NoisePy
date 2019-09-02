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

if __name__ == "__main__":
    pass
