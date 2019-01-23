import numpy as np
import scipy
import math
import time
from numba import cuda,float32,complex64

@cuda.jit('void(float32[:],float32[:])')
def smooth_gpu(A,B):
    '''
    Cuda kernel to do running smooth ave
    A, B are both 1-D arrays
    '''
    pos=cuda.grid(1)
    if pos < A.shape[0]:
        N=10
        tmp=0.
        if pos<N or pos>A.shape[0]-N-1:
            B[pos]=1
        else:
            for i in range(-N,N+1):
	            tmp+=A[pos+i]
            B[pos]=tmp/(2*N+1)


@cuda.jit('void(complex64[:],complex64[:],complex64[:])')
def decon_gpu(A,B,C):
    '''
    Kernels to perform freq domain cross-correlation and smoothing
    A, B are 1-D arrays of noise spectrum, of complex64 type
    '''
    pos = cuda.grid(1)
    N=10
    if pos<A.size:
        C[pos] = A[pos].conjugate()*B[pos]
        if pos<A.size-N-1 and pos>N:
            tmp=0.
            for i in range(-N,N+1):
                tmp+=math.sqrt(A[pos+i].real**2+A[pos+i].imag**2)
            tmp=tmp/(2*N+1)
            if tmp==0:
                C[pos]=complex(0,0)
            else:
                C[pos]=C[pos]/tmp**2


@cuda.jit('void(complex64[:],complex64[:],complex64[:])')
def coherence_gpu(A,B,C):
    '''
    Kernels to perform freq domain cross-correlation and smoothing
    A, B are 1-D arrays of noise spectrum, of complex64 type
    '''
    pos = cuda.grid(1)
    N=10
    if pos<A.size:
        C[pos] = A[pos].conjugate()*B[pos]
        if pos<A.size-N-1 and pos>N:
            tmp1=0.
            tmp2=0.
            for i in range(-N,N+1):
                tmp1+=math.sqrt(A[pos+i].real**2+A[pos+i].imag**2)
                tmp2+=math.sqrt(B[pos+i].real**2+B[pos+i].imag**2)
            tmp1=tmp1/(2*N+1)
            tmp2=tmp2/(2*N+1)
            if tmp1==0 or tmp2==0:
                C[pos]=complex(0,0)
            else:
                C[pos]=C[pos]/(tmp1*tmp2)


def correlate(fft1,fft2, maxlag,dt, Nfft, method="cross-correlation"):
    
    """GPU version of cc function, which takes ndimensional *data* array, computes 
    the cross-correlation in the frequency domain and returns the 
    cross-correlation function between [-*maxlag*:*maxlag*].

    :type fft1: :class:`numpy.ndarray`
    :param fft1: This array contains the fft of each timeseries to be cross-correlated.
    :type maxlag: int
    :param maxlag: This number defines the number of samples (N=2*maxlag + 1) of the CCF that will be returned.

    :rtype: :class:`numpy.ndarray`
    :returns: The cross-correlation function between [-maxlag:maxlag]
    """

    if fft1.ndim == 1:
        nwin=1
    elif fft1.ndim == 2:
        nwin= int(fft1.shape[0])

    t0=time.time()
    corr=np.zeros(shape=(nwin,Nfft),dtype=np.complex64)
    fft1_globe = cuda.to_device(fft1[:,:Nfft//2].reshape(fft1.size,))
    fft2_globe = cuda.to_device(fft2[:,:Nfft//2].reshape(fft2.size,))
    corr_globe = cuda.device_array(shape=(nwin*(Nfft//2),),dtype=np.complex64)
    
    threadsperblock = 2000
    blockspergrid = math.ceil(fft1_globe.size/threadsperblock)
    
    if method == 'deconv':
        decon_gpu[threadsperblock,blockspergrid](fft1_globe,fft2_globe,corr_globe)
    elif method =='coherence':
        coherence_gpu[threadsperblock,blockspergrid](fft1_globe,fft2_globe,corr_globe)

    tcorr = corr_globe.copy_to_host()
    corr  = tcorr.reshape(nwin,Nfft//2)

    ncorr = np.zeros(shape=Nfft,dtype=np.complex64)
    ncorr[:Nfft//2] = np.mean(corr,axis=0)
    ncorr[-(Nfft//2)+1:]=np.flip(np.conj(ncorr[1:(Nfft//2)]),axis=0)
    ncorr[0]=complex(0,0)
    ncorr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(ncorr, Nfft, axis=0)))

    t1=time.time()
    print('it takes '+str(t1-t0)+' s')

    tcorr = np.arange(-Nfft//2 + 1, Nfft//2)*dt
    ind   = np.where(np.abs(tcorr) <= maxlag)[0]
    ncorr = ncorr[ind]
    tcorr = tcorr[ind]

    return ncorr,tcorr
