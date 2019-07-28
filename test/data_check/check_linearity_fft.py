import pyasdf 
import numpy as np 
import scipy.fftpack
import matplotlib.pyplot as plt 

'''
this script takes a chunk of noise spectrum for a station pair and 
compare their cross-correlation functions computed using two schemes:
one is averaging the frequency domain and the other is in the time
domain
'''

def cross_correlation1(fft1,fft2,maxlag,dt,Nfft):
    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(fft1.shape,dtype=np.complex64)
    corr = np.conj(fft1) * fft2

    ncorr = np.zeros((fft1.shape[0],Nfft),dtype=np.complex64)
    ncorr[:,:Nfft//2] = corr[:,:]
    ncorr[:,-(Nfft//2)+1:]=np.flip(np.conj(ncorr[:,1:(Nfft//2)]),axis=1)
    ncorr[:,0]=complex(0,0)
    ncorr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(ncorr, Nfft, axis=1)))

    tcorr = np.arange(-Nfft//2 + 1, Nfft//2)*dt
    ind   = np.where(np.abs(tcorr) <= maxlag)[0]
    ncorr = ncorr[:,ind]
    ncorr = np.mean(ncorr,axis=0)
    return ncorr

def cross_correlation2(fft1,fft2,maxlag,dt,Nfft):
    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(fft1.shape,dtype=np.complex64)
    corr = np.conj(fft1) * fft2

    ncorr = np.zeros(shape=Nfft,dtype=np.complex64)
    ncorr[:Nfft//2] = np.mean(corr,axis=0)
    ncorr[-(Nfft//2)+1:]=np.flip(np.conj(ncorr[1:(Nfft//2)]),axis=0)
    ncorr[0]=complex(0,0)
    ncorr = np.fft.ifftshift(scipy.fftpack.ifft(ncorr, Nfft, axis=0))
    print(ncorr.real,ncorr.imag)

    tcorr = np.arange(-Nfft//2 + 1, Nfft//2)*dt
    ind   = np.where(np.abs(tcorr) <= maxlag)[0]
    ncorr = ncorr[ind]
    return ncorr


#-----common parameters------
iday   = '2010_01_10'
icomp  = 'EHZ'
dt     = 0.05
maxlag = 800

sfile1 = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT/N.AC2H.h5'
sfile2 = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT/N.CHHH.h5'

#-----------reading the data------------
ds1 = pyasdf.ASDFDataSet(sfile1,mode='r')
ds2 = pyasdf.ASDFDataSet(sfile2,mode='r')

spect1 = ds1.auxiliary_data[icomp][iday].data[:]
spect2 = ds2.auxiliary_data[icomp][iday].data[:]
std1 = ds1.auxiliary_data[icomp][iday].parameters['std']
std2 = ds2.auxiliary_data[icomp][iday].parameters['std']
nwin = spect1.shape[0]
nfft = spect1.shape[1]*2

print('data dimension for spect1 and spect2 are %d and %d' % (spect1.ndim,spect2.ndim))

#------select the sections-------
indx1 = np.where(std1<10)[0]
indx2 = np.where(std2<10)[0]
bb=np.intersect1d(indx1,indx2)
print(spect1[bb,:],spect2[bb,:])

corr1=cross_correlation1(spect1[bb,:],spect2[bb,:],np.round(maxlag),dt,nfft)
corr2=cross_correlation2(spect1[bb,:],spect2[bb,:],np.round(maxlag),dt,nfft)

#---plotting----
plt.subplot(311)
plt.plot(corr1)
plt.subplot(312)
plt.plot(corr2)
plt.subplot(313)
plt.plot(corr2)
plt.plot(corr1)
plt.show()