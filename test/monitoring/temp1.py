import scipy
import pycwt
import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression

'''
generate some random signals and convolve with ricker to create synthetic waveforms
for testing the monitoring methods

this script is used to understand intermediate steps involved in the monitoring functions
'''
###############################
### generate synthetic data ###
###############################

# generate a time series
npts = 1000
dt   = 0.1
tr   = np.random.rand(npts)
tvec = np.arange(0,npts*dt,dt)

# ricker wavelet
pts = 100
fc  = 0.5
rvec = np.arange(-pts/2,pts/2)*dt
rick = (1.0 - 2.0*(np.pi**2)*(fc**2)*(rvec**2)) * np.exp(-(np.pi**2)*(fc**2)*(rvec**2))

# convolution
data = np.convolve(tr,rick)[:npts]
tmp  = cosine_taper(npts,0.1)
data *= tmp

# stretch the waveform
dv = 1      # in %
ntvec = tvec*(1+dv/100)
ndata = np.interp(x=ntvec, xp=tvec, fp=data)

# plots for the new waveforms
plt.subplot(311);plt.plot(tvec,tr,'r-')
plt.subplot(312);plt.plot(rvec,rick)
plt.subplot(313);plt.plot(data,'r-');plt.plot(ndata,'b--')
plt.legend(['original','1%'],loc='upper right')
plt.tight_layout()
plt.show()

# freq info of the data
nfft = int(next_fast_len(npts))
freq = scipy.fftpack.fftfreq(nfft,d=dt)[:nfft//2]
spect = scipy.fftpack.fft(data,nfft,axis=0)[:nfft//2]
spect1= scipy.fftpack.fft(ndata,nfft,axis=0)[:nfft//2]

# check the spectrum for the signals
plot_spect = True
if plot_spect:

    # plot the spectrum
    plt.subplot(211)
    plt.plot(freq,np.real(spect),'r-')
    plt.plot(freq,np.imag(spect),'b-')
    plt.legend(['amp','pha'],loc='upper right')
    plt.subplot(212)
    plt.plot(freq,np.real(spect1),'r-')
    plt.plot(freq,np.imag(spect1),'b-')
    plt.legend(['amp','pha'],loc='upper right')
    plt.show()

##########################################
### test the data with various methods ###
##########################################

# 1. stretching: time domain
############################

# basic parameters
dv_range = 0.02                        
nbtrial = 50

# make useful one for measurements
dvmin = -np.abs(dv_range)
dvmax = np.abs(dv_range)
Eps = 1+(np.linspace(dvmin, dvmax, nbtrial))
cof = np.zeros(Eps.shape,dtype=np.float32)

# set of stretched/compressed current waveforms
for ii in range(len(Eps)):
    nt = tvec*Eps[ii]
    s = np.interp(x=tvec, xp=nt, fp=ndata)
    waveform_ref = data
    waveform_cur = s
    cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

# correlation coefficient between the reference and initial current waveforms
cdp = np.corrcoef(data, ndata)[0, 1] 
plt.plot(100*(Eps-1),cof)
plt.show()


# DTW: cool!!
##############

# some basic parameters
maxlag = 50     # maxmum points to move
b      = 20     # velocity percentage
direct = 1      # direction to accumulate errors (1=forward, -1=backward)

# get error 
err   = monitor_modules.computeErrorFunction( ndata, data, npts, maxlag ) 
# calcuate distance based on the eror
dist  = monitor_modules.accumulateErrorFunction( direct, err, npts, maxlag, b )
# find the best solutions to shift
stbar = monitor_modules.backtrackDistanceFunction( -1*direct, dist, err, -maxlag, b )

# plot each matrix
fig,ax = plt.subplots(2,sharex=True)
im1=ax[0].matshow(np.transpose(err),cmap='jet',extent=[0,npts*dt,-maxlag*dt,maxlag*dt],aspect='auto')
ax[0].set_xlabel('time [s]')
ax[0].set_ylabel('dt [s]')
ax[0].xaxis.set_ticks_position('bottom')
ax[0].plot(tvec,-stbar*dt,'r*',markersize=1)
cbar=fig.colorbar(im1,ax=ax[0])
cbar.ax.set_ylabel('accumulated error')
im2=ax[1].matshow(np.transpose(dist),cmap='jet',extent=[0,npts*dt,-maxlag*dt,maxlag*dt],aspect='auto')
ax[1].set_xlabel('time [s]')
ax[1].set_ylabel('dt [s]')
ax[1].xaxis.set_ticks_position('bottom')
ax[1].plot(tvec,-stbar*dt,'r*',markersize=1)
cbar=fig.colorbar(im2,ax=ax[1])
cbar.ax.set_ylabel('distance')
fig.tight_layout()
fig.show()


# waveform cross-correlation (WCC)
##################################
# TO DO: how sensitive it is to move_win and sstep
# it needs to move the window as MWCS

# basic parameters
freq = [0.2,1]
move_win = int(1.5/np.min(freq))      # keep at least 2.5 times of wavelength of the longest period
sstep = 0.3*move_win
sps   = int(1/dt)

# moving window
indx1 = 0
indx2 = indx1+int(move_win*sps)
ttt = []
tdt = []
while indx2 <= npts: 
    win = data[indx1:indx2]
    win1= ndata[indx1:indx2]

    # normalize matrix
    win = (win - win.mean()) / win.std()
    win1 = (win1 - win1.mean()) / win1.std()

    # get cross correlation coeff
    cc = np.correlate(win1, win, mode='same')
    cc = cc/np.sqrt((win**2).sum() * (win1**2).sum())
        
    # find time shift from the cross correlation functions
    imaxcc2 = np.where(cc==np.max(cc))[0]
    maxcc2 = np.max(cc)
    m = (imaxcc2-((move_win*sps)//2))*dt
    ttt.append(0.5*(indx1+indx2)*dt)
    tdt.append(m)

    indx1 += move_win*sps
    indx2 += move_win*sps

    plot_wcc = False
    if plot_wcc:
        plt.subplot(211);plt.plot(win,'r-');plt.plot(win1,'g-')
        plt.subplot(212);plt.plot(cc*dt)
        plt.show()

# plot the overal results
plt.subplot(211);plt.plot(tvec,data,'r-');plt.plot(tvec,ndata,'b-')
plt.subplot(212);plt.plot(ttt,tdt,'r-');plt.plot(ttt,tdt,'*',markersize=5);plt.show()


# MWCS: yeah!! a mixture of time and freq domain processing
###########################################################
# TO DO: how sensitive it is to move_win and sstep

# basic parameters
freq = [0.1,1]
move_win = 2.5*int(1/np.min(freq))      # keep at least 2.5 times of wavelength of the longest period
sstep = 0.2*move_win
sps   = int(1/dt)

# loop through each moving window
indx1 = 0
indx2 = indx1+move_win*sps
while indx2 <= npts: 
    win = data[indx1:indx2]
    win1= ndata[indx1:indx2]

    # calculate the spectrum 
    tnpts = indx2-indx1
    tnfft = int(next_fast_len(tnpts))
    tfreq = scipy.fftpack.fftfreq(tnfft,d=dt)[:tnfft//2]
    tspect = scipy.fftpack.fft(win,tnfft,axis=0)[:tnfft//2]
    tspect1= scipy.fftpack.fft(win1,tnfft,axis=0)[:tnfft//2]

    # cross spectrum and wrap for phi and freq
    cross_spect = tspect*np.conjugate(tspect1)
    dpha = np.unwrap(np.angle(cross_spect))
    dfre = tfreq*2*np.pi

    # linear regression to get dt at each t point

    # moving to the next window
    indx1 += move_win*sps
    indx2 += move_win*sps

    # plot phi and freq
    plt.plot(tfreq,dpha,'r-');plt.show()


# wavelet strectching: sweet!
################################

# basic parameters
dj=1/12
s0=-1
J=-1
wvn='morlet'

# continous wavelet transform
cwt1, sj, freq, coi1, _, _ = pycwt.cwt(data, dt, dj, s0, J, wvn)
cwt2, sj, freq, coi2, _, _ = pycwt.cwt(ndata, dt, dj, s0, J, wvn)

period = 1/freq
rcwt1,rcwt2 = np.real(cwt1),np.real(cwt2)

# plot the wavelet spectrum
plot_wct = False
if plot_wct:
    fig,ax = plt.subplots(2,sharex=True)
    im1=ax[0].imshow(rcwt1,cmap='jet',extent=[0,npts*dt,freq[-1],freq[0]],aspect='auto')
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('freq [Hz]')
    ax[0].xaxis.set_ticks_position('bottom')
    ax[0].fill(np.concatenate([tvec, tvec[-1:]+dt, tvec[-1:]+dt, tvec[:1]-dt, tvec[:1]-dt]), \
        np.concatenate([np.log2(coi1), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]), \
        'k', alpha=0.3, hatch='x')
    cbar=fig.colorbar(im1,ax=ax[0])
    #cbar.ax.set_ylabel('accumulated error')
    im2=ax[1].imshow(rcwt2,cmap='jet',extent=[0,npts*dt,freq[-1],freq[0]],aspect='auto')
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('freq [Hz]')
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].fill(np.concatenate([tvec, tvec[-1:]+dt, tvec[-1:]+dt, tvec[:1]-dt, tvec[:1]-dt]), \
        np.concatenate([np.log2(coi2), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]), \
        'k', alpha=0.3, hatch='x')
    cbar=fig.colorbar(im2,ax=ax[1])
    fig.tight_layout()
    fig.show()

# use real part of the wavelet transform to find difference between cur/ref
dv_range = 0.02                        
nbtrial = 50

# make useful one for measurements
dvmin = -np.abs(dv_range)
dvmax = np.abs(dv_range)
Eps = 1+(np.linspace(dvmin, dvmax, nbtrial))
cof = np.zeros(Eps.shape,dtype=np.float32)

# loop through frequency
for jj in range(len(freq)):

    # set of stretched/compressed current waveforms
    for ii in range(len(Eps)):
        nt = tvec*Eps[ii]
        s = np.interp(x=tvec, xp=nt, fp=rcwt2[jj])
        waveform_ref = rcwt1[jj]
        waveform_cur = s
        cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    # correlation coefficient between the reference and initial current waveforms
    cdp = np.corrcoef(rcwt1, rcwt2)[0, 1]
    indx = np.nanargmax(cof) 
    plt.plot((Eps-1)*100,cof);plt.plot((Eps[indx]-1)*100,np.max(cof),'*',markersize=10)
    plt.title('wavelet strecthing at %s Hz'%(freq[jj]));plt.show()


# wavelet cross spectrum method
################################

# basic parameters
fs=1/dt
dj=1/12
s0=-1
J=-1
sig=False
wvn='morlet'

# wavelet cross spectrm
WCT, aWCT, coi, freq, sig = pycwt.wct(ndata, data, 1/fs, dj=dj, s0=s0, J=J, sig=sig, wavelet=wvn, normalize=True)

# unwrap the phase
phase = np.unwrap(aWCT,axis=-1) # axis=0, upwrap along time; axis=-1, unwrap along frequency
delta_t = phase / (2*np.pi*freq[:,None]) # normalize phase by (2*pi*frequency) 
plot_wxt = True
if plot_wxt:
    plt.imshow(delta_t,cmap='jet',extent=[0,npts*dt,freq[-1],freq[0]],aspect='auto')
    plt.xlabel('time [s]')
    plt.ylabel('freq [Hz]')
    plt.colorbar()
    plt.show()