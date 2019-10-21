import scipy
import pyasdf
import numpy as np
import core_functions
from scipy import signal 
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from obspy.signal.filter import bandpass
from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression

'''
generate some random signals and convolve with ricker to create synthetic waveforms
to test the monitoring functions

by Chengxin Jiang
'''

######################################
####### generate synthetic data ######
######################################

synth = 'time_syn'              # choose between pure and real synthetic data

#########PURELY SYNTHETIC INPUT#########
if synth == 'time_syn':
    # generate a time series
    lag  = 100
    dt   = 0.05
    npts = lag*int(1/dt)
    tr   = np.random.rand(npts)
    tvec = np.arange(0,npts*dt,dt)

    # ricker wavelet
    pts = 100
    fc  = 0.5
    rvec = np.arange(-pts/2,pts/2)*dt
    rick = (1.0-2.0*(np.pi**2)*(fc**2)*(rvec**2)) * np.exp(-(np.pi**2)*(fc**2)*(rvec**2))

    # convolution
    data = np.convolve(tr,rick)[:npts]
    tmp  = cosine_taper(npts,0.1)
    data *= tmp

elif synth == 'freq_syn':
    lag  = 100
    dt   = 0.05
    # generate a random spectrum
    nfft2  = lag*int(1/dt)
    tspect = np.zeros(shape=(nfft2),dtype=np.complex64)
    tspect[:nfft2] = 1+(np.random.rand(nfft2))*1j
    data = np.real(scipy.fftpack.ifft(tspect, nfft2, axis=0))
    data[0] = 0
    npts = len(data)
    tvec = np.arange(0,npts*dt,dt)

#############REAL INPYUT##############
elif synth == 'real_syn':
    sfile = '/Volumes/Chengxin/KANTO/STACK_ALL/E.ABHM/pws_E.ABHM_E.AYHM.h5'
    lag = 100

    # load 'stacked' waveform from a ASDF file
    with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
        try:
            dt   = ds.auxiliary_data['Allstack']['ZZ'].parameters['dt']
            maxlag  = ds.auxiliary_data['Allstack']['ZZ'].parameters['maxlag']
            tdata = ds.auxiliary_data['Allstack']['ZZ'].data[:]
        except Exception:
            raise ValueError('cannot open %s to read'%sfile)
    
    tvec_tmp = np.arange(-maxlag,maxlag+1,dt)
    indx = np.where((tvec_tmp>=0)&(tvec_tmp<lag))
    tvec = tvec_tmp[indx]
    data = tdata[indx]
    npts = len(data)

# stretch the waveform
dv = 1      # in %
ntvec = tvec*(1+dv/100)
ndata = np.interp(x=ntvec, xp=tvec, fp=data)

# plots for the new waveforms
plot_syn = True
if plot_syn:
    if synth == 'time_syn':
        plt.subplot(311);plt.plot(tvec,tr,'r-')
        plt.subplot(312);plt.plot(rvec,rick)
        plt.subplot(313);plt.plot(tvec,data,'r-');plt.plot(tvec,ndata,'b--')
        plt.legend(['original','1%'],loc='upper right')
        plt.tight_layout()
        plt.show()
    else:
        plt.plot(tvec,data,'r-');plt.plot(tvec,ndata,'b--')
        plt.legend(['original','1%'],loc='upper right')
        plt.show()

# freq info of the data
nfft = int(next_fast_len(npts))
sfreq = scipy.fftpack.fftfreq(nfft,d=dt)[:nfft//2]
spect = scipy.fftpack.fft(data,nfft,axis=0)[:nfft//2]
spect1= scipy.fftpack.fft(ndata,nfft,axis=0)[:nfft//2]

# check the spectrum for the signals
plot_spect = True
if plot_spect:

    # plot the spectrum
    plt.subplot(211)
    plt.plot(sfreq,np.real(spect),'r-')
    plt.plot(sfreq,np.imag(spect),'b-')
    plt.legend(['amp','pha'],loc='upper right')
    plt.subplot(212)
    plt.plot(sfreq,np.real(spect1),'r-')
    plt.plot(sfreq,np.imag(spect1),'b-')
    plt.legend(['amp','pha'],loc='upper right')
    plt.show()


######################################
##### parameters for monitoring ######
######################################
twin    = [0,lag]                           # targeted time window for waveform monitoring (could be both lags)
freq    = [0.1,0.2,0.4,0.8,1.6]               # targeted frequency band for waveform monitoring
ccomp   = 'ZZ'                              # measurements on which cross-component
onelag  = False                             # make measurement one one lag or two 
norm_flag = True                            # whether to normalize the cross-correlation waveforms

# save parameters as a dic
para = {'twin':twin,'freq':freq,'dt':dt,'ccomp':ccomp,'onelag':onelag,'norm_flag':norm_flag}

# variables for stretching method
epsilon = 2/100                              # limit for dv/v (in decimal)
nbtrial = 50                                # number of increment of dt [-epsilon,epsilon] for the streching

# variables for DTW
maxlag = 50                                 # maxmum points to move (times dt gives the maximum time shifts)
b      = 5                                  # strain limit (to be tested)
direct = 1                                  # direction to accumulate errors (1=forward, -1=backward)

# variables for MWCS & MWCC
move_win_sec = 1.2*int(1/np.min(freq))      # moving window length (in sec)
step_sec = 0.3*move_win_sec                 # step for moving window sliding (in sec)

if move_win_sec > 0.5*(np.max(twin)-np.min(twin)):
    raise IOError('twin too small for MWCS')

# variables for wavelet wct/cwt
dj=1/12                                     # Spacing between discrete scales. Default value is 1/12.
s0=-1                                       # Smallest scale of the wavelet. Default value is 2*dt.
J=-1                                        # Number of scales less one.
wvn='morlet'                                # wavelet class


###############################################
############ monitoring processing ############
###############################################

# load data and do broad filtering
nfreq = len(freq)-1

# allocate matrix
dvv_stretch = np.zeros(shape=(nfreq,2),dtype=np.float32)
dvv_dtw  = np.zeros(shape=(nfreq,2),dtype=np.float32)
dvv_mwcs = np.zeros(shape=(nfreq,2),dtype=np.float32)
dvv_wcc  = np.zeros(shape=(nfreq,2),dtype=np.float32)
dvv_wts  = np.zeros(shape=(nfreq,2),dtype=np.float32)
dvv_wxs  = np.zeros(shape=(nfreq,2),dtype=np.float32)
dvv_wdw  = np.zeros(shape=(nfreq,2),dtype=np.float32)

ref = data
cur = ndata

# loop through each frequency
for jj in range(nfreq):

    # define new freq range for dict of para
    freq1 = freq[jj]
    freq2 = freq[jj+1]
    para['freq'] = [freq1,freq2]
    move_win_sec = 1.2*int(1/freq1)

    # filter waveforms for time/freq domain methods
    ncur = bandpass(cur,freq1,freq2,int(1/dt),corners=4,zerophase=True)
    nref = bandpass(ref,freq1,freq2,int(1/dt),corners=4,zerophase=True)
    if norm_flag:
        ncur /= np.max(ncur)
        nref /= np.max(nref)

    # functions working in time domain
    dvv_stretch[jj,0],dvv_stretch[jj,1],cc,cdp = core_functions.stretching(nref,ncur,epsilon,nbtrial,para)
    dvv_dtw[jj,0],dvv_dtw[jj,1],dist   = core_functions.dtw_dvv(nref,ncur,para,maxlag,b,direct)

    # functions with moving window 
    dvv_mwcs[jj,0],dvv_mwcs[jj,1] = core_functions.mwcs_dvv(nref,ncur,move_win_sec,step_sec,para)
    dvv_wcc[jj,0],dvv_wcc[jj,1]   = core_functions.WCC_dvv(nref,ncur,move_win_sec,step_sec,para)

    allfreq = False  # average dv/v over the frequency band for wts and wxs
    dvv_wts[jj,0],dvv_wts[jj,1] = core_functions.wts_allfreq(ref,cur,allfreq,para,epsilon,nbtrial,dj,s0,J,wvn)
    dvv_wxs[jj,0],dvv_wxs[jj,1] = core_functions.wxs_allfreq(ref,cur,allfreq,para,dj,s0,J)
    dvv_wdw[jj,0],dvv_wdw[jj,1] = core_functions.wtdtw_allfreq(ref,cur,allfreq,para,maxlag,b,direct,dj,s0,J)

allfreq = True     # look at all frequency range
para['freq'] = freq
# functions working in wavelet domain
dfreq,dv_wts,unc5 = core_functions.wts_allfreq(ref,cur,allfreq,para,epsilon,nbtrial,dj,s0,J,wvn)
dfreq,dv_wxs,unc6 = core_functions.wxs_allfreq(ref,cur,allfreq,para,dj,s0,J)
dfreq,dv_wdw,unc7 = core_functions.wtdtw_allfreq(ref,cur,allfreq,para,maxlag,b,direct,dj,s0,J)

###############################################
############ plotting results #################
###############################################

# original and strectched data
plt.subplot(221)
plt.plot(tvec,data,'r-');plt.plot(tvec,ndata,'b--')
plt.xlabel('time [s]')
plt.title('waveform comparision')
plt.legend(['original','1%'],loc='upper right')

# spectrum of original/strectched data
plt.subplot(222)
plt.plot(sfreq,np.abs(spect),'r-')
plt.plot(sfreq,np.abs(spect1),'b-')
plt.xscale('log')
plt.xlabel('freq [Hz]')
plt.title('spectral comparision')
plt.legend(['original','1%'],loc='upper right')
plt.grid('true')

# dv/v at each filtered frequency band
plt.subplot(223)
plt.plot(dvv_stretch[:,0],'o-',markersize=2)
plt.plot(dvv_dtw[:,0],'v-',markersize=2)
plt.plot(dvv_mwcs[:,0],'^-',markersize=2)
plt.plot(dvv_wcc[:,0],'s-',markersize=2)
plt.plot(dvv_wts[:,0],'x-',markersize=2)
plt.plot(dvv_wxs[:,0],'d-',markersize=2)
plt.plot(dvv_wdw[:,0],'+-',markersize=2)
tick1 = np.arange(0,nfreq)
tick2 = []
for jj in range(nfreq):
    tick2.append('%4.2f-%4.2fHz'%(freq[jj],freq[jj+1]))
#plt.xticks([0,1,2],['0.1-0.2Hz','0.2-0.3Hz','0.3-0.5Hz'])
plt.xticks(tick1,tick2)
plt.title('dv/v at each freq band (true anw is 1%)')
plt.legend(['strech','dtw','mwcs','wcc','wts','wxs','wdw'],loc='upper right')
plt.grid('true')

# dv/v from wavelet domain method
plt.subplot(224)
plt.plot(dfreq,dv_wts,'o-',markersize=2)
plt.plot(dfreq,dv_wxs,'v-',markersize=2)
plt.plot(dfreq,dv_wdw,'s-',markersize=2)
plt.xlabel('freq [Hz]')
plt.title('wavelet domain methods (true anw is 1%)')
plt.legend(['wts','wxs','wdw'],loc='upper right')
plt.grid('true')
plt.tight_layout()
plt.show()