import os
import glob
import pyasdf
import scipy
import numpy as np
import matplotlib.pyplot as plt 
from obspy.signal.filter import bandpass,highpass
from scipy.fftpack import fft,ifft,next_fast_len

'''
compares the effectiveness of the linear and pws stacking for Kanto basin
'''

# basic info
path = '/Volumes/Chengxin/KANTO/STACK/E.AYHM'
outpath = '/Users/chengxin/Documents/NoisePy_example/Kanto/figures/pws_linear'
if not os.path.isdir(outpath): os.mkdir(outpath)
lfiles = sorted(glob.glob(os.path.join(path,'linear_*.h5')))
pfiles = sorted(glob.glob(os.path.join(path,'pws_*.h5')))
freq  = [0.05,5]
ccomp = 'ZZ'


# loop through all station pairs
for ii in range(len(lfiles)):
    lfile = lfiles[ii]
    pfile = pfiles[ii]

    # load data
    ds1 = pyasdf.ASDFDataSet(lfile,mode='r')
    ds2 = pyasdf.ASDFDataSet(pfile,mode='r')
    try:
        data1 = ds1.auxiliary_data['Allstack'][ccomp].data[:]
        data2 = ds2.auxiliary_data['Allstack'][ccomp].data[:]
        dt = ds1.auxiliary_data['Allstack'][ccomp].parameters['dt']
        dist = ds1.auxiliary_data['Allstack'][ccomp].parameters['dist']
    except Exception:
        print('abort! cannot open %s or %s to read'%(lfile,pfile))
        continue

    # make filtering
    #ndata1 = bandpass(data1,min(freq),max(freq),df=int(1/dt),corners=4,zerophase=True)
    #ndata2 = bandpass(data2,min(freq),max(freq),df=int(1/dt),corners=4,zerophase=True)
    ndata1 = highpass(data1,min(freq),df=int(1/dt),corners=4,zerophase=True)
    ndata2 = highpass(data2,min(freq),df=int(1/dt),corners=4,zerophase=True)

    # get spectrum 
    nfft = int(next_fast_len(len(data1)))
    nfft2 = nfft//2
    nfreq = scipy.fftpack.fftfreq(nfft, d=dt)[:nfft2]
    ntime = np.arange(-len(data1)//2,len(data1)//2)*dt
    spect1 = fft(data1,nfft,axis=0)
    spect2 = fft(data2,nfft,axis=0)

    # plotting now
    plt.subplot(211)
    plt.plot(ntime,ndata1/max(ndata1),'r-')
    plt.plot(ntime,ndata2/max(ndata2),'g-')
    plt.title('%s %6.2fkm @%5.2f-%5.2fHz' % (lfile.split('/')[-1],dist,min(freq),max(freq)))
    plt.legend(['linear','pws'],loc='upper right')
    plt.xlabel('time[s]')

    plt.subplot(212)
    plt.plot(nfreq,np.abs(spect1[:nfft2])/max(np.abs(spect1[:nfft2])),'r-')
    plt.plot(nfreq,np.abs(spect2[:nfft2])/max(np.abs(spect2[:nfft2])),'g-')
    plt.legend(['linear','pws'],loc='upper right')
    plt.xlabel('freq[Hz]')
    #plt.show()

    tmp,sta1,sta2 = lfile.split('/')[-1].split('_')
    outfname = outpath+'/{0:s}_{1:s}.pdf'.format(sta1,sta2[0:6])
    plt.savefig(outfname, format='pdf', dpi=400)
    plt.close()