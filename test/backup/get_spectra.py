import os
import scipy
import pyasdf 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fftpack import fft,ifft,next_fast_len

'''
look at the spectra of the raw UW noise data
'''

sfile  = '/Volumes/Chengxin/Seattle/RAW_DATA_resp/2018_04_02_00_00_00T2018_04_03_00_00_00.h5'
sdir   = '/Volumes/Chengxin/Seattle/figures/Apr2nd/raw_spect_resp'
if not os.path.isfile(sfile):
    raise ValueError('no file %s exists'%sfile)
freqmin = 0.0332
freqmax = 0.0336

with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
    # extract common variables
    station_list = ds.waveforms.list()
    
    # loop through each station
    for ista in station_list:
        chan_list = ds.waveforms[ista].get_waveform_tags()

        if len(chan_list)==3:
            plt.figure(figsize=(8,12))
            for ic in range(len(chan_list)):
                tr = ds.waveforms[ista][chan_list[ic]]

                if ic==0:
                    npts = tr[0].stats.npts
                    dt   = tr[0].stats.delta
                    nfft  = int(next_fast_len(npts))
                    freq  = scipy.fftpack.fftfreq(nfft,d=dt)[:nfft//2]
                    indx  = np.where((freq>=freqmin)&(freq<=freqmax))
                spec  = scipy.fftpack.fft(tr[0].data,nfft,axis=0)[:nfft//2]

                # plot the spectrum
                tmp = '61'+str(ic*2)
                plt.subplot(tmp)
                plt.xscale('log')
                plt.xlabel('freq [Hz]')
                plt.plot(freq[indx],np.abs(spec[indx]),'r-')
                plt.grid(True,axis='both')
                tmp = '61'+str(ic*2+1)
                plt.subplot(tmp)
                plt.plot(freq[indx],np.angle(spec[indx]),'o')
                plt.grid(True,axis='both')
                plt.xscale('log')
                plt.xlabel('freq [Hz]')
                plt.title('%s %s'%(ista,chan_list[ic]))
                plt.tight_layout()
            if not os.path.isdir(sdir):os.mkdir(sdir)
            outfname = sdir+'/{0:s}.pdf'.format(ista)
            plt.savefig(outfname, format='pdf', dpi=400)
            plt.close()