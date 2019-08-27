import os
import glob
import scipy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt 
from scipy.fftpack import fft,next_fast_len

'''
do selective stacking based on the correlation coefficient
'''

# input files
sfiles  = glob.glob('/Users/chengxin/Documents/NoisePy_example/Kanto/STACK/stack_1800s/E.AYHM/*.h5')
outpath = '/Users/chengxin/Documents/NoisePy_example/Kanto/figures/selective_stacking_cc'
if not os.path.isdir(outpath): os.mkdir(outpath)
ccomp = 'ZZ'

for sfile in sfiles:
    # load input file
    ds = pyasdf.ASDFDataSet(sfile,mode='r')
    alist = ds.auxiliary_data.list()
    sdata = ds.auxiliary_data[alist[0]][ccomp].data[:]
    dt    = ds.auxiliary_data[alist[0]][ccomp].parameters['dt']

    # window info
    nwin  = len(alist[1:])
    nfft  = int(next_fast_len(sdata.size))
    nfreq = scipy.fftpack.fftfreq(nfft,d=dt)[:nfft//2]
    ndata = np.zeros(shape=(len(alist[1:]),sdata.size),dtype=np.float32)
    corr  = np.zeros(len(alist)-1,dtype=np.float32)

    # get spectrum
    sspect = fft(sdata,nfft,axis=0)[:nfft//2]

    # load data matrix
    for ii,ilist in enumerate(alist[1:]):
        try:
            ndata[ii] = ds.auxiliary_data[ilist][ccomp].data[:]
            corr[ii] = np.corrcoef(sdata,ndata[ii])[0,1]
        except Exception:
            continue 

    # find bad corr
    #indx = np.where(corr>0)[0]
    indx  = np.where(np.isnan(corr)==0)[0]
    ndata = ndata[indx]
    ncorr = corr[indx]

    ############################
    ## do selective stacking ###
    ############################

    # remove the last 5%
    tcorr = sorted(ncorr)
    num = int(0.05*len(tcorr))
    if num >1:
        indx1 = np.where(ncorr>tcorr[num])[0]
        sdata1 = np.mean(ndata[indx1],axis=0)
        sspect1 = fft(sdata1,nfft,axis=0)[:nfft//2]
    else:
        raise ValueError('abort! matrix less than 20 rows')

    # remove the last 10% 
    num = int(0.1*len(tcorr))
    if num >1:
        indx2 = np.where(ncorr>tcorr[num])[0]
        sdata2 = np.mean(ndata[indx2],axis=0)
        sspect2 = fft(sdata2,nfft,axis=0)[:nfft//2]
    else:
        raise ValueError('abort! matrix less than 10 rows')

    # remove beyond 2 std
    std = np.std(ncorr)
    ave = np.mean(ncorr)
    indx3 = np.where(ncorr>(ave-2*std))[0]
    sdata3 = np.mean(ndata[indx3],axis=0)
    sspect3 = fft(sdata3,nfft,axis=0)[:nfft//2]

    num = int(0.3*len(tcorr))
    if num >1:
        indx4 = np.where(ncorr>tcorr[num])[0]
        sdata4 = np.mean(ndata[indx4],axis=0)
        sspect4 = fft(sdata4,nfft,axis=0)[:nfft//2]
    else:
        raise ValueError('abort! matrix less than 10 rows')
    print('total %d, remove %d, %d, %d respectively'%(len(ncorr),len(ncorr)-len(indx1),len(ncorr)-len(indx2),len(ncorr)-len(indx3)))

    #####################
    ### plotting now ####
    #####################
    plt.subplot(311)
    plt.plot(sdata,'k-')
    plt.plot(sdata2,'b-')
    plt.plot(sdata4,'r-')
    plt.xlabel('time')
    plt.legend(['orignal','10%','30%'],loc='upper right')
    plt.subplot(312)
    plt.plot(np.abs(sspect),'k-')
    plt.plot(np.abs(sspect2),'b-')
    plt.plot(np.abs(sspect4),'r-')
    plt.legend(['orignal','10%','30%'],loc='upper right')
    plt.subplot(313)
    plt.plot(ncorr)
    plt.plot([0,len(ncorr)],[0,0],'r--')
    plt.ylabel('cc coeff')
    plt.xlabel('segment index')
    plt.tight_layout()

    tmp = sfile.split('/')[-1]
    outfname = outpath+'/{0:s}.pdf'.format(tmp[0:-3])
    plt.savefig(outfname, format='pdf', dpi=400)
    plt.close()