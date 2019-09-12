import os
import glob
import pycwt
import pyasdf
import numpy as np
import matplotlib.pyplot as plt

'''
this function uses cwt to track group wave energy
'''

# some path variables
data_path = '/Volumes/Chengxin/KANTO/STACK_2012'
allfiles = glob.glob(os.path.join(data_path,'*/*.h5'))
sdir = '/Volumes/Chengxin/KANTO/STACK_2012/figures/cwt_group'
dist_max = 30

# load cross-correlation functions
################################
for sfile in allfiles:
    tmp = sfile.split('/')[-1].split('_')
    spair = tmp[0]+'_'+tmp[1][0:6]
    with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
        try:
            maxlag = ds.auxiliary_data['Allstack0linear']['ZZ'].parameters['maxlag']
            dist   = ds.auxiliary_data['Allstack0linear']['ZZ'].parameters['dist']
            dt = ds.auxiliary_data['Allstack0linear']['ZZ'].parameters['dt']
            tdata1 = ds.auxiliary_data['Allstack0linear']['ZZ'].data[:]
            tdata2 = ds.auxiliary_data['Allstack0pws']['ZZ'].data[:]
        except Exception as e:
            print(e);continue

    if dist>dist_max or dist<5:continue 

    # freq bands
    fmin = 0.1
    fmax = 2

    # stack positive and negative lags
    npts = int(1/dt)*2*maxlag+1
    indx = npts//2
    data1 = 0.5*tdata1[indx:]+0.5*np.flip(tdata1[:indx+1],axis=0)
    data2 = 0.5*tdata2[indx:]+0.5*np.flip(tdata2[:indx+1],axis=0)

    # set time window
    vmin = 0.25
    vmax = 4
    t1 = int(dist/vmax/dt)
    t2 = int(dist/vmin/dt)
    if t2>(npts//2):
        t2 = npts//2

    # trim the data
    #lag = 60
    indx = np.arange(t1,t2)
    data1 = data1[indx]
    data2 = data2[indx]
    tvec = indx*dt

    # wavelet transformation
    ################################

    # basic parameters
    dj=1/12
    s0=-1
    J=-1
    wvn='morlet'

    # continous wavelet transform
    cwt1, sj, freq, coi1, _, _ = pycwt.cwt(data1, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi2, _, _ = pycwt.cwt(data2, dt, dj, s0, J, wvn)

    # do filtering here
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: frequency out of limits!')
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt1 = cwt1[freq_ind]
    cwt2 = cwt2[freq_ind]
    freq = freq[freq_ind]

    period = 1/freq
    rcwt1,rcwt2 = np.abs(cwt1)**2,np.abs(cwt2)**2
    pcwt1,pcwt2 = np.real(cwt1),np.real(cwt2)

    # do normalization for each frequency
    for ii in range(len(period)):
        rcwt1[ii] /= np.max(rcwt1[ii])
        rcwt2[ii] /= np.max(rcwt2[ii])
        pcwt1[ii] /= np.max(pcwt1[ii])
        pcwt2[ii] /= np.max(pcwt2[ii])

    # plot wavelet spectrum
    ##########################
    plot_wct = True
    if plot_wct:
        fig,ax = plt.subplots(2,2,figsize=(10,8), sharex=True)
        im1=ax[0,0].imshow(rcwt1,cmap='jet',extent=[0,tvec[-1],np.log2(period[-1]),np.log2(period[0])],aspect='auto')
        ax[0,0].set_xlabel('time [s]')
        ax[0,0].set_ylabel('period [s]')
        ax[0,0].set_title('%s %5.2fkm linear'%(spair,dist))
        Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),np.ceil(np.log2(period.max())))
        ax[0,0].set_yticks(np.log2(Yticks))
        ax[0,0].set_yticklabels(Yticks)
        ax[0,0].xaxis.set_ticks_position('bottom')
        #ax[0,0].fill(np.concatenate([tvec, tvec[-1:]+dt, tvec[-1:]+dt, tvec[:1]-dt, tvec[:1]-dt]), \
        #    np.concatenate([np.log2(coi1), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]), \
        #    'k', alpha=0.3, hatch='x')
        cbar=fig.colorbar(im1,ax=ax[0,0])
        im2=ax[1,0].imshow(pcwt1,cmap='jet',extent=[0,tvec[-1],np.log2(period[-1]),np.log2(period[0])],aspect='auto')
        ax[1,0].set_xlabel('time [s]')
        ax[1,0].set_ylabel('period [s]')
        ax[1,0].set_yticks(np.log2(Yticks))
        ax[1,0].set_yticklabels(Yticks)
        ax[1,0].xaxis.set_ticks_position('bottom')
        cbar=fig.colorbar(im2,ax=ax[1,0])
        #cbar.ax.set_ylabel('accumulated error')
        im3=ax[0,1].imshow(rcwt2,cmap='jet',extent=[0,tvec[-1],np.log2(period[-1]),np.log2(period[0])],aspect='auto')
        ax[0,1].set_xlabel('time [s]')
        ax[0,1].set_ylabel('period [s]')
        ax[0,1].set_title('%s %5.2fkm pws'%(spair,dist))
        ax[0,1].set_yticks(np.log2(Yticks))
        ax[0,1].set_yticklabels(Yticks)
        ax[0,1].xaxis.set_ticks_position('bottom')
        #ax[1].fill(np.concatenate([tvec, tvec[-1:]+dt, tvec[-1:]+dt, tvec[:1]-dt, tvec[:1]-dt]), \
        #    np.concatenate([np.log2(coi2), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]), \
        #    'k', alpha=0.3, hatch='x')
        cbar=fig.colorbar(im3,ax=ax[0,1])
        im4=ax[1,1].imshow(pcwt2,cmap='jet',extent=[0,tvec[-1],np.log2(period[-1]),np.log2(period[0])],aspect='auto')
        ax[1,1].set_xlabel('time [s]')
        ax[1,1].set_ylabel('period [s]')
        ax[1,1].set_yticks(np.log2(Yticks))
        ax[1,1].set_yticklabels(Yticks)
        ax[1,1].xaxis.set_ticks_position('bottom')
        cbar=fig.colorbar(im4,ax=ax[1,1])
        fig.tight_layout()
        #fig.show()

        outfname = sdir+'/'+spair+'.pdf'
        fig.savefig(outfname, format='pdf', dpi=300)
        plt.close()