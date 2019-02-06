import os
import sys
import glob
from datetime import datetime
import numpy as np
import scipy
from scipy.fftpack.helper import next_fast_len
from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_sac_taper
from obspy.signal.filter import bandpass
import obspy
import matplotlib.pyplot as plt
import noise_module
import time
import pyasdf
import pandas as pd
from mpi4py import MPI

'''
this script pre-processs the noise data for each single station using the parameters given below 
and stored the whitened and nomalized fft trace for each station in a HDF5 file as *.h5.

this version is implemented with MPI (Nov.09.2018)
by C.Jiang, T.Clements, M.Denolle
'''

t00=time.time()
#------form the absolute paths-------
#locations = '/n/home13/chengxin/cases/KANTO/locations.txt'
#FFTDIR = '/n/flashlfs/mdenolle/KANTO/DATA/FFT/'
#FFTDIR = '/n/regal/denolle_lab/cjiang/FFT'
#event = '/n/flashlfs/mdenolle/KANTO/DATA/????/Event_????_???'
#resp_dir = '/n/flashlfs/mdenolle/KANTO/DATA/resp'

locations = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/locations_small.txt'
FFTDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT_opt'
event = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/noise_data/Event_2010_???'
resp_dir = '/Users/chengxin/Documents/Harvard/Kanto_basin/instrument/resp_all/resp_spectrum_20Hz'

#-----some control parameters------
prepro=False    # do you need to reprocess the data?
to_whiten=False   # do you want to whiten the spectrum?
time_norm=False
rm_resp=True
ftype='mseed'   # file data type
down_sample=False
pre_filt=[0.04,0.05,4,6]
downsamp_freq=20
#dt=1/downsamp_freq
cc_len=3600
step=1800
freqmin=0.05   # minimum frequency to whiten in
freqmax=3   # maximum frequency to whiten in
#method='coherence' # type of normalization
method='deconv' # type of normalization
maxlag=800 # max lag to keep in the correlations
#norm_type='one_bit'#running_mean'
norm_type='running_mean'


#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------


if rank == 0:
    #----define common variables----
    locs = pd.read_csv(locations)
    nsta = len(locs)
    splits = nsta
    tdir = sorted(glob.glob(event))
else:
    splits,locs,tdir = [None for _ in range(3)]


#-----broadcast the variables------
splits = comm.bcast(splits,root=0)
locs = comm.bcast(locs,root=0)
tdir = comm.bcast(tdir,root=0)
extra = splits % size


#--------MPI: loop through each station--------
for ista in range (rank,splits+size-extra,size):
    #time.sleep(rank/1000)

    if ista<splits:
        t10=time.time()
        #----loop through each day on each core----
        for jj in range (len(tdir)):
            station = locs.iloc[ista]['station']
            network = locs.iloc[ista]['network']
            
            tfiles = glob.glob(os.path.join(tdir[jj],'*'+station+'*.sac'))
            if len(tfiles)==0:
                print(str(station)+' does not have sac file at '+str(tdir[jj]))
                continue

            #----loop through each channel----
            for tfile in tfiles:
                sacfile = os.path.basename(tfile)
                comp = sacfile.split('.')[3]
                sta  = sacfile.split('.')[1]
                if (sta != station):
                    print("station names not consistent! "+str(sta)+" is not "+str(station))
                source1 = obspy.read(tfile)
                del sta

                #---------make an inventory---------
                inv1=noise_module.stats2inv(source1[0].stats,locs=locs)

                #------------Pre-Processing-----------
                source = obspy.Stream()
                source = source1.merge(method=1,fill_value=0.)[0]
                
                if prepro:
                    source = noise_module.process_raw(source1, downsamp_freq)
                    source = source.merge(method=1, fill_value=0.)[0]
                
                #----remove instrument response using extracted files: only for Kanto data-----
                if rm_resp:
                    
                    #---------do the downsampling here--------
                    if down_sample:
                        source = noise_module.downsample(source,downsamp_freq)
                    dt=1/source.stats.sampling_rate

                    if source.stats.npts!=downsamp_freq*24*cc_len:
                        print('Moving to next: extraced response file does not match the sac length')
                        continue

                    #-----load the instrument response nyc file-----
                    resp_file = os.path.join(resp_dir,'resp.'+station+'.npy')
                    respz = np.load(resp_file)
                    if not os.path.isfile(resp_file):
                        print("no instrument response for "+station)
                        continue

                    #----------do fft now----------
                    nfft = _npts2nfft(source.stats.npts)
                    source_spect = np.fft.rfft(source.data,n=nfft)

                    fy = 1 / (dt * 2.0)
                    freq = np.linspace(0, fy, nfft // 2 + 1)

                    #-----apply a cosine taper to target freq-----
                    #cos_win = cosine_sac_taper(freq, flimit=pre_filt)
                    #source_spect *=cos_win
                    source_spect *=respz
                    source.data = np.fft.irfft(source_spect)[0:source.stats.npts]
                    source.data=bandpass(source.data, freqmin, freqmax, downsamp_freq, corners=4, zerophase=False)

                #----------variables to define days with earthquakes----------
                all_madS = noise_module.mad(source.data)
                all_stdS = np.std(source.data)

                #-------silly ways to count the total number of windows for sliding-------
                ii=0
                for ii,win in enumerate(source.slide(window_length=cc_len,step=step)):
                    pass
                N=ii+1

                if ii==0:
                    continue

                trace_madS = np.zeros(N)
                trace_stdS = np.zeros(N) 
                nonzeroS = np.zeros(N)
                nptsS =np.zeros((N,),dtype=np.int32)
                source_slice = obspy.Stream()

                #--------breaken a continous recording into pieces----------
                for ii,win in enumerate(source.slide(window_length=cc_len, step=step)):
                    win.detrend(type="constant")
                    win.detrend(type="linear")
                    trace_madS[ii] = np.max(np.abs(win.data))/all_madS
                    trace_stdS[ii] = np.max(np.abs(win.data))/all_stdS
                    nonzeroS[ii] = np.count_nonzero(win.data)/win.stats.npts
                    nptsS[ii] = win.stats.npts
                    win.taper(max_percentage=0.05,max_length=20)
                    source_slice += win
                del source, source1

                if len(source_slice) == 0:
                    print("No traces in Stream of "+str(tfile))

                source_params= np.vstack([trace_madS,trace_stdS,nonzeroS]).T
                del trace_madS, trace_stdS, nonzeroS

                #---------seems un-necesary for data already pre-processed with same length (zero-padding)-------
                NtS = np.max(nptsS)
                dataS_t= np.zeros(shape=(N,2))
                dataS = np.zeros(shape=(N,NtS),dtype=np.float32)
                for ii,trace in enumerate(source_slice):
                    dataS_t[ii,0]= source_slice[ii].stats.starttime-obspy.UTCDateTime(1970,1,1)# convert to dataframe
                    dataS_t[ii,1]= source_slice[ii].stats.endtime -obspy.UTCDateTime(1970,1,1)# convert to dataframe
                    dataS[ii,0:nptsS[ii]] = trace.data
                    if ii==0:
                        dataS_stats=trace.stats

                #------check the dimension of the dataS-------
                if dataS.ndim == 1:
                    axis = 0
                elif dataS.ndim == 2:
                    axis = 1

                #-----------FFT NOW--------------
                Nt = len(source_slice)
                Nfft = int(next_fast_len(int(dataS.shape[axis])))
                
                if Nfft==0:
                    continue

                #-----to whiten or not------
                if to_whiten:
                    source_white = noise_module.whiten(dataS,dt,freqmin,freqmax)
                else:
                    source_white = scipy.fftpack.fft(dataS, Nfft, axis=axis)

                #------to normalize in time or not------
                if time_norm:   
                    white = np.real(scipy.fftpack.ifft(source_white, Nfft, axis=axis)) #/ Nt

                    if norm_type == 'one_bit': 
                        white = np.sign(white)
                    elif norm_type == 'running_mean':
                        white = noise_module.running_abs_mean(white,int(1 / freqmin / 2))
                    source_white = scipy.fftpack.fft(white, Nfft, axis=axis)
                    del white

                #-------------save FFTs as HDF5 files-----------------
                crap=np.zeros(shape=(Nt,Nfft//2),dtype=np.complex64)
                fft_h5 = os.path.join(FFTDIR,locs.iloc[ista]["network"] + "." + locs.iloc[ista]["station"] + '.h5')

                if not os.path.isfile(fft_h5):
                    with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as ds:
                        pass # create pyasdf file 
                #else:
                #    print(locs.iloc[ista]["network"] + "." + locs.iloc[ista]["station"],' already exists',obspy.UTCDateTime())
                
                #tt0=time.time()
                with pyasdf.ASDFDataSet(fft_h5,mpi=False,compression=None) as fft_ds:
                    parameters = noise_module.fft_parameters(dt,cc_len,dataS_stats,dataS_t,source_params, \
                        locs.iloc[ista],comp,Nfft,Nt)
                    
                    savedate = '_'.join((str(dataS_stats.starttime.year),str(dataS_stats.starttime.month), \
                        str(dataS_stats.starttime.day)))
                    savedate = datetime.strptime(savedate,'%Y_%m_%d')
                    savedate = datetime.strftime(savedate,'%Y_%m_%d')
                    path = savedate

                    data_type = str(comp)
                    fft_ds.add_stationxml(inv1)
                    crap[:,:Nfft//2]=source_white[:,:Nfft//2]
                    fft_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

                #tt1=time.time()
                #print('write one day asdf takes '+str(tt1-tt0)+' s')
                del fft_ds, crap, parameters, source_slice, source_white, dataS, dataS_stats, dataS_t, source_params, inv1            

            del tfiles
        t11=time.time()
        print('it takes '+str(t11-t10)+' s to process one station in step 1')

t01=time.time()
print('step1 takes '+str(t01-t00)+' s')

comm.barrier()

if rank == 0:
    sys.exit()
