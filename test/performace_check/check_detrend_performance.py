import os
import glob
import scipy
import time
import pyasdf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from obspy.core.util.base import _get_function_from_entry_point
import obspy

'''
script to test the cut_trace_make_statis function
'''

def cut_trace_make_statis(cc_len,step,source,flag):
    '''
    cut continous noise data into user-defined segments, estimate the statistics of
    each segment and keep timestamp for later use.

    fft_para: dictionary containing all useful variables for the fft step.
    source: obspy stream of noise data.
    flag: boolen variable to output intermediate variables or not.
    '''
    # define return variables first
    source_params=[];dataS_t=[];dataS=[]

    # statistic to detect segments that may be associated with earthquakes
    all_madS = mad(source[0].data)	# median absolute deviation over all noise window
    all_stdS = np.std(source[0].data)	# standard deviation over all noise window
    if all_madS==0 or all_stdS==0 or np.isnan(all_madS) or np.isnan(all_stdS):
        print("continue! madS or stdS equeals to 0 for %s" % source)
        return source_params,dataS_t,dataS

    # inititialize variables
    trace_madS = []
    trace_stdS = []
    nonzeroS = []
    nptsS = []
    source_slice = obspy.Stream()

    #--------break a continous recording into pieces----------
    t0 = time.time()
    for ii,win in enumerate(source[0].slide(window_length=cc_len, step=step)):
	# note: these two steps are the most time consuming. This is to be sped up.
	#	obspy uses scipy, so using scipy does not speed up much.
        win.detrend(type="constant")	# remove mean
        win.detrend(type="linear")	# remove trend
        trace_madS.append(np.max(np.abs(win.data))/all_madS)
        trace_stdS.append(np.max(np.abs(win.data))/all_stdS)
        nonzeroS.append(np.count_nonzero(win.data)/win.stats.npts)
        nptsS.append(win.stats.npts)	# number of points in window
        win.taper(max_percentage=0.05,max_length=20)	# taper window
        source_slice.append(win)	# append slice of tapered noise window
    t1 = time.time()
    print('inside obspy %6.2f'%(t1-t0))

    if len(source_slice) == 0:
        print("No traces for %s " % source)
        return source_params,dataS_t,dataS
    else:
        source_params = np.vstack([trace_madS,trace_stdS,nonzeroS]).T

    Nseg   = len(source_slice)	# number of segments in the original window
    Npts   = np.max(nptsS)	# number of points in the segments
    dataS_t= np.zeros(shape=(Nseg,2),dtype=np.float)	# initialize
    dataS  = np.zeros(shape=(Nseg,Npts),dtype=np.float32)# initialize
    # create array of starttime and endtimes.
    for ii,trace in enumerate(source_slice):
        dataS_t[ii,0]= source_slice[ii].stats.starttime-obspy.UTCDateTime(1970,1,1)# convert to dataframe
        dataS_t[ii,1]= source_slice[ii].stats.endtime -obspy.UTCDateTime(1970,1,1)# convert to dataframe
        dataS[ii,0:nptsS[ii]] = trace.data

    return source_params,dataS_t,dataS


def cut_trace_make_statis1(cc_len,step,inc_hours,source,flag):
    '''
    cut continous noise data into user-defined segments, estimate the statistics of
    each segment and keep timestamp for later use.

    fft_para: dictionary containing all useful variables for the fft step.
    source: obspy stream of noise data.
    flag: boolen variable to output intermediate variables or not.
    '''
    # define return variables first
    source_params=[];dataS_t=[];dataS=[]

    # useful parameters for trace sliding
    nseg = int(np.floor((inc_hours/24*86400-cc_len)/step))
    sps  = int(source[0].stats.sampling_rate)
    starttime = source[0].stats.starttime-obspy.UTCDateTime(1970,1,1)
    # copy data into array
    data = source[0].data

    # statistic to detect segments that may be associated with earthquakes
    all_madS = mad(data)	            # median absolute deviation over all noise window
    all_stdS = np.std(data)	        # standard deviation over all noise window
    if all_madS==0 or all_stdS==0 or np.isnan(all_madS) or np.isnan(all_stdS):
        print("continue! madS or stdS equeals to 0 for %s" % source)
        return source_params,dataS_t,dataS

    # inititialize variables
    npts = cc_len*sps
    trace_madS = np.zeros(nseg,dtype=np.float32)
    trace_stdS = np.zeros(nseg,dtype=np.float32)
    dataS    = np.zeros(shape=(nseg,npts),dtype=np.float32)
    dataS_t  = np.zeros(shape=(nseg,2),dtype=np.float)

    indx1 = 0
    for iseg in range(nseg):
        indx2 = indx1+npts
        dataS[iseg] = adata[indx1:indx2]
        trace_madS[iseg] = (np.max(np.abs(dataS[iseg]))/all_madS)
        trace_stdS[iseg] = (np.max(np.abs(dataS[iseg]))/all_stdS)
        dataS_t[iseg,0]  = starttime+cc_len*iseg
        dataS_t[iseg,1]  = starttime+cc_len*(iseg+1)
        indx1 = indx1+step*sps

    t0=time.time()
    dataS = demean(dataS)
    dataS = detrend(dataS)
    dataS = taper(dataS)
    t1=time.time()
    print('inside new takes %6.2f'%(t1-t0))

    source_params = np.vstack([trace_madS,trace_stdS]).T

    return source_params,dataS_t,dataS

def detrend(data):
    '''
    remove the trend of the signal based on QR decomposion
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        X = np.ones((data.shape[0],2))
        X[:,0] = np.arange(0,data.shape[0])/data.shape[0]
        Q,R = np.linalg.qr(X)
        rq  = np.dot(np.linalg.inv(R),Q.transpose())
        coeff = np.dot(rq,data)
        data = data-np.dot(X,coeff)
    elif data.ndim == 2:
        X = np.ones((data.shape[1],2))
        X[:,0] = np.arange(0,data.shape[1])/data.shape[1]
        Q,R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R),Q.transpose())
        for ii in range(data.shape[0]):
            coeff = np.dot(rq,data[ii])
            data[ii] = data[ii] - np.dot(X,coeff)
    return data

def demean(data):
    '''
    remove the mean of the signal
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        data = data-np.mean(data)
    elif data.ndim == 2:
        for ii in range(data.shape[0]):
            data[ii] = data[ii]-np.mean(data[ii])
    return data

def taper1(data):
    '''
    apply a cosine taper using tukey window
    '''
    ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts  = data.shape[0]
        win   = signal.tukey(npts,alpha=0.05)
        ndata = data*win
    elif data.ndim == 2:
        npts = data.shape[1]
        win   = signal.tukey(npts,alpha=0.05)
        for ii in range(data.shape[0]):
            ndata[ii] = data[ii]*win
    return ndata

def taper(data):
    '''
    apply a cosine taper using tukey window
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]

        # window length
        if npts*0.05>20:wlen = 20
        else:wlen = npts*0.05

        # taper values
        func = _get_function_from_entry_point('taper', 'hann')
        if 2*wlen == npts:
            taper_sides = func(2*wlen)
        else:
            taper_sides = func(2*wlen + 1)

        # taper window
        win  = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),taper_sides[len(taper_sides) - wlen:]))
        data = data*win
    elif data.ndim == 2:
        npts = data.shape[1]

        # window length
        if npts*0.05>20:wlen = 20
        else:wlen = npts*0.05

        # taper values
        func = _get_function_from_entry_point('taper', 'hann')
        if 2*wlen == npts:
            taper_sides = func(2*wlen)
        else:
            taper_sides = func(2*wlen + 1)

        # taper window
        win  = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),taper_sides[len(taper_sides) - wlen:]))
        for ii in range(data.shape[0]):
            data[ii] = data[ii]*win
    return data

def mad(arr):
    """
    Median Absolute Deviation: MAD = median(|Xi- median(X)|)
    :type arr: numpy.ndarray
    :param arr: seismic trace data array
    :return: Median Absolute Deviation of data
    """
    if not np.ma.is_masked(arr):
        med = np.median(arr)
        data = np.median(np.abs(arr - med))
    else:
        med = np.ma.median(arr)
        data = np.ma.median(np.ma.abs(arr-med))
    return data

# path information
rootpath = os.path.join(os.path.expanduser('~'), 'Documents/Harvard/Kanto_basin/Mesonet_BW/noise_data/Event_2010_340')
sacfiles = glob.glob(os.path.join(rootpath,'*.sac'))

# cc info
cc_len = 3600
step   = 900
inc_hours = 24

# loop through each stream
nfile = len(sacfiles)
if not nfile:raise ValueError('no sac data in %s'%rootpath)

for ii in range(nfile):
    source = obspy.read(sacfiles[ii])
    t0=time.time()
    source_params1,dataS_t1,dataS1 = cut_trace_make_statis(cc_len,step,source,1)
    t1=time.time()
    source_params2,dataS_t2,dataS2 = cut_trace_make_statis1(cc_len,step,inc_hours,source,1)
    t2=time.time()
    print('v0 and v1 takes %6.2fs and %6.2fs'%(t1-t0,t2-t1))
