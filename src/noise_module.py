import os
import glob
import copy
import obspy
import scipy
import time
import pycwt
import pyasdf
import datetime
import numpy as np
import pandas as pd
from numba import jit
from scipy.signal import hilbert
from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_taper
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.filter import bandpass,lowpass
from obspy.signal.regression import linear_regression
from obspy.core.util.base import _get_function_from_entry_point
from obspy.core.inventory import Inventory, Network, Station, Channel, Site


'''
This VERY LONG noise module file is necessary to keep the NoisePy working properly. In general,
the modules are organized based on their functionality in the following way. it includes:

1) core functions called directly by the main NoisePy scripts;
2) utility functions used by the core functions;
3) monitoring functions representing different methods to measure dv/v;
4) monitoring utility functions used by the monitoring functions.

by: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
    Marine Denolle (mdenolle@fas.harvard.edu)

several utility functions are modified based on https://github.com/tclements/noise
'''

####################################################
############## CORE FUNCTIONS ######################
####################################################

def get_event_list(str1,str2,inc_hours):
    '''
    this function calculates the event list between time1 and time2 by increment of inc_hours
    in the formate of %Y_%m_%d_%H_%M_%S' (used in S0A & S0B)
    PARAMETERS:
    ----------------
    str1: string of the starting time -> 2010_01_01_0_0
    str2: string of the ending time -> 2010_10_11_0_0
    inc_hours: integer of incremental hours
    RETURNS:
    ----------------
    event: a numpy character list
    '''
    date1=str1.split('_')
    date2=str2.split('_')
    y1=int(date1[0]);m1=int(date1[1]);d1=int(date1[2])
    h1=int(date1[3]);mm1=int(date1[4]);mn1=int(date1[5])
    y2=int(date2[0]);m2=int(date2[1]);d2=int(date2[2])
    h2=int(date2[3]);mm2=int(date2[4]);mn2=int(date2[5])

    d1=datetime.datetime(y1,m1,d1,h1,mm1,mn1)
    d2=datetime.datetime(y2,m2,d2,h2,mm2,mn2)
    dt=datetime.timedelta(hours=inc_hours)

    event = []
    while(d1<d2):
        event.append(d1.strftime('%Y_%m_%d_%H_%M_%S'))
        d1+=dt
    event.append(d2.strftime('%Y_%m_%d_%H_%M_%S'))

    return event

def make_timestamps(prepro_para):
    '''
    this function prepares the timestamps of both the starting and ending time of each mseed/sac file that
    is stored on local machine. this time info is used to search all stations in specific time chunck
    when preparing noise data in ASDF format. it creates a csv file containing all timestamp info if the
    file does not exist (used in S0B)f
    PARAMETERS:
    -----------------------
    prepro_para: a dic containing all pre-processing parameters used in S0B
    RETURNS:
    -----------------------
    all_stimes: numpy float array containing startting and ending time for all SAC/mseed files
    '''
    # load parameters from para dic
    wiki_file = prepro_para['wiki_file']
    messydata = prepro_para['messydata']
    RAWDATA   = prepro_para['RAWDATA']
    allfiles_path = prepro_para['allfiles_path']

    if os.path.isfile(wiki_file):
        tmp = pd.read_csv(wiki_file)
        allfiles = tmp['names']
        all_stimes = np.zeros(shape=(len(allfiles),2),dtype=np.float)
        all_stimes[:,0] = tmp['starttime']
        all_stimes[:,1] = tmp['endtime']

    # have to read each sac/mseed data one by one
    else:
        allfiles = glob.glob(allfiles_path)
        nfiles   = len(allfiles)
        if not nfiles: raise ValueError('Abort! no data found in subdirectory of %s'%RAWDATA)
        all_stimes = np.zeros(shape=(nfiles,2),dtype=np.float)

        if messydata:
            # get VERY precise trace-time from the header
            for ii in range(nfiles):
                try:
                    tr = obspy.read(allfiles[ii])
                    all_stimes[ii,0] = tr[0].stats.starttime-obspy.UTCDateTime(1970,1,1)
                    all_stimes[ii,1] = tr[0].stats.endtime-obspy.UTCDateTime(1970,1,1)
                except Exception as e:
                    print(e);continue
        else:
            # get rough estimates of the time based on the folder: need modified to accommodate your data
            for ii in range(nfiles):
                year  = int(allfiles[ii].split('/')[-2].split('_')[1])
                #julia = int(allfiles[ii].split('/')[-2].split('_')[2])
                #all_stimes[ii,0] = obspy.UTCDateTime(year=year,julday=julia)-obspy.UTCDateTime(year=1970,month=1,day=1)
                month = int(allfiles[ii].split('/')[-2].split('_')[2])
                day   = int(allfiles[ii].split('/')[-2].split('_')[3])
                all_stimes[ii,0] = obspy.UTCDateTime(year=year,month=month,day=day)-obspy.UTCDateTime(year=1970,month=1,day=1)
                all_stimes[ii,1] = all_stimes[ii,0]+86400

        # save name and time info for later use if the file not exist
        if not os.path.isfile(wiki_file):
            wiki_info = {'names':allfiles,'starttime':all_stimes[:,0],'endtime':all_stimes[:,1]}
            df = pd.DataFrame(wiki_info,columns=['names','starttime','endtime'])
            df.to_csv(wiki_file)
    return all_stimes

def preprocess_raw(st,inv,prepro_para,date_info):
    '''
    this function pre-processes the raw data stream by:
        1) check samping rate and gaps in the data;
        2) remove sigularity, trend and mean of each trace
        3) filter and correct the time if integer time are between sampling points
        4) remove instrument responses with selected methods including:
            "inv"   -> using inventory information to remove_response;
            "spectrum"   -> use the inverse of response spectrum. (a script is provided in additional_module to estimate response spectrum from RESP files)
            "RESP_files" -> use the raw download RESP files
            "polezeros"  -> use pole/zero info for a crude correction of response
        5) trim data to a day-long sequence and interpolate it to ensure starting at 00:00:00.000
    (used in S0A & S0B)
    PARAMETERS:
    -----------------------
    st:  obspy stream object, containing noise data to be processed
    inv: obspy inventory object, containing stations info
    prepro_para: dict containing fft parameters, such as frequency bands and selection for instrument response removal etc.
    date_info:   dict of start and end time of the stream data
    RETURNS:
    -----------------------
    ntr: obspy stream object of cleaned, merged and filtered noise data
    '''
    # load paramters from fft dict
    rm_resp       = prepro_para['rm_resp']
    if 'rm_resp_out' in prepro_para.keys():
        rm_resp_out   = prepro_para['rm_resp_out']
    else:
        rm_resp_out   = 'VEL'
    respdir       = prepro_para['respdir']
    freqmin       = prepro_para['freqmin']
    freqmax       = prepro_para['freqmax']
    samp_freq     = prepro_para['samp_freq']

    # parameters for butterworth filter
    f1 = 0.9*freqmin;f2=freqmin
    if 1.1*freqmax > 0.45*samp_freq:
        f3 = 0.4*samp_freq
        f4 = 0.45*samp_freq
    else:
        f3 = freqmax
        f4= 1.1*freqmax
    pre_filt  = [f1,f2,f3,f4]

    # check sampling rate and trace length
    st = check_sample_gaps(st,date_info)
    if len(st) == 0:
        print('No traces in Stream: Continue!');return st
    sps = int(st[0].stats.sampling_rate)
    station = st[0].stats.station

    # remove nan/inf, mean and trend of each trace before merging
    for ii in range(len(st)):

        #-----set nan/inf values to zeros (it does happens!)-----
        tttindx = np.where(np.isnan(st[ii].data))
        if len(tttindx) >0:st[ii].data[tttindx]=0
        tttindx = np.where(np.isinf(st[ii].data))
        if len(tttindx) >0:st[ii].data[tttindx]=0

        st[ii].data = np.float32(st[ii].data)
        st[ii].data = scipy.signal.detrend(st[ii].data,type='constant')
        st[ii].data = scipy.signal.detrend(st[ii].data,type='linear')

    # merge, taper and filter the data
    if len(st)>1:st.merge(method=1,fill_value=0)
    st[0].taper(max_percentage=0.05,max_length=50)	# taper window
    st[0].data = np.float32(bandpass(st[0].data,pre_filt[0],pre_filt[-1],df=sps,corners=4,zerophase=True))

    # make downsampling if needed
    if abs(samp_freq-sps) > 1E-4:
        # downsampling here
        st.interpolate(samp_freq,method='weighted_average_slopes')
        delta = st[0].stats.delta

        # when starttimes are between sampling points
        fric = st[0].stats.starttime.microsecond%(delta*1E6)
        if fric>1E-4:
            st[0].data = segment_interpolate(np.float32(st[0].data),float(fric/(delta*1E6)))
            #--reset the time to remove the discrepancy---
            st[0].stats.starttime-=(fric*1E-6)

    # remove traces of too small length

    # options to remove instrument response
    if rm_resp != 'no':
        if rm_resp != 'inv':
            if (respdir is None) or (not os.path.isdir(respdir)):
                raise ValueError('response file folder not found! abort!')

        if rm_resp == 'inv':
            #----check whether inventory is attached----
            if not inv[0][0][0].response:
                raise ValueError('no response found in the inventory! abort!')
            else:
                try:
                    print('removing response for %s using inv'%st[0])
                    st[0].attach_response(inv)
                    st[0].remove_response(output=rm_resp_out,pre_filt=pre_filt,water_level=60)
                except Exception:
                    st = []
                    return st

        elif rm_resp == 'spectrum':
            print('remove response using spectrum')
            specfile = glob.glob(os.path.join(respdir,'*'+station+'*'))
            if len(specfile)==0:
                raise ValueError('no response sepctrum found for %s' % station)
            st = resp_spectrum(st,specfile[0],samp_freq,pre_filt)

        elif rm_resp == 'RESP':
            print('remove response using RESP files')
            resp = glob.glob(os.path.join(respdir,'RESP.'+station+'*'))
            if len(resp)==0:
                raise ValueError('no RESP files found for %s' % station)
            seedresp = {'filename':resp[0],'date':date_info['starttime'],'units':'DIS'}
            st.simulate(paz_remove=None,pre_filt=pre_filt,seedresp=seedresp[0])

        elif rm_resp == 'polozeros':
            print('remove response using polos and zeros')
            paz_sts = glob.glob(os.path.join(respdir,'*'+station+'*'))
            if len(paz_sts)==0:
                raise ValueError('no polozeros found for %s' % station)
            st.simulate(paz_remove=paz_sts[0],pre_filt=pre_filt)

        else:
            raise ValueError('no such option for rm_resp! please double check!')

    ntr = obspy.Stream()
    # trim a continous segment into user-defined sequences
    st[0].trim(starttime=date_info['starttime'],endtime=date_info['endtime'],pad=True,fill_value=0)
    ntr.append(st[0])

    return ntr


def stats2inv(stats,prepro_para,locs=None):
    '''
    this function creates inventory given the stats parameters in an obspy stream or a station list.
    (used in S0B)
    PARAMETERS:
    ------------------------
    stats: obspy trace stats object containing all station header info
    prepro_para: dict containing fft parameters, such as frequency bands and selection for instrument response removal etc.
    locs:  panda data frame of the station list. it is needed for convering miniseed files into ASDF
    RETURNS:
    ------------------------
    inv: obspy inventory object of all station info to be used later
    '''
    staxml    = prepro_para['stationxml']
    respdir   = prepro_para['respdir']
    input_fmt = prepro_para['input_fmt']

    if staxml:
        if not respdir:
            raise ValueError('Abort! staxml is selected but no directory is given to access the files')
        else:
            invfile = glob.glob(os.path.join(respdir,'*'+stats.station+'*'))
            if os.path.isfile(str(invfile)):
                inv = obspy.read_inventory(invfile)
                return inv

    inv = Inventory(networks=[],source="homegrown")

    if input_fmt=='sac':
        net = Network(
            # This is the network code according to the SEED standard.
            code=stats.network,
            stations=[],
            description="created from SAC and resp files",
            start_date=stats.starttime)

        sta = Station(
            # This is the station code according to the SEED standard.
            code=stats.station,
            latitude=stats.sac["stla"],
            longitude=stats.sac["stlo"],
            elevation=stats.sac["stel"],
            creation_date=stats.starttime,
            site=Site(name="First station"))

        cha = Channel(
            # This is the channel code according to the SEED standard.
            code=stats.channel,
            # This is the location code according to the SEED standard.
            location_code=stats.location,
            # Note that these coordinates can differ from the station coordinates.
            latitude=stats.sac["stla"],
            longitude=stats.sac["stlo"],
            elevation=stats.sac["stel"],
            depth=-stats.sac["stel"],
            azimuth=stats.sac["cmpaz"],
            dip=stats.sac["cmpinc"],
            sample_rate=stats.sampling_rate)

    elif input_fmt == 'mseed':
        ista=locs[locs['station']==stats.station].index.values.astype('int64')[0]

        net = Network(
            # This is the network code according to the SEED standard.
            code=locs.iloc[ista]["network"],
            stations=[],
            description="created from SAC and resp files",
            start_date=stats.starttime)

        sta = Station(
            # This is the station code according to the SEED standard.
            code=locs.iloc[ista]["station"],
            latitude=locs.iloc[ista]["latitude"],
            longitude=locs.iloc[ista]["longitude"],
            elevation=locs.iloc[ista]["elevation"],
            creation_date=stats.starttime,
            site=Site(name="First station"))

        cha = Channel(
            code=stats.channel,
            location_code=stats.location,
            latitude=locs.iloc[ista]["latitude"],
            longitude=locs.iloc[ista]["longitude"],
            elevation=locs.iloc[ista]["elevation"],
            depth=-locs.iloc[ista]["elevation"],
            azimuth=0,
            dip=0,
            sample_rate=stats.sampling_rate)

    response = obspy.core.inventory.response.Response()

    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)

    return inv


def sta_info_from_inv(inv):
    '''
    this function outputs station info from the obspy inventory object
    (used in S0B)
    PARAMETERS:
    ----------------------
    inv: obspy inventory object
    RETURNS:
    ----------------------
    sta: station name
    net: netowrk name
    lon: longitude of the station
    lat: latitude of the station
    elv: elevation of the station
    location: location code of the station
    '''
    # load from station inventory
    sta = inv[0][0].code
    net = inv[0].code
    lon = inv[0][0].longitude
    lat = inv[0][0].latitude
    if inv[0][0].elevation:
        elv = inv[0][0].elevation
    else: elv = 0.

    if inv[0][0][0].location_code:
        location = inv[0][0][0].location_code
    else: location = '00'

    return sta,net,lon,lat,elv,location


def cut_trace_make_stat(fc_para,source):
    '''
    this function cuts continous noise data into user-defined segments, estimate the statistics of
    each segment and keep timestamp of each segment for later use. (used in S1)
    PARAMETERS:
    ----------------------
    fft_para: A dictionary containing all fft and cc parameters.
    source: obspy stream object
    RETURNS:
    ----------------------
    trace_stdS: standard deviation of the noise amplitude of each segment
    dataS_t:    timestamps of each segment
    dataS:      2D matrix of the segmented data
    '''
    # define return variables first
    source_params=[];dataS_t=[];dataS=[]

    # load parameter from dic
    inc_hours = fc_para['inc_hours']
    cc_len    = fc_para['cc_len']
    step      = fc_para['step']

    # useful parameters for trace sliding
    nseg = int(np.floor((inc_hours/24*86400-cc_len)/step))
    sps  = int(source[0].stats.sampling_rate)
    starttime = source[0].stats.starttime-obspy.UTCDateTime(1970,1,1)
    # copy data into array
    data = source[0].data

    # if the data is shorter than the tim chunck, return zero values
    if data.size < sps*inc_hours*3600:
        return source_params,dataS_t,dataS

    # statistic to detect segments that may be associated with earthquakes
    all_madS = mad(data)	            # median absolute deviation over all noise window
    all_stdS = np.std(data)	        # standard deviation over all noise window
    if all_madS==0 or all_stdS==0 or np.isnan(all_madS) or np.isnan(all_stdS):
        print("continue! madS or stdS equals to 0 for %s" % source)
        return source_params,dataS_t,dataS

    # initialize variables
    npts = cc_len*sps
    #trace_madS = np.zeros(nseg,dtype=np.float32)
    trace_stdS = np.zeros(nseg,dtype=np.float32)
    dataS    = np.zeros(shape=(nseg,npts),dtype=np.float32)
    dataS_t  = np.zeros(nseg,dtype=np.float)

    indx1 = 0
    for iseg in range(nseg):
        indx2 = indx1+npts
        dataS[iseg] = data[indx1:indx2]
        #trace_madS[iseg] = (np.max(np.abs(dataS[iseg]))/all_madS)
        trace_stdS[iseg] = (np.max(np.abs(dataS[iseg]))/all_stdS)
        dataS_t[iseg]    = starttime+step*iseg
        indx1 = indx1+step*sps

    # 2D array processing
    dataS = demean(dataS)
    dataS = detrend(dataS)
    dataS = taper(dataS)

    return trace_stdS,dataS_t,dataS


def noise_processing(fft_para,dataS):
    '''
    this function performs time domain and frequency domain normalization if needed. in real case, we prefer use include
    the normalization in the cross-correaltion steps by selecting coherency or decon (Prieto et al, 2008, 2009; Denolle et al, 2013)
    PARMAETERS:
    ------------------------
    fft_para: dictionary containing all useful variables used for fft and cc
    dataS: 2D matrix of all segmented noise data
    # OUTPUT VARIABLES:
    source_white: 2D matrix of data spectra
    '''
    # load parameters first
    time_norm   = fft_para['time_norm']
    freq_norm   = fft_para['freq_norm']
    smooth_N    = fft_para['smooth_N']
    N = dataS.shape[0]

    #------to normalize in time or not------
    if time_norm != 'no':

        if time_norm == 'one_bit': 	# sign normalization
            white = np.sign(dataS)
        elif time_norm == 'rma': # running mean: normalization over smoothed absolute average
            white = np.zeros(shape=dataS.shape,dtype=dataS.dtype)
            for kkk in range(N):
                white[kkk,:] = dataS[kkk,:]/moving_ave(np.abs(dataS[kkk,:]),smooth_N)

    else:	# don't normalize
        white = dataS

    #-----to whiten or not------
    if freq_norm != 'no':
        source_white = whiten(white,fft_para)	# whiten and return FFT
    else:
        Nfft = int(next_fast_len(int(dataS.shape[1])))
        source_white = scipy.fftpack.fft(white, Nfft, axis=1) # return FFT

    return source_white


def smooth_source_spect(cc_para,fft1):
    '''
    this function smoothes amplitude spectrum of the 2D spectral matrix. (used in S1)
    PARAMETERS:
    ---------------------
    cc_para: dictionary containing useful cc parameters
    fft1:    source spectrum matrix

    RETURNS:
    ---------------------
    sfft1: complex numpy array with normalized spectrum
    '''
    cc_method = cc_para['cc_method']
    smoothspect_N = cc_para['smoothspect_N']

    if cc_method == 'deconv':

        #-----normalize single-station cc to z component-----
        temp = moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = np.conj(fft1)/temp**2
        except Exception:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'coherency':
        temp = moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = np.conj(fft1)/temp
        except Exception:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'xcorr':
        sfft1 = np.conj(fft1)

    else:
        raise ValueError('no correction correlation method is selected at L59')

    return sfft1

def correlate(fft1_smoothed_abs,fft2,D,Nfft,dataS_t):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)
    PARAMETERS:
    ---------------------
    fft1_smoothed_abs: smoothed power spectral density of the FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    dataS_t: matrix of datetime object.
    
    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function

    MODIFICATIONS:
    ---------------------
    output the linear stack of each time chunk even when substack is selected (by Chengxin @Aug2020)
    '''
    #----load paramters----
    dt      = D['dt']
    maxlag  = D['maxlag']
    method  = D['cc_method']
    cc_len  = D['cc_len']
    substack= D['substack']
    substack_len  = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    nwin  = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size,)*fft2.reshape(fft2.size,)

    if method == "coherency":
        temp = moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    if substack:
        if substack_len == cc_len:
            # choose to keep all fft data for a day
            s_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)   # stacked correlation
            ampmax = np.zeros(nwin,dtype=np.float32)
            n_corr = np.zeros(nwin,dtype=np.int16)                  # number of correlations for each substack
            t_corr = dataS_t                                        # timestamp
            crap   = np.zeros(Nfft,dtype=np.complex64)
            for i in range(nwin):
                n_corr[i]= 1
                crap[:Nfft2] = corr[i,:]
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = dataS_t[-1]-dataS_t[0]             # total duration of what we have now
            tstart = dataS_t[0]

            nstack = int(np.round(Ttotal/substack_len))
            ampmax = np.zeros(nstack,dtype=np.float32)
            s_corr = np.zeros(shape=(nstack,Nfft),dtype=np.float32)
            n_corr = np.zeros(nstack,dtype=np.int)
            t_corr = np.zeros(nstack,dtype=np.float)
            crap   = np.zeros(Nfft,dtype=np.complex64)

            for istack in range(nstack):
                # find the indexes of all of the windows that start or end within
                itime = np.where( (dataS_t >= tstart) & (dataS_t < tstart+substack_len) )[0]
                if len(itime)==0:tstart+=substack_len;continue

                crap[:Nfft2] = np.mean(corr[itime,:],axis=0)   # linear average of the correlation
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[istack,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(itime)               # number of windows stacks
                t_corr[istack] = tstart                   # save the time stamps
                tstart += substack_len
                #print('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        ampmax = np.max(corr,axis=1)
        tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
        n_corr = nwin
        s_corr = np.zeros(Nfft,dtype=np.float32)
        t_corr = dataS_t[0]
        crap   = np.zeros(Nfft,dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr[tindx],axis=0)
        crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2],axis=0)
        crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
        s_corr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2+1, Nfft2)*dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim==1:
        s_corr = s_corr[ind]
    elif s_corr.ndim==2:
        s_corr = s_corr[:,ind]
    return s_corr,t_corr,n_corr

def correlate_nonlinear_stack(fft1_smoothed_abs,fft2,D,Nfft,dataS_t):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)
    PARAMETERS:
    ---------------------
    fft1_smoothed_abs: smoothed power spectral density of the FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    dataS_t: matrix of datetime object.
    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function
    '''
    #----load paramters----
    dt      = D['dt']
    maxlag  = D['maxlag']
    method  = D['cc_method']
    cc_len  = D['cc_len']
    substack= D['substack']
    stack_method  = D['stack_method']
    substack_len  = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    nwin  = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size,)*fft2.reshape(fft2.size,)

    # normalize by receiver spectral for coherency
    if method == "coherency":
        temp = moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    # transform back to time domain waveforms
    s_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)   # stacked correlation
    ampmax = np.zeros(nwin,dtype=np.float32)
    n_corr = np.zeros(nwin,dtype=np.int16)                  # number of correlations for each substack
    t_corr = dataS_t                                        # timestamp
    crap   = np.zeros(Nfft,dtype=np.complex64)
    for i in range(nwin):
        n_corr[i]= 1
        crap[:Nfft2] = corr[i,:]
        crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
        crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
        crap[0]=complex(0,0)
        s_corr[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    ns_corr = s_corr
    for iii in range(ns_corr.shape[0]):
        ns_corr[iii] /= np.max(np.abs(ns_corr[iii]))

    if substack:
        if substack_len == cc_len:

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = dataS_t[-1]-dataS_t[0]             # total duration of what we have now
            tstart = dataS_t[0]

            nstack = int(np.round(Ttotal/substack_len))
            ampmax = np.zeros(nstack,dtype=np.float32)
            s_corr = np.zeros(shape=(nstack,Nfft),dtype=np.float32)
            n_corr = np.zeros(nstack,dtype=np.int)
            t_corr = np.zeros(nstack,dtype=np.float)
            crap   = np.zeros(Nfft,dtype=np.complex64)

            for istack in range(nstack):
                # find the indexes of all of the windows that start or end within
                itime = np.where( (dataS_t >= tstart) & (dataS_t < tstart+substack_len) )[0]
                if len(itime)==0:tstart+=substack_len;continue

                crap[:Nfft2] = np.mean(corr[itime,:],axis=0)   # linear average of the correlation
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[istack,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(itime)               # number of windows stacks
                t_corr[istack] = tstart                   # save the time stamps
                tstart += substack_len
                #print('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        if stack_method == 'linear':
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = np.mean(s_corr[tindx],axis=0)
            t_corr = dataS_t[0]
            n_corr = len(tindx)
        elif stack_method == 'robust':
            print('do robust substacking')
            s_corr = robust_stack(s_corr,0.001)
            t_corr = dataS_t[0]
            n_corr = nwin
      #  elif stack_method == 'selective':
      #      print('do selective substacking')
      #      s_corr = selective_stack(s_corr,0.001)
      #      t_corr = dataS_t[0]
      #      n_corr = nwin

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2+1, Nfft2)*dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim==1:
        s_corr = s_corr[ind]
    elif s_corr.ndim==2:
        s_corr = s_corr[:,ind]
    return s_corr,t_corr,n_corr,ns_corr[:,ind]

def cc_parameters(cc_para,coor,tcorr,ncorr,comp):
    '''
    this function assembles the parameters for the cc function, which is used
    when writing them into ASDF files
    PARAMETERS:
    ---------------------
    cc_para: dict containing parameters used in the fft_cc step
    coor:    dict containing coordinates info of the source and receiver stations
    tcorr:   timestamp matrix
    ncorr:   matrix of number of good segments for each sub-stack/final stack
    comp:    2 character strings for the cross correlation component
    RETURNS:
    ------------------
    parameters: dict containing above info used for later stacking/plotting
    '''
    latS = coor['latS']
    lonS = coor['lonS']
    latR = coor['latR']
    lonR = coor['lonR']
    dt        = cc_para['dt']
    maxlag    = cc_para['maxlag']
    substack  = cc_para['substack']
    cc_method = cc_para['cc_method']

    dist,azi,baz = obspy.geodetics.base.gps2dist_azimuth(latS,lonS,latR,lonR)
    parameters = {'dt':dt,
        'maxlag':int(maxlag),
        'dist':np.float32(dist/1000),
        'azi':np.float32(azi),
        'baz':np.float32(baz),
        'lonS':np.float32(lonS),
        'latS':np.float32(latS),
        'lonR':np.float32(lonR),
        'latR':np.float32(latR),
        'ngood':ncorr,
        'cc_method':cc_method,
        'time':tcorr,
        'substack':substack,
        'comp':comp}
    return parameters

def stacking(cc_array,cc_time,cc_ngood,stack_para):
    '''
    this function stacks the cross correlation data according to the user-defined substack_len parameter

    PARAMETERS:
    ----------------------
    cc_array: 2D numpy float32 matrix containing all segmented cross-correlation data
    cc_time:  1D numpy array of timestamps for each segment of cc_array
    cc_ngood: 1D numpy int16 matrix showing the number of segments for each sub-stack and/or full stack
    stack_para: a dict containing all stacking parameters

    RETURNS:
    ----------------------
    cc_array, cc_ngood, cc_time: same to the input parameters but with abnormal cross-correaltions removed
    allstacks1: 1D matrix of stacked cross-correlation functions over all the segments
    nstacks:    number of overall segments for the final stacks
    '''
    # load useful parameters from dict
    samp_freq = stack_para['samp_freq']
    smethod   = stack_para['stack_method']
    start_date   = stack_para['start_date']
    end_date     = stack_para['end_date']
    npts = cc_array.shape[1]

    # remove abnormal data
    ampmax = np.max(cc_array,axis=1)
    tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
    if not len(tindx):
        allstacks1=[];allstacks2=[];allstacks3=[];nstacks=0
        cc_array=[];cc_ngood=[];cc_time=[]
        return cc_array,cc_ngood,cc_time,allstacks1,allstacks2,allstacks3,nstacks
    else:

        # remove ones with bad amplitude
        cc_array = cc_array[tindx,:]
        cc_time  = cc_time[tindx]
        cc_ngood = cc_ngood[tindx]

        # do stacking
        allstacks1 = np.zeros(npts,dtype=np.float32)
        allstacks2 = np.zeros(npts,dtype=np.float32)
        allstacks3 = np.zeros(npts,dtype=np.float32)
        allstacks4 = np.zeros(npts,dtype=np.float32)
        allstacks5 = np.zeros(npts,dtype=np.float32)

        if smethod == 'linear':
            allstacks1 = np.mean(cc_array,axis=0)
        elif smethod == 'pws':
            allstacks1 = pws(cc_array,samp_freq)
        elif smethod == 'robust':
            allstacks1,w,nstep = robust_stack(cc_array,0.001)
        elif smethod == 'auto_covariance':
            allstacks1 = adaptive_filter(cc_array,1)
        elif smethod == 'nroot':
            allstacks1 = nroot_stack(cc_array,2)
        elif smethod == 'all':
            allstacks1 = np.mean(cc_array,axis=0)
            allstacks2 = pws(cc_array,samp_freq)
            allstacks3,w,nstep = robust_stack(cc_array,0.001)
            allstacks4 = adaptive_filter(cc_array,1)
            allstacks5 = nroot_stack(cc_array,2)
        nstacks = np.sum(cc_ngood)

    # good to return
    return cc_array,cc_ngood,cc_time,allstacks1,allstacks2,allstacks3,nstacks


def stacking_rma(cc_array,cc_time,cc_ngood,stack_para):
    '''
    this function stacks the cross correlation data according to the user-defined substack_len parameter
    PARAMETERS:
    ----------------------
    cc_array: 2D numpy float32 matrix containing all segmented cross-correlation data
    cc_time:  1D numpy array of timestamps for each segment of cc_array
    cc_ngood: 1D numpy int16 matrix showing the number of segments for each sub-stack and/or full stack
    stack_para: a dict containing all stacking parameters
    RETURNS:
    ----------------------
    cc_array, cc_ngood, cc_time: same to the input parameters but with abnormal cross-correaltions removed
    allstacks1: 1D matrix of stacked cross-correlation functions over all the segments
    nstacks:    number of overall segments for the final stacks
    '''
    # load useful parameters from dict
    samp_freq = stack_para['samp_freq']
    smethod   = stack_para['stack_method']
    rma_substack = stack_para['rma_substack']
    rma_step     = stack_para['rma_step']
    start_date   = stack_para['start_date']
    end_date     = stack_para['end_date']
    npts = cc_array.shape[1]

    # remove abnormal data
    ampmax = np.max(cc_array,axis=1)
    tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
    if not len(tindx):
        allstacks1=[];allstacks2=[];nstacks=0
        cc_array=[];cc_ngood=[];cc_time=[]
        return cc_array,cc_ngood,cc_time,allstacks1,allstacks2,nstacks
    else:

        # remove ones with bad amplitude
        cc_array = cc_array[tindx,:]
        cc_time  = cc_time[tindx]
        cc_ngood = cc_ngood[tindx]

        # do substacks
        if rma_substack:
            tstart = obspy.UTCDateTime(start_date)-obspy.UTCDateTime(1970,1,1)
            tend   = obspy.UTCDateTime(end_date)-obspy.UTCDateTime(1970,1,1)
            ttime  = tstart
            nstack = int(np.round((tend-tstart)/(rma_step*3600)))
            ncc_array = np.zeros(shape=(nstack,npts),dtype=np.float32)
            ncc_time  = np.zeros(nstack,dtype=np.float)
            ncc_ngood = np.zeros(nstack,dtype=np.int)

            # loop through each time
            for ii in range(nstack):
                sindx = np.where((cc_time>=ttime) & (cc_time<ttime+rma_substack*3600))[0]

                # when there are data in the time window
                if len(sindx):
                    ncc_array[ii] = np.mean(cc_array[sindx],axis=0)
                    ncc_time[ii]  = ttime
                    ncc_ngood[ii] = np.sum(cc_ngood[sindx],axis=0)
                ttime += rma_step*3600

            # remove bad ones
            tindx = np.where(ncc_ngood>0)[0]
            ncc_array = ncc_array[tindx]
            ncc_time  = ncc_time[tindx]
            ncc_ngood  = ncc_ngood[tindx]

        # do stacking
        allstacks1 = np.zeros(npts,dtype=np.float32)
        allstacks2 = np.zeros(npts,dtype=np.float32)
        allstacks3 = np.zeros(npts,dtype=np.float32)
        allstacks4 = np.zeros(npts,dtype=np.float32)

        if smethod == 'linear':
            allstacks1 = np.mean(cc_array,axis=0)
        elif smethod == 'pws':
            allstacks1 = pws(cc_array,samp_freq)
        elif smethod == 'robust':
            allstacks1,w, = robust_stack(cc_array,0.001)
        elif smethod == 'selective':
            allstacks1 = selective_stack(cc_array,0.001)
        elif smethod == 'all':
            allstacks1 = np.mean(cc_array,axis=0)
            allstacks2 = pws(cc_array,samp_freq)
            allstacks3 = robust_stack(cc_array,0.001)
            allstacks4 = selective_stack(cc_array,0.001)
        nstacks = np.sum(cc_ngood)

    # replace the array for substacks
    if rma_substack:
        cc_array = ncc_array
        cc_time  = ncc_time
        cc_ngood = ncc_ngood

    # good to return
    return cc_array,cc_ngood,cc_time,allstacks1,allstacks2,allstacks3,allstacks4,nstacks

def rotation(bigstack,parameters,locs,flag):
    '''
    this function transfers the Green's tensor from a E-N-Z system into a R-T-Z one

    PARAMETERS:
    -------------------
    bigstack:   9 component Green's tensor in E-N-Z system
    parameters: dict containing all parameters saved in ASDF file
    locs:       dict containing station angle info for correction purpose
    RETURNS:
    -------------------
    tcorr: 9 component Green's tensor in R-T-Z system
    '''
    # load parameter dic
    pi = np.pi
    azi = parameters['azi']
    baz = parameters['baz']
    ncomp,npts = bigstack.shape
    if ncomp<9:
        print('crap did not get enough components')
        tcorr=[]
        return tcorr
    staS  = parameters['station_source']
    staR  = parameters['station_receiver']

    if len(locs):
        sta_list = list(locs['station'])
        angles   = list(locs['angle'])
        # get station info from the name of ASDF file
        ind   = sta_list.index(staS)
        acorr = angles[ind]
        ind   = sta_list.index(staR)
        bcorr = angles[ind]

    #---angles to be corrected----
    if len(locs):
        cosa = np.cos((azi+acorr)*pi/180)
        sina = np.sin((azi+acorr)*pi/180)
        cosb = np.cos((baz+bcorr)*pi/180)
        sinb = np.sin((baz+bcorr)*pi/180)
    else:
        cosa = np.cos(azi*pi/180)
        sina = np.sin(azi*pi/180)
        cosb = np.cos(baz*pi/180)
        sinb = np.sin(baz*pi/180)

    # rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    tcorr = np.zeros(shape=(9,npts),dtype=np.float32)
    tcorr[0] = -cosb*bigstack[7]-sinb*bigstack[6]
    tcorr[1] = sinb*bigstack[7]-cosb*bigstack[6]
    tcorr[2] = bigstack[8]
    tcorr[3] = -cosa*cosb*bigstack[4]-cosa*sinb*bigstack[3]-sina*cosb*bigstack[1]-sina*sinb*bigstack[0]
    tcorr[4] = cosa*sinb*bigstack[4]-cosa*cosb*bigstack[3]+sina*sinb*bigstack[1]-sina*cosb*bigstack[0]
    tcorr[5] = cosa*bigstack[5]+sina*bigstack[2]
    tcorr[6] = sina*cosb*bigstack[4]+sina*sinb*bigstack[3]-cosa*cosb*bigstack[1]-cosa*sinb*bigstack[0]
    tcorr[7] = -sina*sinb*bigstack[4]+sina*cosb*bigstack[3]+cosa*sinb*bigstack[1]-cosa*cosb*bigstack[0]
    tcorr[8] = -sina*bigstack[5]+cosa*bigstack[2]

    return tcorr


####################################################
############## UTILITY FUNCTIONS ###################
####################################################

def check_sample_gaps(stream,date_info):
    """
    this function checks sampling rate and find gaps of all traces in stream.
    PARAMETERS:
    -----------------
    stream: obspy stream object.
    date_info: dict of starting and ending time of the stream

    RETURENS:
    -----------------
    stream: List of good traces in the stream
    """
    # remove empty/big traces
    if len(stream)==0 or len(stream)>100:
        stream = []
        return stream

    # remove traces with big gaps
    if portion_gaps(stream,date_info)>0.3:
        stream = []
        return stream

    freqs = []
    for tr in stream:
        freqs.append(int(tr.stats.sampling_rate))
    freq = max(freqs)
    for tr in stream:
        if int(tr.stats.sampling_rate) != freq:
            stream.remove(tr)
        if tr.stats.npts < 10:
            stream.remove(tr)

    return stream


def portion_gaps(stream,date_info):
    '''
    this function tracks the gaps (npts) from the accumulated difference between starttime and endtime
    of each stream trace. it removes trace with gap length > 30% of trace size.
    PARAMETERS:
    -------------------
    stream: obspy stream object
    date_info: dict of starting and ending time of the stream

    RETURNS:
    -----------------
    pgaps: proportion of gaps/all_pts in stream
    '''
    # ideal duration of data
    starttime = date_info['starttime']
    endtime   = date_info['endtime']
    npts      = (endtime-starttime)*stream[0].stats.sampling_rate

    pgaps=0
    #loop through all trace to accumulate gaps
    for ii in range(len(stream)-1):
        pgaps += (stream[ii+1].stats.starttime-stream[ii].stats.endtime)*stream[ii].stats.sampling_rate
    if npts!=0:pgaps=pgaps/npts
    if npts==0:pgaps=1
    return pgaps


@jit('float32[:](float32[:],float32)')
def segment_interpolate(sig1,nfric):
    '''
    this function interpolates the data to ensure all points located on interger times of the
    sampling rate (e.g., starttime = 00:00:00.015, delta = 0.05.)
    PARAMETERS:
    ----------------------
    sig1:  seismic recordings in a 1D array
    nfric: the amount of time difference between the point and the adjacent assumed samples
    RETURNS:
    ----------------------
    sig2:  interpolated seismic recordings on the sampling points
    '''
    npts = len(sig1)
    sig2 = np.zeros(npts,dtype=np.float32)

    #----instead of shifting, do a interpolation------
    for ii in range(npts):

        #----deal with edges-----
        if ii==0 or ii==npts-1:
            sig2[ii]=sig1[ii]
        else:
            #------interpolate using a hat function------
            sig2[ii]=(1-nfric)*sig1[ii+1]+nfric*sig1[ii]

    return sig2

def resp_spectrum(source,resp_file,downsamp_freq,pre_filt=None):
    '''
    this function removes the instrument response using response spectrum from evalresp.
    the response spectrum is evaluated based on RESP/PZ files before inverted using the obspy
    function of invert_spectrum. a module of create_resp.py is provided in directory of 'additional_modules'
    to create the response spectrum
    PARAMETERS:
    ----------------------
    source: obspy stream object of targeted noise data
    resp_file: numpy data file of response spectrum
    downsamp_freq: sampling rate of the source data
    pre_filt: pre-defined filter parameters
    RETURNS:
    ----------------------
    source: obspy stream object of noise data with instrument response removed
    '''
    #--------resp_file is the inverted spectrum response---------
    respz = np.load(resp_file)
    nrespz= respz[1][:]
    spec_freq = max(respz[0])

    #-------on current trace----------
    nfft = _npts2nfft(source[0].stats.npts)
    sps  = int(source[0].stats.sampling_rate)

    #---------do the interpolation if needed--------
    if spec_freq < 0.5*sps:
        raise ValueError('spectrum file has peak freq smaller than the data, abort!')
    else:
        indx = np.where(respz[0]<=0.5*sps)
        nfreq = np.linspace(0,0.5*sps,nfft//2+1)
        nrespz= np.interp(nfreq,np.real(respz[0][indx]),respz[1][indx])

    #----do interpolation if necessary-----
    source_spect = np.fft.rfft(source[0].data,n=nfft)

    #-----nrespz is inversed (water-leveled) spectrum-----
    source_spect *= nrespz
    source[0].data = np.fft.irfft(source_spect)[0:source[0].stats.npts]

    if pre_filt is not None:
        source[0].data = np.float32(bandpass(source[0].data,pre_filt[0],pre_filt[-1],df=sps,corners=4,zerophase=True))

    return source


def mad(arr):
    """
    Median Absolute Deviation: MAD = median(|Xi- median(X)|)
    PARAMETERS:
    -------------------
    arr: numpy.ndarray, seismic trace data array
    RETURNS:
    data: Median Absolute Deviation of data
    """
    if not np.ma.is_masked(arr):
        med = np.median(arr)
        data = np.median(np.abs(arr - med))
    else:
        med = np.ma.median(arr)
        data = np.ma.median(np.ma.abs(arr-med))
    return data


def detrend(data):
    '''
    this function removes the signal trend based on QR decomposion
    NOTE: QR is a lot faster than the least square inversion used by
    scipy (also in obspy).
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with trend removed
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        X = np.ones((npts,2))
        X[:,0] = np.arange(0,npts)/npts
        Q,R = np.linalg.qr(X)
        rq  = np.dot(np.linalg.inv(R),Q.transpose())
        coeff = np.dot(rq,data)
        data = data-np.dot(X,coeff)
    elif data.ndim == 2:
        npts = data.shape[1]
        X = np.ones((npts,2))
        X[:,0] = np.arange(0,npts)/npts
        Q,R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R),Q.transpose())
        for ii in range(data.shape[0]):
            coeff = np.dot(rq,data[ii])
            data[ii] = data[ii] - np.dot(X,coeff)
    return data

def demean(data):
    '''
    this function remove the mean of the signal
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with mean removed
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        data = data-np.mean(data)
    elif data.ndim == 2:
        for ii in range(data.shape[0]):
            data[ii] = data[ii]-np.mean(data[ii])
    return data

def taper(data):
    '''
    this function applies a cosine taper using obspy functions
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with taper applied
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
            taper_sides = func(2*wlen+1)
        # taper window
        win  = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),taper_sides[len(taper_sides) - wlen:]))
        data *= win
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
            data[ii] *= win
    return data


@jit(nopython = True)
def moving_ave(A,N):
    '''
    this Numba compiled function does running smooth average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the half window length to smooth

    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    '''
    A = np.concatenate((A[:N],A,A[-N:]),axis=0)
    B = np.zeros(A.shape,A.dtype)

    tmp=0.
    for pos in range(N,A.size-N):
        # do summing only once
        if pos==N:
            for i in range(-N,N+1):
                tmp+=A[pos+i]
        else:
            tmp=tmp-A[pos-N-1]+A[pos+N]
        B[pos]=tmp/(2*N+1)
        if B[pos]==0:
            B[pos]=1
    return B[N:-N]


def robust_stack(cc_array,epsilon):
    """
    this is a robust stacking algorithm described in Palvis and Vernon 2010

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle
    """
    res  = 9E9  # residuals
    w = np.ones(cc_array.shape[0])
    nstep=0
    newstack = np.median(cc_array,axis=0)
    while res > epsilon:
        stack = newstack
        for i in range(cc_array.shape[0]):
            crap = np.multiply(stack,cc_array[i,:].T)
            crap_dot = np.sum(crap)
            di_norm = np.linalg.norm(cc_array[i,:])
            ri = cc_array[i,:] -  crap_dot*stack
            ri_norm = np.linalg.norm(ri)
            w[i]  = np.abs(crap_dot) /di_norm/ri_norm#/len(cc_array[:,1])
        # print(w)
        w =w /np.sum(w)
        newstack =np.sum( (w*cc_array.T).T,axis=0)#/len(cc_array[:,1])
        res = np.linalg.norm(newstack-stack,ord=1)/np.linalg.norm(newstack)/len(cc_array[:,1])
        nstep +=1
        if nstep>10:
            return newstack, w, nstep
    return newstack, w, nstep



def selective_stack(cc_array,epsilon):
    """
    this is a selective stacking algorithm developed by Jared Bryan.

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle
    """
    res  = 9E9  # residuals
    cc = np.ones(cc_array.shape[0])
    nstep=0
    newstack = np.mean(cc_array,axis=0)
    for i in range(cc_array.shape[0]):
    	CC[i] = np.sum(np.multiply(stack,cc_array[i,:].T))
    ik = np.where(CC>=epsilon)
    newstack = np.mean(cc_array[ik,:],axis=0)

    return newstack, cc



def whiten(data, fft_para):
    '''
    This function takes 1-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: dict containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    '''

    # load parameters
    delta   = fft_para['dt']
    freqmin = fft_para['freqmin']
    freqmax = fft_para['freqmax']
    smooth_N  = fft_para['smooth_N']
    freq_norm = fft_para['freq_norm']

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    Nfft = int(next_fast_len(int(data.shape[axis])))

    Napod = 100
    Nfft = int(Nfft)
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1

    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft/2:
        high = int(Nfft//2)

    FFTRawSign = scipy.fftpack.fft(data, Nfft,axis=axis)
    # Left tapering:
    if axis == 1:
        FFTRawSign[:,0:low] *= 0
        FFTRawSign[:,low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[:,left:right] = np.exp(1j * np.angle(FFTRawSign[:,left:right]))
        elif freq_norm == 'rma':
            for ii in range(data.shape[0]):
                tave = moving_ave(np.abs(FFTRawSign[ii,left:right]),smooth_N)
                FFTRawSign[ii,left:right] = FFTRawSign[ii,left:right]/tave
        # Right tapering:
        FFTRawSign[:,right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,right:high]))
        FFTRawSign[:,high:Nfft//2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[:,-(Nfft//2)+1:] = np.flip(np.conj(FFTRawSign[:,1:(Nfft//2)]),axis=axis)
    else:
        FFTRawSign[0:low] *= 0
        FFTRawSign[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        elif freq_norm == 'rma':
            tave = moving_ave(np.abs(FFTRawSign[left:right]),smooth_N)
            FFTRawSign[left:right] = FFTRawSign[left:right]/tave
        # Right tapering:
        FFTRawSign[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[right:high]))
        FFTRawSign[high:Nfft//2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[-(Nfft//2)+1:] = FFTRawSign[1:(Nfft//2)].conjugate()[::-1]

    return FFTRawSign

def adaptive_filter(arr,g):
    '''
    the adaptive covariance filter to enhance coherent signals. Fellows the method of
    Nakata et al., 2015 (Appendix B)

    the filtered signal [x1] is given by x1 = ifft(P*x1(w)) where x1 is the ffted spectra
    and P is the filter. P is constructed by using the temporal covariance matrix.

    PARAMETERS:
    ----------------------
    arr: numpy.ndarray contains the 2D traces of daily/hourly cross-correlation functions
    g: a positive number to adjust the filter harshness
    RETURNS:
    ----------------------
    narr: numpy vector contains the stacked cross correlation function
    '''
    if arr.ndim == 1:
        return arr
    N,M = arr.shape
    Nfft = next_fast_len(M)

    # fft the 2D array
    spec = scipy.fftpack.fft(arr,axis=1,n=Nfft)[:,:M]

    # make cross-spectrm matrix
    cspec = np.zeros(shape=(N*N,M),dtype=np.complex64)
    for ii in range(N):
        for jj in range(N):
            kk = ii*N+jj
            cspec[kk] = spec[ii]*np.conjugate(spec[jj])

    S1 = np.zeros(M,dtype=np.complex64)
    S2 = np.zeros(M,dtype=np.complex64)
    # construct the filter P
    for ii in range(N):
        mm = ii*N+ii
        S2 += cspec[mm]
        for jj in range(N):
            kk = ii*N+jj
            S1 += cspec[kk]

    p = np.power((S1-S2)/(S2*(N-1)),g)

    # make ifft
    narr = np.real(scipy.fftpack.ifft(np.multiply(p,spec),Nfft,axis=1)[:,:M])
    return np.mean(narr,axis=0)

def pws(arr,sampling_rate,power=2,pws_timegate=5.):
    '''
    Performs phase-weighted stack on array of time series. Modified on the noise function by Tim Climents.
    Follows methods of Schimmel and Paulssen, 1997.
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    PARAMETERS:
    ---------------------
    arr: N length array of time series data (numpy.ndarray)
    sampling_rate: sampling rate of time series arr (int)
    power: exponent for phase stack (int)
    pws_timegate: number of seconds to smooth phase stack (float)

    RETURNS:
    ---------------------
    weighted: Phase weighted stack of time series data (numpy.ndarray)
    '''

    if arr.ndim == 1:
        return arr
    N,M = arr.shape
    analytic = hilbert(arr,axis=1, N=next_fast_len(M))[:,:M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j*phase),axis=0)
    phase_stack = np.abs(phase_stack)**(power)

    # smoothing
    #timegate_samples = int(pws_timegate * sampling_rate)
    #phase_stack = moving_ave(phase_stack,timegate_samples)
    weighted = np.multiply(arr,phase_stack)
    return np.mean(weighted,axis=0)


def nroot_stack(cc_array,power):
    '''
    this is nth-root stacking algorithm translated based on the matlab function
    from https://github.com/xtyangpsp/SeisStack (by Xiaotao Yang; follows the 
    reference of Millet, F et al., 2019 JGR) 

    Parameters:
    ------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    power: np.int, nth root for the stacking

    Returns:
    ------------
    nstack: np.ndarray, final stacked waveforms

    Written by Chengxin Jiang @ANU (May2020)
    '''
    if cc_array.ndim == 1:
        print('2D matrix is needed for nroot_stack')
        return cc_array
    N,M = cc_array.shape 
    dout = np.zeros(M,dtype=np.float32)

    # construct y
    for ii in range(N):
        dat = cc_array[ii,:]
        dout += np.sign(dat)*np.abs(dat)**(1/power)
    dout /= N

    # the final stacked waveform
    nstack = dout*np.abs(dout)**(power-1)

    return nstack


def selective_stack(cc_array,epsilon,cc_th):
    ''' 
    this is a selective stacking algorithm developed by Jared Bryan/Kurama Okubo.

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    cc_th: numpy.float, threshold of correlation coefficient to be selected

    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation
    nstep: np.int, total number of iterations for the stacking

    Originally ritten by Marine Denolle 
    Modified by Chengxin Jiang @Harvard (Oct2020)
    '''
    if cc_array.ndim == 1:
        print('2D matrix is needed for nroot_stack')
        return cc_array
    N,M = cc_array.shape 

    res  = 9E9  # residuals
    cof  = np.zeros(N,dtype=np.float32)
    newstack = np.mean(cc_array,axis=0)

    nstep = 0
    # start iteration
    while res>epsilon:
        for ii in range(N):
            cof[ii] = np.corrcoef(newstack, cc_array[ii,:])[0, 1]
        
        # find good waveforms
        indx = np.where(cof>=cc_th)[0]
        if not len(indx): raise ValueError('cannot find good waveforms inside selective stacking')
        oldstack = newstack
        newstack = np.mean(cc_array[indx],axis=0)
        res = np.linalg.norm(newstack-oldstack)/(np.linalg.norm(newstack)*M)
        nstep +=1

    return newstack, nstep


def get_cc(s1,s_ref):
    # returns the correlation coefficient between waveforms in s1 against reference
    # waveform s_ref.
    #
    cc=np.zeros(s1.shape[0])
    s_ref_norm = np.linalg.norm(s_ref)
    for i in range(s1.shape[0]):
        cc[i]=np.sum(np.multiply(s1[i,:],s_ref))/np.linalg.norm(s1[i,:])/s_ref_norm
    return cc


########################################################
################ MONITORING FUNCTIONS ##################
########################################################

'''
a compilation of all available core functions for computing phase delays based on ambient noise interferometry

quick index of dv/v methods:
1) stretching (time stretching; Weaver et al (2011))
2) dtw_dvv (Dynamic Time Warping; Mikesell et al. 2015)
3) mwcs_dvv (Moving Window Cross Spectrum; Clark et al., 2011)
4) mwcc_dvv (Moving Window Cross Correlation; Snieder et al., 2012)
5) wts_dvv (Wavelet Streching; Yuan et al., in prep)
6) wxs_dvv (Wavelet Xross Spectrum; Mao et al., 2019)
7) wdw_dvv (Wavelet Dynamic Warping; Yuan et al., in prep)
'''

def stretching(ref, cur, dv_range, nbtrial, para):

    """
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.

    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    dv_range: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change ('float')
    nbtrial: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
    para: vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
    For error computation, we need parameters:
        fmin: minimum frequency of the data
        fmax: maximum frequency of the data
        tmin: minimum time window where the dv/v is computed
        tmax: maximum time window where the dv/v is computed
    RETURNS:
    ----------------
    dv: Relative velocity change dv/v (in %)
    cc: correlation coefficient between the reference waveform and the best stretched/compressed current waveform
    cdp: correlation coefficient between the reference waveform and the initial current waveform
    error: Errors in the dv/v measurements based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)

    Note: The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values.
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .

    Originally by L. Viens 04/26/2018 (Viens et al., 2018 JGR)
    modified by Chengxin Jiang
    """
    # load common variables from dictionary
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin,tmax,dt)

    # make useful one for measurements
    dvmin = -np.abs(dv_range)
    dvmax = np.abs(dv_range)
    Eps = 1+(np.linspace(dvmin, dvmax, nbtrial))
    cof = np.zeros(Eps.shape,dtype=np.float32)

    # Set of stretched/compressed current waveforms
    for ii in range(len(Eps)):
        nt = tvec*Eps[ii]
        s = np.interp(x=tvec, xp=nt, fp=cur)
        waveform_ref = ref
        waveform_cur = s
        cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cdp = np.corrcoef(cur, ref)[0, 1] # correlation coefficient between the reference and initial current waveforms

    # find the maximum correlation coefficient
    imax = np.nanargmax(cof)
    if imax >= len(Eps)-2:
        imax = imax - 2
    if imax <= 2:
        imax = imax + 2

    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = np.linspace(Eps[imax-2], Eps[imax+2], 100)
    ncof    = np.zeros(dtfiner.shape,dtype=np.float32)
    for ii in range(len(dtfiner)):
        nt = tvec*dtfiner[ii]
        s = np.interp(x=tvec, xp=nt, fp=cur)
        waveform_ref = ref
        waveform_cur = s
        ncof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cc = np.max(ncof) # Find maximum correlation coefficient of the refined  analysis
    dv = 100. * dtfiner[np.argmax(ncof)]-100 # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

    # Error computation based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, error, cc, cdp


def stretching_vect(ref, cur, dv_range, nbtrial, para):
    
    """
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.
    
    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    dv_range: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change ('float')
    nbtrial: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
    para: vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
    For error computation, we need parameters:
        fmin: minimum frequency of the data
        fmax: maximum frequency of the data
        tmin: minimum time window where the dv/v is computed 
        tmax: maximum time window where the dv/v is computed 
    RETURNS:
    ----------------
    dv: Relative velocity change dv/v (in %)
    cc: correlation coefficient between the reference waveform and the best stretched/compressed current waveform
    cdp: correlation coefficient between the reference waveform and the initial current waveform
    error: Errors in the dv/v measurements based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)

    Note: The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values. 
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .

    Originally by L. Viens 04/26/2018 (Viens et al., 2018 JGR)
    modified by Chengxin Jiang
    modified by Laura Ermert: vectorized version
    """ 
    # load common variables from dictionary
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin, tmax, dt)

    # make useful one for measurements
    dvmin = -np.abs(dv_range)
    dvmax = np.abs(dv_range)
    Eps = 1 + (np.linspace(dvmin, dvmax, nbtrial))
    cdp = np.corrcoef(cur, ref)[0, 1] # correlation coefficient between the reference and initial current waveforms
    waveforms = np.zeros((nbtrial + 1, len(ref)))
    waveforms[0, :] = ref

    # Set of stretched/compressed current waveforms
    for ii in range(nbtrial):
        nt = tvec * Eps[ii]
        s = np.interp(x=tvec, xp=nt, fp=cur)
        waveforms[ii + 1, :] = s
    cof = np.corrcoef(waveforms)[0][1:]
    
    # find the maximum correlation coefficient
    imax = np.nanargmax(cof)
    if imax >= len(Eps)-2:
        imax = imax - 2
    if imax < 2:
        imax = imax + 2

    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = np.linspace(Eps[imax-2], Eps[imax+2], nbtrial)
    #ncof    = np.zeros(dtfiner.shape,dtype=np.float32)
    waveforms = np.zeros((nbtrial + 1, len(ref)))
    waveforms[0, :] = ref
    for ii in range(len(dtfiner)):
        nt = tvec * dtfiner[ii]
        s = np.interp(x=tvec, xp=nt, fp=cur)
        waveforms[ii + 1, :] = s
    ncof = np.corrcoef(waveforms)[0][1: ]
    cc = np.max(ncof) # Find maximum correlation coefficient of the refined  analysis
    dv = 100. * dtfiner[np.argmax(ncof)] - 100 # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

    # Error computation based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, error, cc, cdp

def dtw_dvv(ref, cur, para, maxLag, b, direction):
    """
    Dynamic time warping for dv/v estimation.

    PARAMETERS:
    ----------------
    ref : reference signal (np.array, size N)
    cur : current signal (np.array, size N)
    para: dict containing useful parameters about the data window and targeted frequency
    maxLag : max number of points to search forward and backward.
            Suggest setting it larger if window is set larger.
    b : b-value to limit strain, which is to limit the maximum velocity perturbation.
            See equation 11 in (Mikesell et al. 2015)
    direction: direction to accumulate errors (1=forward, -1=backward)
    RETURNS:
    ------------------
    -m0 : estimated dv/v
    em0 : error of dv/v estimation

    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)
    """
    twin = para['twin']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    tvect = np.arange(tmin,tmax,dt)

    # setup other parameters
    npts = len(ref) # number of time samples

    # compute error function over lags, which is independent of strain limit 'b'.
    err = computeErrorFunction( cur, ref, npts, maxLag )

    # direction to accumulate errors (1=forward, -1=backward)
    dist  = accumulateErrorFunction( direction, err, npts, maxLag, b )
    stbar = backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )
    stbarTime = stbar * dt   # convert from samples to time

    # cut the first and last 5% for better regression
    indx = np.where((tvect>=0.05*npts*dt) & (tvect<=0.95*npts*dt))[0]

    # linear regression to get dv/v
    if npts >2:

        # weights
        w = np.ones(npts)
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(tvect.flatten()[indx], stbarTime.flatten()[indx], w.flatten()[indx], intercept_origin=True)

    else:
        print('not enough points to estimate dv/v for dtw')
        m0=0;em0=0

    return m0*100,em0*100,dist


def mwcs_dvv(ref, cur, moving_window_length, slide_step, para, smoothing_half_win=5):
    """
    Moving Window Cross Spectrum method to measure dv/v (relying on phi=2*pi*f*t in freq domain)

    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    moving_window_length: moving window length to calculate cross-spectrum (np.float, in sec)
    slide_step: steps in time to shift the moving window (np.float, in seconds)
    para: a dict containing parameters about input data window and frequency info, including
        delta->The sampling rate of the input timeseries (in Hz)
        window-> The target window for measuring dt/t
        freq-> The frequency bound to compute the dephasing (in Hz)
        tmin: The leftmost time lag (used to compute the "time lags array")
    smoothing_half_win: If different from 0, defines the half length of the smoothing hanning window.

    RETURNS:
    ------------------
    time_axis: the central times of the windows.
    delta_t: dt
    delta_err:error
    delta_mcoh: mean coherence

    Copied from MSNoise (https://github.com/ROBelgium/MSNoise/tree/master/msnoise)
    Modified by Chengxin Jiang
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvect = np.arange(tmin,tmax,dt)

    # parameter initialize
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    # info on the moving window
    window_length_samples = np.int(moving_window_length/dt)
    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    count = 0
    tp = cosine_taper(window_length_samples, 0.15)

    minind = 0
    maxind = window_length_samples

    # loop through all sub-windows
    while maxind <= len(ref):
        cci = cur[minind:maxind]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[minind:maxind]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step/dt)
        maxind += int(slide_step/dt)

        # do fft
        fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
        fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # get cross-spectrum & do filtering
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
            dcur = np.sqrt(smooth(fcur2, window='hanning',half_win=smoothing_half_win))
            dref = np.sqrt(smooth(fref2, window='hanning',half_win=smoothing_half_win))
            X = smooth(X, window='hanning',half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, dt)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= fmin,freq_vec <= fmax))

        # Get Coherence and its mean value
        coh = getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[index_range])

        # Get Weights
        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        # Frequency array:
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin; weights for the WLS must be the variance !
        m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())
        delta_t.append(m)

        # print phi.shape, v.shape, w.shape
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2)

        delta_err.append(e)
        delta_mcoh.append(np.real(mcoh))
        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(cur) + int(slide_step/dt):
        print("The last window was too small, but was computed")

    # ensure all matrix are np array
    delta_t = np.array(delta_t)
    delta_err = np.array(delta_err)
    delta_mcoh = np.array(delta_mcoh)
    time_axis  = np.array(time_axis)

    # ready for linear regression
    delta_mincho = 0.65
    delta_maxerr = 0.1
    delta_maxdt  = 0.1
    indx1 = np.where(delta_mcoh>delta_mincho)
    indx2 = np.where(delta_err<delta_maxerr)
    indx3 = np.where(delta_t<delta_maxdt)

    #-----find good dt measurements-----
    indx = np.intersect1d(indx1,indx2)
    indx = np.intersect1d(indx,indx3)

    if len(indx) >2:

        #----estimate weight for regression----
        w = 1/delta_err[indx]
        w[~np.isfinite(w)] = 1.0

        #---------do linear regression-----------
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=True)

    else:
        print('not enough points to estimate dv/v for mwcs')
        m0=0;em0=0

    return -m0*100,em0*100


def WCC_dvv(ref, cur, moving_window_length, slide_step, para):
    """
    Windowed cross correlation (WCC) for dt or dv/v mesurement (Snieder et al. 2012)

    Parameters:
    -----------
    ref: The "Reference" timeseries
    cur: The "Current" timeseries
    moving_window_length: The moving window length (in seconds)
    slide_step: The step to jump for the moving window (in seconds)
    para: a dict containing freq/time info of the data matrix

    Returns:
    ------------
    time_axis: central times of the moving window
    delta_t: dt
    delta_err: error
    delta_mcoh: mean coherence for each window

    Written by Congcong Yuan (1 July, 2019)
    """
    # common variables
    twin = para['twin']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)

    # parameter initialize
    delta_t = []
    delta_t_coef = []
    time_axis = []

    # info on the moving window
    window_length_samples = np.int(moving_window_length/dt)
    count = 0
    tp = cosine_taper(window_length_samples, 0.15)

    minind = 0
    maxind = window_length_samples

    # loop through all sub-windows
    while maxind <= len(ref):
        cci = cur[minind:maxind]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[minind:maxind]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step/dt)
        maxind += int(slide_step/dt)

        # normalize signals before cross correlation
        cci = (cci - cci.mean()) / cci.std()
        cri = (cri - cri.mean()) / cri.std()

        # get maximum correlation coefficient and its index
        cc2 = np.correlate(cci, cri, mode='same')
        cc2 = cc2 / np.sqrt((cci**2).sum() * (cri**2).sum())

        imaxcc2 = np.where(cc2==np.max(cc2))[0]
        maxcc2 = np.max(cc2)

        # get the time shift
        m = (imaxcc2-((maxind-minind)//2))*dt
        delta_t.append(m)
        delta_t_coef.append(maxcc2)

        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

    del cci, cri, cc2, imaxcc2, maxcc2
    del m

    if maxind > len(cur) + int(slide_step/dt):
        print("The last window was too small, but was computed")

    delta_t = np.array(delta_t)
    delta_t_coef = np.array(delta_t_coef)
    time_axis  = np.array(time_axis)

    # linear regression to get dv/v
    if count >2:
        # simple weight
        w = np.ones(count)
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis.flatten(), delta_t.flatten(), w.flatten(),intercept_origin=True)

    else:
        print('not enough points to estimate dv/v for wcc')
        m0=0;em0=0

    return -m0*100,em0*100


def wxs_dvv(ref,cur,allfreq,para,dj=1/12, s0=-1, J=-1, sig=False, wvn='morlet',unwrapflag=False):
    """
    Compute dt or dv/v in time and frequency domain from wavelet cross spectrum (wxs).
    for all frequecies in an interest range

    Parameters
    --------------
    ref: The "Reference" timeseries (numpy.ndarray)
    cur: The "Current" timeseries (numpy.ndarray)
    allfreq: a boolen variable to make measurements on all frequency range or not
    para: a dict containing freq/time info of the data matrix
    dj, s0, J, sig, wvn: common parameters used in 'wavelet.wct'
    unwrapflag: True - unwrap phase delays. Default is False

    RETURNS:
    ------------------
    dvv*100 : estimated dv/v in %
    err*100 : error of dv/v estimation in %

    Originally written by Tim Clements (1 March, 2019)
    Modified by Congcong Yuan (30 June, 2019) based on (Mao et al. 2019).
    Updated by Chengxin Jiang (10 Oct, 2019) to merge the functionality for mesurements across all frequency and one freq range
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin,tmax,dt)
    npts = len(tvec)

    # perform cross coherent analysis, modified from function 'wavelet.cwt'
    WCT, aWCT, coi, freq, sig = wct_modified(ref, cur, dt, dj=dj, s0=s0, J=J, sig=sig, wavelet=wvn, normalize=True)

    if unwrapflag:
        phase = np.unwrap(aWCT,axis=-1) # axis=0, upwrap along time; axis=-1, unwrap along frequency
    else:
        phase = aWCT

    # zero out data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]

    # follow MWCS to do two steps of linear regression
    if not allfreq:

        delta_t_m, delta_t_unc = np.zeros(npts,dtype=np.float32),np.zeros(npts,dtype=np.float32)
        # assume the tvec is the time window to measure dt
        for it in range(npts):
            w = 1/WCT[freq_indin,it]
            w[~np.isfinite(w)] = 1.
            delta_t_m[it],delta_t_unc[it] = linear_regression(freq[freq_indin]*2*np.pi, phase[freq_indin,it], w)

        # new weights for regression
        w2 = 1/np.mean(WCT[freq_indin,:],axis=0)
        w2[~np.isfinite(w2)] = 1.

        # now use dt and t to get dv/v
        if len(w2)>2:
            if not np.any(delta_t_m):
                dvv, err = np.nan,np.nan
            m, em = linear_regression(tvec, delta_t_m, w2, intercept_origin=True)
            dvv, err = -m, em
        else:
            print('not enough points to estimate dv/v for wts')
            dvv, err=np.nan, np.nan

        return dvv*100,err*100

    # convert phase directly to delta_t for all frequencies
    else:

        # convert phase delay to time delay
        delta_t = phase / (2*np.pi*freq[:,None]) # normalize phase by (2*pi*frequency)
        dvv, err = np.zeros(freq_indin.shape), np.zeros(freq_indin.shape)

        # loop through freq for linear regression
        for ii, ifreq in enumerate(freq_indin):
            if len(tvec)>2:
                if not np.any(delta_t[ifreq]):
                    continue

                # how to better approach the uncertainty of delta_t
                w = 1/WCT[ifreq]
                w[~np.isfinite(w)] = 1.0

                #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
                m, em = linear_regression(tvec, delta_t[ifreq], w, intercept_origin=True)
                dvv[ii], err[ii] = -m, em
            else:
                print('not enough points to estimate dv/v for wts')
                dvv[ii], err[ii]=np.nan, np.nan

        return freq[freq_indin], dvv*100, err*100


def wts_dvv(ref,cur,allfreq,para,dv_range,nbtrial,dj=1/12,s0=-1,J=-1,wvn='morlet',normalize=True):
    """
    Apply stretching method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range

    Parameters
    --------------
    ref: The "Reference" timeseries (numpy.ndarray)
    cur: The "Current" timeseries (numpy.ndarray)
    allfreq: a boolen variable to make measurements on all frequency range or not
    para: a dict containing freq/time info of the data matrix
    dv_range: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change (float)
    nbtrial: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  (float)
    dj, s0, J, sig, wvn: common parameters used in 'wavelet.wct'
    normalize: normalize the wavelet spectrum or not. Default is True

    RETURNS:
    ------------------
    dvv: estimated dv/v
    err: error of dv/v estimation

    Written by Congcong Yuan (30 Jun, 2019)
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin,tmax,dt)

    # apply cwt on two traces
    cwt1, sj, freq, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)

    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)

    # zero out data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]

    # convert wavelet domain back to time domain (~filtering)
    if not allfreq:

        # inverse cwt to time domain
        icwt1 = pycwt.icwt(cwt1[freq_indin], sj[freq_indin], dt, dj, wvn)
        icwt2 = pycwt.icwt(cwt2[freq_indin], sj[freq_indin], dt, dj, wvn)

        # assume all time window is used
        wcwt1, wcwt2 = np.real(icwt1), np.real(icwt2)

        # Normalizes both signals, if appropriate.
        if normalize:
            ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
            ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
        else:
            ncwt1 = wcwt1
            ncwt2 = wcwt2

        # run stretching
        dvv, err, cc, cdp = stretching(ncwt2, ncwt1, dv_range, nbtrial, para)
        return dvv, err

    # directly take advantage of the
    else:
        # initialize variable
        nfreq=len(freq_indin)
        dvv, cc, cdp, err = np.zeros(nfreq,dtype=np.float32), np.zeros(nfreq,dtype=np.float32),\
            np.zeros(nfreq,dtype=np.float32),np.zeros(nfreq,dtype=np.float32)

        # loop through each freq
        for ii, ifreq in enumerate(freq_indin):

            # prepare windowed data
            wcwt1, wcwt2 = rcwt1[ifreq], rcwt2[ifreq]

            # Normalizes both signals, if appropriate.
            if normalize:
                ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
                ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
            else:
                ncwt1 = wcwt1
                ncwt2 = wcwt2

            # run stretching
            dv, error, c1, c2 = stretching(ncwt2, ncwt1, dv_range, nbtrial, para)
            dvv[ii], cc[ii], cdp[ii], err[ii]=dv, c1, c2, error

        return freq[freq_indin], dvv, err


def wtdtw_allfreq(ref,cur,allfreq,para,maxLag,b,direction,dj=1/12,s0=-1,J=-1,wvn='morlet',normalize=True):
    """
    Apply dynamic time warping method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range

    Parameters
    --------------
    ref: The "Reference" timeseries (numpy.ndarray)
    cur: The "Current" timeseries (numpy.ndarray)
    allfreq: a boolen variable to make measurements on all frequency range or not
    maxLag: max number of points to search forward and backward.
    b: b-value to limit strain, which is to limit the maximum velocity perturbation. See equation 11 in (Mikesell et al. 2015)
    direction: direction to accumulate errors (1=forward, -1=backward)
    dj, s0, J, sig, wvn: common parameters used in 'wavelet.wct'
    normalize: normalize the wavelet spectrum or not. Default is True

    RETURNS:
    ------------------
    dvv: estimated dv/v
    err: error of dv/v estimation

    Written by Congcong Yuan (30 Jun, 2019)
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin,tmax)*dt

    # apply cwt on two traces
    cwt1, sj, freq, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)

    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)

    # zero out cone of influence and data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: input frequency out of limits!')
    else:
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]

        # Use DTW method to extract dvv
        nfreq=len(freq_indin)
        dvv, err = np.zeros(nfreq,dtype=np.float32), np.zeros(nfreq,dtype=np.float32)

        for ii,ifreq in enumerate(freq_indin):

            # prepare windowed data
            wcwt1, wcwt2 = rcwt1[ifreq], rcwt2[ifreq]
            # Normalizes both signals, if appropriate.
            if normalize:
                ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
                ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
            else:
                ncwt1 = wcwt1
                ncwt2 = wcwt2

            # run dtw
            dv, error, dist  = dtw_dvv(ncwt2, ncwt1, para, maxLag, b, direction)
            dvv[ii], err[ii] = dv, error

    del cwt1, cwt2, rcwt1, rcwt2, ncwt1, ncwt2, wcwt1, wcwt2, coi, sj, dist

    if not allfreq:
        return np.mean(dvv),np.mean(err)
    else:
        return freq[freq_indin], dvv, err


#############################################################
################ MONITORING UTILITY FUNCTIONS ###############
#############################################################

'''
below are assembly of the monitoring utility functions called by monitoring functions
'''

def smooth(x, window='boxcar', half_win=3):
    """
    performs smoothing in interested time window

    Parameters
    --------------
    x: timeseris data
    window: types of window to do smoothing
    half_win: half window length

    RETURNS:
    ------------------
    y: smoothed time window
    """
    # TODO: docsting
    window_len = 2 * half_win + 1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype('complex')
    else:
        w = scipy.signal.hanning(window_len).astype('complex')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[half_win:len(y) - half_win]


def nextpow2(x):
    """
    Returns the next power of 2 of x.
    """
    return int(np.ceil(np.log2(np.abs(x))))


def getCoherence(dcs, ds1, ds2):
    """
    get cross coherence between reference and current waveforms following equation of A3 in Clark et al., 2011

    Parameters
    --------------
    dcs: amplitude of the cross spectrum
    ds1: amplitude of the spectrum of current waveform
    ds2: amplitude of the spectrum of reference waveform

    RETURNS:
    ------------------
    coh: cohrerency matrix used for estimate the robustness of the cross spectrum
    """
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2) > 0))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh


def computeErrorFunction(u1, u0, nSample, lag, norm='L2'):
    """
    compute Error Function used in DTW. The error function is equation 1 in Hale, 2013. You could uncomment the
    L1 norm and comment the L2 norm if you want on Line 29

    Parameters
    --------------
    u1:  trace that we want to warp; size = (nsamp,1)
    u0:  reference trace to compare with: size = (nsamp,1)
    nSample: numer of points to compare in the traces
    lag: maximum lag in sample number to search
    norm: 'L2' or 'L1' (default is 'L2')

    RETURNS:
    ------------------
    err: the 2D error function; size = (nsamp,2*lag+1)

    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)

    """

    if lag >= nSample:
        raise ValueError('computeErrorFunction:lagProblem','lag must be smaller than nSample')

    # Allocate error function variable
    err = np.zeros([nSample, 2 * lag + 1])

    # initial error calculation
    # loop over lags
    for ll in np.arange(-lag,lag + 1):
        thisLag = ll + lag

        # loop over samples
        for ii in range(nSample):

            # skip corners for now, we will come back to these
            if (ii + ll >= 0) & (ii + ll < nSample):
                err[ii,thisLag] = u1[ii] - u0[ii + ll]

    if norm == 'L2':
        err = err**2
    elif norm == 'L1':
        err = np.abs(err)

    # Now fix corners with constant extrapolation
    for ll in np.arange(-lag,lag + 1):
        thisLag = ll + lag

        for ii in range(nSample):
            if ii + ll < 0:
                err[ii, thisLag] = err[-ll, thisLag]

            elif ii + ll > nSample - 1:
                err[ii,thisLag] = err[nSample - ll - 1,thisLag]

    return err


def accumulateErrorFunction(dir, err, nSample, lag, b ):
    """
    accumulation of the error, which follows the equation 6 in Hale, 2013.

    Parameters
    --------------
    dir: accumulation direction ( dir > 0 = forward in time, dir <= 0 = backward in time)
    err: the 2D error function; size = (nsamp,2*lag+1)
    nSample: numer of points to compare in the traces
    lag: maximum lag in sample number to search
    b: strain limit (integer value >= 1)

    RETURNS:
    ------------------
    d: the 2D distance function; size = (nsamp,2*lag+1)

    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)

    """

    # number of lags from [ -lag : +lag ]
    nLag = ( 2 * lag ) + 1

    # allocate distance matrix
    d = np.zeros([nSample, nLag])

    # Setup indices based on forward or backward accumulation direction
    if dir > 0: # FORWARD
        iBegin, iEnd, iInc = 0, nSample - 1, 1
    else: # BACKWARD
        iBegin, iEnd, iInc = nSample - 1, 0, -1

    # Loop through all times ii in forward or backward direction
    for ii in range(iBegin,iEnd + iInc,iInc):

        # min/max to account for the edges/boundaries
        ji = max([0, min([nSample - 1, ii - iInc])])
        jb = max([0, min([nSample - 1, ii - iInc * b])])

        # loop through all lag
        for ll in range(nLag):

            # check limits on lag indices
            lMinus1 = ll - 1

            # check lag index is greater than 0
            if lMinus1 < 0:
                lMinus1 = 0 # make lag = first lag

            lPlus1 = ll + 1 # lag at l+1

            # check lag index less than max lag
            if lPlus1 > nLag - 1:
                lPlus1 = nLag - 1

            # get distance at lags (ll-1, ll, ll+1)
            distLminus1 = d[jb, lMinus1] # minus:  d[i-b, j-1]
            distL = d[ji,ll] # actual d[i-1, j]
            distLplus1 = d[jb, lPlus1] # plus d[i-b, j+1]

            if ji != jb: # equation 10 in Hale, 2013
                for kb in range(ji,jb + iInc - 1, -iInc):
                    distLminus1 = distLminus1 + err[kb, lMinus1]
                    distLplus1 = distLplus1 + err[kb, lPlus1]

            # equation 6 (if b=1) or 10 (if b>1) in Hale (2013) after treating boundaries
            d[ii, ll] = err[ii,ll] + min([distLminus1, distL, distLplus1])

    return d


def backtrackDistanceFunction(dir, d, err, lmin, b):
    """
    The function is equation 2 in Hale, 2013.

    Parameters
    --------------
    dir: side to start minimization ( dir > 0 = front, dir <= 0 =  back)
    d : the 2D distance function; size = (nsamp,2*lag+1)
    err: the 2D error function; size = (nsamp,2*lag+1)
    lmin: minimum lag to search over
    b : strain limit (integer value >= 1)

    RETURNS:
    ------------------
    stbar: vector of integer shifts subject to |u(i)-u(i-1)| <= 1/b

    Original by Di Yang
    Last modified by Dylan Mikesell (19 Dec. 2014)

    Translated to python by Tim Clements (17 Aug. 2018)

    """

    nSample, nLag = d.shape
    stbar = np.zeros(nSample)

    # Setup indices based on forward or backward accumulation direction
    if dir > 0: # FORWARD
        iBegin, iEnd, iInc = 0, nSample - 1, 1
    else: # BACKWARD
        iBegin, iEnd, iInc = nSample - 1, 0, -1

    # start from the end (front or back)
    ll = np.argmin(d[iBegin,:]) # find minimum accumulated distance at front or back depending on 'dir'
    stbar[iBegin] = ll + lmin # absolute value of integer shift

    # move through all time samples in forward or backward direction
    ii = iBegin

    while ii != iEnd:

        # min/max for edges/boundaries
        ji = np.max([0, np.min([nSample - 1, ii + iInc])])
        jb = np.max([0, np.min([nSample - 1, ii + iInc * b])])

        # check limits on lag indices
        lMinus1 = ll - 1

        if lMinus1 < 0: # check lag index is greater than 1
            lMinus1 = 0 # make lag = first lag

        lPlus1 = ll + 1

        if lPlus1 > nLag - 1: # check lag index less than max lag
            lPlus1 = nLag - 1

        # get distance at lags (ll-1, ll, ll+1)
        distLminus1 = d[jb, lMinus1] # minus:  d[i-b, j-1]
        distL = d[ji,ll] # actual d[i-1, j]
        distLplus1 = d[jb, lPlus1] # plus d[i-b, j+1]

        # equation 10 in Hale (2013)
        # sum errors over i-1:i-b+1
        if ji != jb:
            for kb in range(ji, jb - iInc - 1, iInc):
                distLminus1 = distLminus1 + err[kb, lMinus1]
                distLplus1  = distLplus1  + err[kb, lPlus1]

        # update minimum distance to previous sample
        dl = np.min([distLminus1, distL, distLplus1 ])

        if dl != distL: # then ll ~= ll and we check forward and backward
            if dl == distLminus1:
                ll = lMinus1
            else:
                ll = lPlus1

        # assume ii = ii - 1
        ii += iInc

        # absolute integer of lag
        stbar[ii] = ll + lmin

        # now move to correct time index, if smoothing difference over many
        # time samples using 'b'
        if (ll == lMinus1) | (ll == lPlus1): # check edges to see about b values
            if ji != jb: # if b>1 then need to move more steps
                for kb in range(ji, jb - iInc - 1, iInc):
                    ii = ii + iInc # move from i-1:i-b-1
                    stbar[ii] = ll + lmin  # constant lag over that time

    return stbar


def wct_modified(y1, y2, dt, dj=1/12, s0=-1, J=-1, sig=True, significance_level=0.95, wavelet='morlet', normalize=True, **kwargs):
    '''
    Wavelet coherence transform (WCT).
​
    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.
​
    Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
    significance_level (float, optional) :
        Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.
​
    Returns
    -------
    TODO: Something TBA and TBC
​
    See also
    --------
    cwt, xwt
​
    '''

    wavelet = pycwt.wavelet._check_parameter_wavelet(wavelet)
    # Checking some input parameters
    if s0 == -1:
        # Number of scales
        s0 = 2 * dt / wavelet.flambda()
    if J == -1:
        # Number of scales
        J = np.int(np.round(np.log2(y1.size * dt / s0) / dj))

    # Makes sure input signals are numpy arrays.
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    # Calculates the standard deviation of both input signals.
    std1 = y1.std()
    std2 = y2.std()
    # Normalizes both signals, if appropriate.
    if normalize:
        y1_normal = (y1 - y1.mean()) / std1
        y2_normal = (y2 - y2.mean()) / std2
    else:
        y1_normal = y1
        y2_normal = y2

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = pycwt.cwt(y1_normal, dt, **_kwargs)
    W2, sj, freq, coi, _, _ = pycwt.cwt(y2_normal, dt, **_kwargs)

    scales1 = np.ones([1, y1.size]) * sj[:, None]
    scales2 = np.ones([1, y2.size]) * sj[:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = wavelet.smooth(np.abs(W1) ** 2 / scales1, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(W2) ** 2 / scales2, dt, dj, sj)

    # Now the wavelet transform coherence
    W12 = W1 * W2.conj()
    scales = np.ones([1, y1.size]) * sj[:, None]
    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
    WCT = np.abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(W12)

    # Calculate cross spectrum & its amplitude
    WXS, WXA = W12, np.abs(S12)

    # Calculates the significance using Monte Carlo simulations with 95%
    # confidence as a function of scale.

    if sig:
        a1, b1, c1 = pycwt.ar1(y1)
        a2, b2, c2 = pycwt.ar1(y2)
        sig = pycwt.wct_significance(a1, a2, dt=dt, dj=dj, s0=s0, J=J,
                               significance_level=significance_level,
                               wavelet=wavelet, **kwargs)
    else:
        sig = np.asarray([0])

    return WCT, aWCT, coi, freq, sig


################################################################
################ DISPERSION EXTRACTION FUNCTIONS ###############
################################################################

# function to extract the dispersion from the image
def extract_dispersion(amp,per,vel):
    '''
    this function takes the dispersion image from CWT as input, tracks the global maxinum on
    the wavelet spectrum amplitude and extract the sections with continous and high quality data

    PARAMETERS:
    ----------------
    amp: 2D amplitude matrix of the wavelet spectrum
    phase: 2D phase matrix of the wavelet spectrum
    per:  period vector for the 2D matrix
    vel:  vel vector of the 2D matrix
    RETURNS:
    ----------------
    per:  central frequency of each wavelet scale with good data
    gv:   group velocity vector at each frequency
    '''
    maxgap = 5
    nper = amp.shape[0]
    gv   = np.zeros(nper,dtype=np.float32)
    dvel = vel[1]-vel[0]

    # find global maximum
    for ii in range(nper):
        maxvalue = np.max(amp[ii],axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]

    # check the continuous of the dispersion
    for ii in range(1,nper-15):
        # 15 is the minumum length needed for output
        for jj in range(15):
            if np.abs(gv[ii+jj]-gv[ii+1+jj])>maxgap*dvel:
                gv[ii] = 0
                break

    # remove the bad ones
    indx = np.where(gv>0)[0]

    return per[indx],gv[indx]
