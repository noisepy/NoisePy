import os
import glob
import copy
import obspy
import scipy
import time
import pyasdf
import datetime
import numpy as np
import pandas as pd
from numba import jit
from scipy.signal import hilbert
from obspy.signal.util import _npts2nfft
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.filter import bandpass,lowpass
from obspy.core.util.base import _get_function_from_entry_point
from obspy.core.inventory import Inventory, Network, Station, Channel, Site


'''
this noise module includes
1) core functions called by the main NoisePy scripts directly;
2) utility functions used by the core functions.
by: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
    Marine Denolle (mdenolle@fas.harvard.edu)
several utility functions are copied from https://github.com/tclements/noise
'''

####################################################
############## CORE FUNCTIONS ######################
####################################################

def make_stationlist_CSV(inv,path):
    '''
    this function outputs station info collected by the obspy module of get_station into a CSV file
    (used in S0A_download_ASDF_MPI)
    
    PARAMETERS:
    ----------------
    inv:  obspy inventory from IRIS server
    path: absolute path to output the CSV file
    '''
    #----to hold all variables-----
    netlist = []
    stalist = []
    lonlist = []
    latlist = []
    elvlist = []
    chalist = []

    #-----silly inventory structures----
    nnet = len(inv)
    for ii in range(nnet):
        net = inv[ii]
        nsta = len(net)
        for jj in range(nsta):
            sta = net[jj]
            ncha = len(sta)
            for kk in range(ncha):
                chan = sta[kk]
                netlist.append(net.code)
                stalist.append(sta.code)
                chalist.append(chan.code)
                lonlist.append(sta.longitude)
                latlist.append(sta.latitude)
                elvlist.append(sta.elevation)

    #------------dictionary for a pandas frame------------
    dict = {'network':netlist,'station':stalist,'channel':chalist,'latitude':latlist,'longitude':lonlist,'elevation':elvlist}
    locs = pd.DataFrame(dict)

    #----------write into a csv file---------------            
    locs.to_csv(os.path.join(path,'station.lst'),index=False)


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
    st[0].taper(max_percentage=0.05,max_length=20)	# taper window
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
                    st[0].remove_response(output="VEL",pre_filt=pre_filt,water_level=60)
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
            seedresp = glob.glob(os.path.join(respdir,'RESP.'+station+'*'))
            if len(seedresp)==0:
                raise ValueError('no RESP files found for %s' % station)
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
            if os.path.isfile(invfile):
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


def cut_trace_make_statis(fc_para,source):
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

    # confim data has been correctly pre-processed
    if data.size < sps*inc_hours*3600:
        return source_params,dataS_t,dataS

    # statistic to detect segments that may be associated with earthquakes
    all_madS = mad(data)	            # median absolute deviation over all noise window
    all_stdS = np.std(data)	        # standard deviation over all noise window
    if all_madS==0 or all_stdS==0 or np.isnan(all_madS) or np.isnan(all_stdS):
        print("continue! madS or stdS equeals to 0 for %s" % source)
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
    to_whiten   = fft_para['to_whiten']
    smooth_N    = fft_para['smooth_N']
    N = dataS.shape[0]

    #------to normalize in time or not------
    if time_norm:

        if time_norm == 'one_bit': 	# sign normalization
            white = np.sign(dataS)
        elif time_norm == 'running_mean': # running mean: normalization over smoothed absolute average           
            white = np.zeros(shape=dataS.shape,dtype=dataS.dtype)
            for kkk in range(N):
                white[kkk,:] = dataS[kkk,:]/moving_ave(np.abs(dataS[kkk,:]),smooth_N)

    else:	# don't normalize
        white = dataS

    #-----to whiten or not------
    if to_whiten:
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

    elif cc_method == 'raw':
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
        n_corr = nwin
        s_corr = np.zeros(Nfft,dtype=np.float32)
        t_corr = dataS_t[0]
        crap   = np.zeros(Nfft,dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr,axis=0)
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
        'comp',comp}
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
    npts = cc_array.shape[1]

    # remove abnormal data     
    ampmax = np.max(cc_array,axis=1)
    tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
    if not len(tindx):
        allstacks1=[];allstacks2=[];nstacks=0
    else:
        cc_array = cc_array[tindx,:]
        cc_time  = cc_time[tindx]
        cc_ngood = cc_ngood[tindx]

        # do stacking
        allstacks1 = np.zeros(npts,dtype=np.float32)
        allstacks2 = np.zeros(npts,dtype=np.float32)

        if smethod == 'linear':
            allstacks1 = np.mean(cc_array,axis=0)
        elif smethod == 'pws':
            allstacks1 = pws(cc_array,samp_freq) 
        elif smethod == 'both':
            allstacks1 = np.mean(cc_array,axis=0)
            allstacks2 = pws(cc_array,samp_freq) 
        nstacks = np.sum(cc_ngood)
    
    # good to return
    if smethod != 'both':
        return cc_array,cc_ngood,cc_time,allstacks1,nstacks
    else:
        return cc_array,cc_ngood,cc_time,allstacks1,allstacks2,nstacks

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
        to_whiten: whitening method between 'one-bit' and 'running-mean'
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    '''

    # load parameters
    delta   = fft_para['dt']
    freqmin = fft_para['freqmin']
    freqmax = fft_para['freqmax']
    smooth_N  = fft_para['smooth_N']
    to_whiten = fft_para['to_whiten']

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
        if to_whiten=='one-bit':
            FFTRawSign[:,left:right] = np.exp(1j * np.angle(FFTRawSign[:,left:right]))
        elif to_whiten == 'running-mean':
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
        if to_whiten == 'one-bit':
            FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        elif to_whiten == 'running-mean':
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


def pws(arr,sampling_rate,power=2,pws_timegate=5.):
