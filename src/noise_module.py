import os
import glob
import copy
import scipy
import time
import pyasdf
import datetime
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft,next_fast_len
from scipy.signal import butter, hilbert, wiener
from scipy.linalg import svd
from obspy.signal.filter import bandpass,lowpass
from obspy.signal.regression import linear_regression
from obspy.signal.invsim import cosine_taper
import obspy
from obspy.signal.util import _npts2nfft
from obspy.core.inventory import Inventory, Network, Station, Channel, Site


def cut_trace_make_statis(fft_para,source,flag):
    '''
    cut continous noise data into user-defined segments, estimate the statistics of 
    each segment and keep timestamp for later use

    fft_para: dictionary containing all useful variables for later fft
    source: obspy stream objects of noise data
    flag: boolen variable to output intermediate variables or not
    '''
    # define return variables first
    source_params=[];dataS_t=[];dataS=[]

    # load parameters from structure
    cc_len = fft_para['cc_len']
    step   = fft_para['step']

    # statistic to detect segments that may be associated with earthquakes
    all_madS = mad(source[0].data)
    all_stdS = np.std(source[0].data)
    if all_madS==0 or all_stdS==0 or np.isnan(all_madS) or np.isnan(all_stdS):
        print("continue! madS or stdS equeals to 0 for %s" % source)
        return source_params,dataS_t,dataS

    trace_madS = []
    trace_stdS = []
    nonzeroS = []
    nptsS = []
    source_slice = obspy.Stream()

    #--------break a continous recording into pieces----------
    t0=time.time()
    for ii,win in enumerate(source[0].slide(window_length=cc_len, step=step)):
        win.detrend(type="constant")
        win.detrend(type="linear")
        trace_madS.append(np.max(np.abs(win.data))/all_madS)
        trace_stdS.append(np.max(np.abs(win.data))/all_stdS)
        nonzeroS.append(np.count_nonzero(win.data)/win.stats.npts)
        nptsS.append(win.stats.npts)
        win.taper(max_percentage=0.05,max_length=20)
        source_slice.append(win)
    
    t1=time.time()
    if flag:
        print("breaking records takes %f s"%(t1-t0))

    if len(source_slice) == 0:
        print("No traces for %s " % source)
        return source_params,dataS_t,dataS
    else:
        source_params = np.vstack([trace_madS,trace_stdS,nonzeroS]).T

    #---------seems unnecessary for data already pre-processed with same length (zero-padding)-------
    Nseg   = len(source_slice)
    Npts   = np.max(nptsS)
    dataS_t= np.zeros(shape=(Nseg,2))
    dataS  = np.zeros(shape=(Nseg,Npts),dtype=np.float32)
    for ii,trace in enumerate(source_slice):
        dataS_t[ii,0]= source_slice[ii].stats.starttime-obspy.UTCDateTime(1970,1,1)# convert to dataframe
        dataS_t[ii,1]= source_slice[ii].stats.endtime -obspy.UTCDateTime(1970,1,1)# convert to dataframe
        dataS[ii,0:nptsS[ii]] = trace.data

    return source_params,dataS_t,dataS


def noise_processing(fft_para,dataS,flag):
    '''
    perform time domain and frequency normalization according to user's need. note that
    this step is not recommended if deconv or coherency method is selected for calculating
    cross-correlation functions. 

    fft_para: dictionary containing all useful variables used for fft
    dataS: data matrix containing all segmented noise data
    flag: boolen variable to output intermediate variables or not
    '''
    # load parameters first
    time_norm   = fft_para['time_norm']
    to_whiten   = fft_para['to_whiten']
    smooth_N    = fft_para['smooth_N']
    N = dataS.shape[0]

    #------to normalize in time or not------
    if time_norm:
        t0=time.time()   

        if time_norm == 'one_bit': 
            white = np.sign(dataS)
        elif time_norm == 'running_mean':
            
            #--------convert to 1D array for smoothing in time-domain---------
            white = np.zeros(shape=dataS.shape,dtype=dataS.dtype)
            for kkk in range(N):
                white[kkk,:] = dataS[kkk,:]/moving_ave(np.abs(dataS[kkk,:]),smooth_N)

        t1=time.time()
        if flag:
            print("temporal normalization takes %f s"%(t1-t0))
    else:
        white = dataS

    #-----to whiten or not------
    if to_whiten:

        t0=time.time()
        source_white = whiten(white,fft_para)
        t1=time.time()
        if flag:
            print("spectral whitening takes %f s"%(t1-t0))
    else:

        Nfft = int(next_fast_len(int(dataS.shape[1])))
        source_white = scipy.fftpack.fft(white, Nfft, axis=1)
    
    return source_white

def smooth_source_spect(cc_para,fft1):
    '''
    take the source/receiver spectrum normalization for coherency/deconv method out of the
    innermost for loop to save time

    input cc_para: dictionary containing useful cc parameters
          fft1: complex matrix containing source spectrum
    output sfft1: complex numpy array with normalized spectrum
    '''
    cc_method = cc_para['cc_method']
    smoothspect_N = cc_para['smoothspect_N']

    if cc_method == 'deconv':
        
        #-----normalize single-station cc to z component-----
        temp = moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = np.conj(fft1)/temp**2
        except ValueError:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'coherency':
        temp = moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = np.conj(fft1)/temp
        except ValueError:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'raw':
        sfft1 = np.conj(fft1)
    
    return sfft1

def stats2inv(stats,resp=None,filexml=None,locs=None):

    # We'll first create all the various objects. These strongly follow the
    # hierarchy of StationXML files.
    inv = Inventory(networks=[],source="homegrown")

    if locs is None:
        net = Network(
            # This is the network code according to the SEED standard.
            code=stats.network,
            # A list of stations. We'll add one later.
            stations=[],
            description="Marine created from SAC and resp files",
            # Start-and end dates are optional.
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

    else:
        ista=locs[locs['station']==stats.station].index.values.astype('int64')[0]

        net = Network(
            # This is the network code according to the SEED standard.
            code=locs.iloc[ista]["network"],
            # A list of stations. We'll add one later.
            stations=[],
            description="Marine created from SAC and resp files",
            # Start-and end dates are optional.
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
            # This is the channel code according to the SEED standard.
            code=stats.channel,
            # This is the location code according to the SEED standard.
            location_code=stats.location,
            # Note that these coordinates can differ from the station coordinates.
            latitude=locs.iloc[ista]["latitude"],
            longitude=locs.iloc[ista]["longitude"],
            elevation=locs.iloc[ista]["elevation"],
            depth=-locs.iloc[ista]["elevation"],
            azimuth=0,
            dip=0,
            sample_rate=stats.sampling_rate)

    response = obspy.core.inventory.response.Response()
    if resp is not None:
        print('i dont have the response')
    # By default this accesses the NRL online. Offline copies of the NRL can
    # also be used instead
    # nrl = NRL()
    # The contents of the NRL can be explored interactively in a Python prompt,
    # see API documentation of NRL submodule:
    # http://docs.obspy.org/packages/obspy.clients.nrl.html
    # Here we assume that the end point of data logger and sensor are already
    # known:
    #response = nrl.get_response( # doctest: +SKIP
    #    sensor_keys=['Streckeisen', 'STS-1', '360 seconds'],
    #    datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])


    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)

    # And finally write it to a StationXML file. We also force a validation against
    # the StationXML schema to ensure it produces a valid StationXML file.
    #
    # Note that it is also possible to serialize to any of the other inventory
    # output formats ObsPy supports.
    if filexml is not None:
        inv.write(filexml, format="stationxml", validate=True)

    return inv        

def sta_info_from_inv(inv):
    '''
    output station information from reading the inventory info

    input parameter of inv: station inventory 
    '''
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

def preprocess_raw(st,inv,prepro_para,date_info):
    '''
    pre-process the raw stream of data including:
    - check whether sample rate is matching (from original process_raw)
    - remove trend and mean of each trace
    - filter and downsample the data if needed (from original process_raw) and correct the
    time if integer time are between sampling points
    - remove instrument responses with selected methods. 
        "inv"        -> using inventory information to remove_response;
        "spectrum"   -> use the response spectrum (inverse; recommened due to efficiency). note
        that one script is provided in the package to estimate response spectrum from RESP files
        "RESP_files" -> use the raw download RESP files
        "polezeros"  -> use pole/zero info for a crude correction of response
    - trim data to a day-long sequence and interpolate it to ensure starting at 00:00:00.000

    st: obspy stream object, containing traces of noise data
    inv: obspy inventory object, containing all information about stations
    prepro_para: dictionary containing all useful fft parameters
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

    if len(st)>1:st.merge(method=1,fill_value=0)

    # make downsampling if needed
    if abs(samp_freq-sps) > 1E-4:
        st[0].data = bandpass(st[0].data,pre_filt[0],pre_filt[-1],df=sps,corners=4,zerophase=True)

        # downsampling here
        st.interpolate(samp_freq,method='weighted_average_slopes')
        delta = st[0].stats.delta

        # when starttimes are between sampling points
        fric = st[0].stats.starttime.microsecond%(delta*1E6)
        if fric>1E-4:
            st[0].data = segment_interpolate(np.float32(st[0].data),float(fric/delta*1E6))
            #--reset the time to remove the discrepancy---
            st[0].stats.starttime-=(fric*1E-6)

    # several options to remove instrument response
    if rm_resp:
        if rm_resp != 'inv':
            if (respdir is None) or (not os.path.isdir(respdir)):
                raise ValueError('response file folder not found! abort!')

        if rm_resp == 'inv':
            #----check whether inventory is attached----
            if not inv[0][0][0].response:
                raise ValueError('no response found in the inventory! abort!')
            else:
                print('removing response for %s using inv'%st[0])
                st[0].attach_response(inv)
                st[0].remove_response(output="VEL",pre_filt=pre_filt,water_level=60)

        elif rm_resp == 'spectrum':
            print('remove response using spectrum')
            specfile = glob.glob(os.path.join(respdir,'*'+station+'*'))
            if len(specfile)==0:
                raise ValueError('no response sepctrum found for %s' % station)
            st = resp_spectrum(st,specfile[0],samp_freq,pre_filt)

        elif rm_resp == 'RESP_files':
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

    #-----fill gaps, trim data and interpolate to ensure all starts at 00:00:00.0------
    st = clean_daily_segments(st,date_info)

    return st

def portion_gaps(stream,date_info):
    '''
    get the accumulated gaps (npts) from the accumulated difference between starttime and endtime.
    trace with gap length of 30% of trace size is removed. 

    stream: obspy stream object
    return float: portions of gaps in stream
    '''
    # ideal duration of data
    starttime = date_info['starttime']
    endtime   = date_info['endtime']
    npts      = (endtime-starttime)*stream[0].stats.sampling_rate

    pgaps=0
    #loop through all trace to accumulate gaps
    for ii in range(len(stream)-1):
        pgaps += (stream[ii+1].stats.starttime-stream[ii].stats.endtime)*stream[ii].stats.sampling_rate
    return pgaps/npts

@jit('float32[:](float32[:],float32)')
def segment_interpolate(sig1,nfric):
    '''
    a sub-function of clean_daily_segments:

    interpolate the data according to fric to ensure all points located on interger times of the
    sampling rate (e.g., starttime = 00:00:00.015, delta = 0.05.)

    input parameters:
    sig1:  float32 -> seismic recordings in a 1D array
    nfric: float32 -> the amount of time difference between the point and the adjacent assumed samples
    '''
    npts = len(sig1)
    sig2 = np.zeros(npts,dtype=np.float32)

    #----instead of shifting, do a interpolation------
    for ii in range(npts):

        #----deal with edges-----
        if ii==0 or ii==npts:
            sig2[ii]=sig1[ii]
        else:
            #------interpolate using a hat function------
            sig2[ii]=(1-nfric)*sig1[ii+1]+nfric*sig1[ii]

    return sig2

def resp_spectrum(source,resp_file,downsamp_freq,pre_filt=None):
    '''
    remove the instrument response with response spectrum from evalresp.
    the response spectrum is evaluated based on RESP/PZ files and then 
    inverted using obspy function of invert_spectrum. 
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
        source[0].data = bandpass(source[0].data,pre_filt[0],pre_filt[-1],df=sps,corners=4,zerophase=True)

    return source

def clean_daily_segments(tr,date_info):
    '''
    subfunction to clean the tr recordings. only the traces with at least 0.5-day long
    sequence (respect to 00:00:00.0 of the day) is kept. note that the trace here could
    be of several days recordings, so this function helps to break continuous chunck 
    into a day-long segment from 00:00:00.0 to 24:00:00.0.

    tr: obspy stream object
    return ntr: obspy stream object
    '''
    # duration of data
    starttime = date_info['starttime']
    endtime = date_info['endtime']

    # make a new stream 
    ntr = obspy.Stream()
    # trim a continous segment into user-defined sequences
    tr[0].trim(starttime=starttime,endtime=endtime,pad=True,fill_value=0)
    ntr.append(tr[0])

    return ntr

def make_stationlist_CSV(inv,path):
    '''
    subfunction to output the station list into a CSV file
    inv: inventory information passed from IRIS server
    '''
    #----to hold all variables-----
    netlist = []
    stalist = []
    lonlist = []
    latlist = []
    elvlist = []

    #-----silly inventory structures----
    nnet = len(inv)
    for ii in range(nnet):
        net = inv[ii]
        nsta = len(net)
        for jj in range(nsta):
            sta = net[jj]
            netlist.append(net.code)
            stalist.append(sta.code)
            lonlist.append(sta.longitude)
            latlist.append(sta.latitude)
            elvlist.append(sta.elevation)

    #------------dictionary for a pandas frame------------
    dict = {'network':netlist,'station':stalist,'latitude':latlist,'longitude':lonlist,'elevation':elvlist}
    locs = pd.DataFrame(dict)

    #----------write into a csv file---------------            
    locs.to_csv(os.path.join(path,'locations.txt'),index=False)


def get_event_list(str1,str2,inc_hours):
    '''
    return the event list in the formate of 2010_01_01 by taking
    advantage of the datetime modules
    
    str1: string of starting date -> 2010_01_01
    str2: string of ending date -> 2010_10_11
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

def get_station_pairs(sta):
    '''
    construct station pairs based on the station list
    works same way as the function of itertools
    '''
    pairs=[]
    for ii in range(len(sta)-1):
        for jj in range(ii+1,len(sta)):
            pairs.append((sta[ii],sta[jj]))
    return pairs

@jit('float32(float32,float32,float32,float32)') 
def get_distance(lon1,lat1,lon2,lat2):
    '''
    calculate distance between two points on earth
    
    lon:longitude in degrees
    lat:latitude in degrees
    '''
    R = 6372800  # Earth radius in meters
    pi = 3.1415926536
    
    phi1    = lat1*pi/180
    phi2    = lat2*pi/180
    dphi    = (lat2 - lat1)*pi/180
    dlambda = (lon2 - lon1)*pi/180
    
    a = np.sin(dphi/2)**2+np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1 - a))/1000

def get_coda_window(dist,vmin,maxlag,dt,wcoda):
    '''
    calculate the coda wave window for the ccfs based on
    the travel time of the balistic wave and select the 
    index for the time window
    '''
    #--------construct time axis----------
    tt = np.arange(-maxlag/dt, maxlag/dt+1)*dt

    #--get time window--
    tbeg=int(dist/vmin)
    tend=tbeg+wcoda
    if tend>maxlag:
        raise ValueError('time window ends at maxlag, too short!')
    if tbeg>maxlag:
        raise ValueError('time window starts later than maxlag')
    
    #----convert to point index----
    ind1 = np.where(abs(tt)==tbeg)[0]
    ind2 = np.where(abs(tt)==tend)[0]

    if len(ind1)!=2 or len(ind2)!=2:
        raise ValueError('index for time axis is wrong')
    ind = [ind2[0],ind1[0],ind1[1],ind2[1]]

    return ind    

def whiten(data, fft_para):
    """This function takes 1-dimensional *data* timeseries array,
    goes to frequency domain using fft, whitens the amplitude of the spectrum
    in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.

    :type data: :class:`numpy.ndarray`
    :param data: Contains the 1D time series to whiten
    :type Nfft: int
    :param Nfft: The number of points to compute the FFT
    :type delta: float
    :param delta: The sampling frequency of the `data`
    :type freqmin: float
    :param freqmin: The lower frequency bound
    :type freqmax: float
    :param freqmax: The upper frequency bound

    :rtype: :class:`numpy.ndarray`
    :returns: The FFT of the input trace, whitened between the frequency bounds
    """

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


def cross_corr_parameters(source, receiver, start_end_t, source_params,
    receiver_params, locs, maxlag):
    """ 
    Creates parameter dict for cross-correlations and header info to ASDF.

    :type source: `~obspy.core.trace.Stats` object.
    :param source: Stats header from xcorr source station
    :type receiver: `~obspy.core.trace.Stats` object.
    :param receiver: Stats header from xcorr receiver station
    :type start_end_t: `~np.ndarray`
    :param start_end_t: starttime, endtime of cross-correlation (UTCDateTime)
    :type source_params: `~np.ndarray`
    :param source_params: max_mad,max_std,percent non-zero values of source trace
    :type receiver_params: `~np.ndarray`
    :param receiver_params: max_mad,max_std,percent non-zero values of receiver trace
    :type locs: dict
    :param locs: dict with latitude, elevation_in_m, and longitude of all stations
    :type maxlag: int
    :param maxlag: number of lag points in cross-correlation (sample points) 
    :return: Auxiliary data parameter dict
    :rtype: dict

    """

    # source and receiver locations in dict with lat, elevation_in_m, and lon
    source_loc = locs.ix[source['network'] + '.' + source['station']]
    receiver_loc = locs.ix[receiver['network'] + '.' + receiver['station']]

    # # get distance (in km), azimuth and back azimuth
    dist,azi,baz = calc_distance(source_loc,receiver_loc) 

    source_mad,source_std,source_nonzero = source_params[:,0],\
                         source_params[:,1],source_params[:,2]
    receiver_mad,receiver_std,receiver_nonzero = receiver_params[:,0],\
                         receiver_params[:,1],receiver_params[:,2]
    
    starttime = start_end_t[:,0] - obspy.UTCDateTime(1970,1,1)
    starttime = starttime.astype('float')
    endtime = start_end_t[:,1] - obspy.UTCDateTime(1970,1,1)
    endtime = endtime.astype('float')
    source = stats_to_dict(source,'source')
    receiver = stats_to_dict(receiver,'receiver')
    # fill Correlation attribDict 
    parameters = {'source_mad':source_mad,
            'source_std':source_std,
            'source_nonzero':source_nonzero,
            'receiver_mad':receiver_mad,
            'receiver_std':receiver_std,
            'receiver_nonzero':receiver_nonzero,
            'dist':dist,
            'azi':azi,
            'baz':baz,
            'lag':maxlag,
            'starttime':starttime,
            'endtime':endtime}
    parameters.update(source)
    parameters.update(receiver)
    return parameters    

def C3_process(S1_data,S2_data,Nfft,win):
    '''
    performs all C3 processes including 1) cutting the time window for P-N parts;
    2) doing FFT for the two time-seris; 3) performing cross-correlations in freq;
    4) ifft to time domain
    '''
    #-----initialize the spectrum variables----
    ccp1 = np.zeros(Nfft,dtype=np.complex64)
    ccn1 = ccp1
    ccp2 = ccp1
    ccn2 = ccp1
    ccp  = ccp1
    ccn  = ccp1

    #------find the time window for sta1------
    S1_data_N = S1_data[win[0]:win[1]]
    S1_data_N = S1_data_N[::-1]
    S1_data_P = S1_data[win[2]:win[3]]
    S2_data_N = S2_data[win[0]:win[1]]
    S2_data_N = S2_data_N[::-1]
    S2_data_P = S2_data[win[2]:win[3]]

    #---------------do FFT-------------
    ccp1 = scipy.fftpack.fft(S1_data_P, Nfft)
    ccn1 = scipy.fftpack.fft(S1_data_N, Nfft)
    ccp2 = scipy.fftpack.fft(S2_data_P, Nfft)
    ccn2 = scipy.fftpack.fft(S2_data_N, Nfft)

    #------cross correlations--------
    ccp = np.conj(ccp1)*ccp2
    ccn = np.conj(ccn1)*ccn2

    return ccp,ccn
    
def optimized_cc_parameters(cc_para,coor,tcorr,ncorr):
    '''
    provide the parameters for computting CC later
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
        'lag':int(maxlag),
        'dist':np.float32(dist/1000),
        'azi':np.float32(azi),
        'baz':np.float32(baz),
        'lonS':np.float32(lonS),
        'latS':np.float32(latS),
        'lonR':np.float32(lonR),
        'latR':np.float32(latR),
        'ngood':ncorr,
        'cc_method':cc_method,
        'time':tcorr}
    return parameters

def optimized_correlate1(fft1_smoothed_abs,fft2,maxlag,dt,Nfft,nwin,method="cross-correlation"):
    '''
    Optimized version of the correlation functions: put the smoothed 
    source spectrum amplitude out of the inner for loop. 
    It also takes advantage of the linear relationship of ifft, so that
    stacking in spectrum first to reduce the total number of times for ifft,
    which is the most time consuming steps in the previous correlate function  
    '''

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*(Nfft//2),dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size,) * fft2.reshape(fft2.size,)

    if method == "coherence":
        temp = moving_ave(np.abs(fft2.reshape(fft2.size,)),10)
        try:
            corr /= temp
        except ValueError:
            raise ValueError('smoothed spectrum has zero values')

    corr  = corr.reshape(nwin,Nfft//2)
    ncorr = np.zeros(shape=Nfft,dtype=np.complex64)
    ncorr[:Nfft//2] = np.mean(corr,axis=0)
    ncorr[-(Nfft//2)+1:]=np.flip(np.conj(ncorr[1:(Nfft//2)]),axis=0)
    ncorr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(ncorr, Nfft, axis=0)))

    tcorr = np.arange(-Nfft//2 + 1, Nfft//2)*dt
    ind   = np.where(np.abs(tcorr) <= maxlag)[0]
    ncorr = ncorr[ind]
    
    return ncorr

def optimized_correlate(fft1_smoothed_abs,fft2,D,Nfft,dataS_t):
    '''
    Optimized version of the correlation functions: put the smoothed 
    source spectrum amplitude out of the inner for loop. 
    It also takes advantage of the linear relationship of ifft, so that
    stacking in spectrum first to reduce the total number of times for ifft,
    which is the most time consuming steps in the previous correlate function.
    Modified by Marine on 02/25/19 to accommodate sub-stacking of over tave seconds in the day
    step is overlap step   

    fft1_smoothed_abs: already smoothed power spectral density of the FFT from source station
    fft2: FFT from receiver station
    D: input dictionary with the following parameters:
        D["maxlag"]: maxlag to keep in the cross correlation
        D["dt"]: sampling rate (in s)
        D["Nfft"]: number of frequency points
        D["nwin"]: number of windows
        D["method"]: either cross-correlation or deconvolution or coherency
        D["freqmin"]: minimum frequency to look at (Hz)
        D["freqmax"]: maximum frequency to look at (Hz)
    Timestamp: array of datetime object.
    '''
    #----load paramters----
    dt      = D['dt']
    freqmin = D['freqmin']
    freqmax = D['freqmax']
    maxlag  = D['maxlag']
    method  = D['cc_method']
    cc_len  = D['cc_len'] 
    substack= D['substack']                                                              # CJ
    substack_len  = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    nwin  = fft1_smoothed_abs.shape[0]
    Nfft2 = Nfft//2

    # convert UTC time to datetime64
    Timestamps = np.empty(dataS_t.size,dtype='datetime64[s]') 
    for ii in range(dataS_t.shape[0]):
        Timestamps[ii] = datetime.datetime.fromtimestamp(dataS_t[ii])

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size,)*fft2.reshape(fft2.size,)

    if method == "coherency":
        temp = moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)               # CJ
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    #--------------- remove outliers in frequency domain -------------------
    # reduce the number of IFFT by pre-selecting good windows before substack
    freq = scipy.fftpack.fftfreq(Nfft, d=dt)[:Nfft2]
    i1 = np.where( (freq>=freqmin) & (freq <= freqmax))[0]

    ###################DOUBLE CHECK THIS SECTION#####################
    # this creates the residuals between each window and their median
    med = np.log10(np.median(corr[:,i1],axis=0))
    r   = np.log10(corr[:,i1]) - med
    #ik  = np.where(r>=med-5 & r<=med+5)[0]
    ik  = np.zeros(nwin,dtype=np.int)
    for i in range(nwin):
        if np.any( (r[i,:]>=med-10) & (r[i,:]<=med+10) ):ik[i]=i
    ik1=np.nonzero(ik)
    ik=ik[ik1]
    #################################################################

    if substack:
        if substack_len == cc_len:
            # choose to keep all fft data for a day
            n_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)
            c_corr = np.ones(nwin,dtype=np.int16)
            t_corr = Timestamps
            crap   = np.zeros(Nfft,dtype=np.complex64)
            for i in range(len(ik)):            
                crap[:Nfft2] = corr[ik[i],:]
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                n_corr[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
        
        else:     
            # make substacks for all windows that either start or end within a period of increment
            Ttotal = Timestamps[-1]-Timestamps[0]             # total duration of what we have now
            dtime  = np.timedelta64(substack_len,'s')

            # number of data chuncks
            tstart = Timestamps[0]
            i=0 
            n_corr = np.zeros(shape=(int(Ttotal/dtime),Nfft),dtype=np.float32)
            t_corr = np.empty(int(Ttotal/dtime),dtype='datetime64[s]')
            c_corr = np.zeros(int(Ttotal/dtime),dtype=np.int)
            crap   = np.zeros(Nfft,dtype=np.complex64)

            # set the endtime variable
            if dtime == cc_len:                                                                     # CJ 06/25
                tend = Timestamps[-1]                                                                # CJ
            else:                                                                                   # CJ
                tend = Timestamps[-1]-dtime                                                          # CJ

            while tstart < tend:                                                                    # CJ
                # find the indexes of all of the windows that start or end within 
                itime = np.where( (Timestamps[ik] >= tstart) & (Timestamps[ik] < tstart+dtime) )[0]   # CJ
                if len(ik[itime])==0:tstart+=dtime;continue
                
                crap[:Nfft2] = np.mean(corr[ik[itime],:],axis=0)   # linear average of the correlation 
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                n_corr[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                c_corr[i] = len(ik[itime])          # number of windows stacks
                t_corr[i] = tstart                   # save the time stamps
                tstart += dtime
                print('correlation done and stacked at time %s' % str(t_corr[i]))
                i+=1
    else:
        # average daily cross correlation functions
        c_corr = len(ik)
        n_corr = np.zeros(Nfft,dtype=np.float32)
        t_corr = Timestamps[0]
        crap  = np.zeros(Nfft,dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr[ik,:],axis=0)
        crap[:Nfft2] = crap[:Nfft//2]-np.mean(crap[:Nfft2],axis=0)
        crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
        n_corr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    t = np.arange(-Nfft2+1, Nfft2)*dt
    ind   = np.where(np.abs(t) <= maxlag)[0]
    if n_corr.ndim==1:
        n_corr = n_corr[ind]
    elif n_corr.ndim==2:
        n_corr = n_corr[:,ind]
    return n_corr,t_corr,c_corr


@jit(nopython = True)
def moving_ave(A,N):
    '''
    Numba compiled function to do running smooth average.
    N is the the half window length to smooth
    A and B are both 1-D arrays (which runs faster compared to 2-D operations)
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


def get_SNR(corr,snr_parameters,parameters):
    '''
    estimate the SNR for the cross-correlation functions. the signal is defined
    as the maxinum in the time window of [dist/max_vel,dist/min_vel]. the noise
    is defined as the std of the trailing 100 s window. flag is to indicate to 
    estimate both lags of the cross-correlation funciton of just the positive

    corr: the noise cross-correlation functions
    snr_parameters: dictionary for some parameters to estimate S-N
    parameters: dictionary for parameters about the ccfs
    '''
    #---------common variables----------
    sampling_rate = int(1/parameters['dt'])
    npts = int(2*sampling_rate*parameters['lag'])
    indx = npts//2
    dist = parameters['dist']
    minvel = snr_parameters['minvel']
    maxvel = snr_parameters['maxvel']

    #-----index to window the signal part------
    indx_sig1 = int(dist/maxvel)*sampling_rate
    indx_sig2 = int(dist/minvel)*sampling_rate
    if maxvel > 5:
        indx_sig1 = 0

    #-------index to window the noise part---------
    indx_noise1 = indx_sig2
    indx_noise2 = indx_noise1+snr_parameters['noisewin']*sampling_rate

    #----prepare the filters----
    fb = snr_parameters['freqmin']
    fe = snr_parameters['freqmax']
    ns = snr_parameters['steps']
    freq = np.zeros(ns,dtype=np.float32)
    psnr = np.zeros(ns,dtype=np.float32)
    nsnr = np.zeros(ns,dtype=np.float32)
    ssnr = np.zeros(ns,dtype=np.float32)

    #--------prepare frequency info----------
    step = (np.log(fb)-np.log(fe))/(ns-1)
    for ii in range(ns):
        freq[ii]=np.exp(np.log(fe)+ii*step)

    for ii in range(1,ns-1):
        f2 = freq[ii-1]
        f1 = freq[ii+1]

        #-------------filter data before estimate SNR------------
        ncorr = bandpass(corr,f1,f2,sampling_rate,corners=4,zerophase=True)
        psignal = max(ncorr[indx+indx_sig1:indx+indx_sig2])
        nsignal = max(ncorr[indx-indx_sig2:indx-indx_sig1])
        ssignal = max((ncorr[indx+indx_sig1:indx+indx_sig2]+np.flip(ncorr[indx-indx_sig2:indx-indx_sig1]))/2)
        pnoise  = np.std(ncorr[indx+indx_noise1:indx+indx_noise2])
        nnoise  = np.std(ncorr[indx-indx_noise2:indx-indx_noise1])
        snoise  = np.std((ncorr[indx+indx_noise1:indx+indx_noise2]+np.flip(ncorr[indx-indx_noise2:indx-indx_noise1]))/2)
        
        #------in case there is no data-------
        if pnoise==0 or nnoise==0 or snoise==0:
            psnr[ii]=0
            nsnr[ii]=0
            ssnr[ii]=0
        else:
            psnr[ii] = psignal/pnoise
            nsnr[ii] = nsignal/nnoise
            ssnr[ii] = ssignal/snoise

        #------plot the signals-------
        '''
        plt.figure(figsize=(16,3))
        indx0 = 100*sampling_rate
        tt = np.arange(-100*sampling_rate,100*sampling_rate+1)/sampling_rate
        plt.plot(tt,ncorr[indx-indx0:indx+indx0+1],'k-',linewidth=0.6)
        plt.title('psnr %4.1f nsnr %4.1f ssnr %4.1f' % (psnr[ii],nsnr[ii],ssnr[ii]))
        plt.grid(True)
        plt.show()
        '''

    parameters['psnr'] = psnr[1:-1]
    parameters['nsnr'] = nsnr[1:-1]
    parameters['ssnr'] = nsnr[1:-1]
    parameters['freq'] = freq[1:-1]

    return parameters

def nextpow2(x):
    """
    Returns the next power of 2 of x.

    :type x: int 
    :returns: the next power of 2 of x

    """

    return int(np.ceil(np.log2(np.abs(x)))) 	

def abs_max(arr):
    """
    Returns array divided by its absolute maximum value.

    :type arr:`~numpy.ndarray` 
    :returns: Array divided by its absolute maximum value
    
    """
    
    return (arr.T / np.abs(arr.max(axis=-1))).T


def pws(arr,sampling_rate,power=2,pws_timegate=5.):
    """
    Performs phase-weighted stack on array of time series. 
    Modified on the noise function by Tim Climents.

    Follows methods of Schimmel and Paulssen, 1997. 
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    :type arr: numpy.ndarray
    :param arr: N length array of time series data 
    :type power: float
    :param power: exponent for phase stack
    :type sampling_rate: float 
    :param sampling_rate: sampling rate of time series 
    :type pws_timegate: float 
    :param pws_timegate: number of seconds to smooth phase stack
    :Returns: Phase weighted stack of time series data
    :rtype: numpy.ndarray  
    """

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


def norm(arr):
    """ Demean and normalize a given input to unit std. """
    arr -= arr.mean(axis=1, keepdims=True)
    return (arr.T / arr.std(axis=-1)).T

def check_sample_gaps(stream,date_info):
    """
    Returns sampling rate and gaps of traces in stream.

    :type stream:`~obspy.core.stream.Stream` object. 
    :param stream: Stream containing one or more day-long trace 
    :return: List of good traces in stream

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
    

def calc_distance(sta1,sta2):
    """ 
    Calcs distance in km, azimuth and back-azimuth between sta1, sta2. 

    Uses obspy.geodetics.base.gps2dist_azimuth for distance calculation. 
    :type sta1: dict
    :param sta1: dict with latitude, elevation_in_m, and longitude of station 1
    :type sta2: dict
    :param sta2: dict with latitude, elevation_in_m, and longitude of station 2
    :return: distance in km, azimuth sta1 -> sta2, and back azimuth sta2 -> sta1
    :rtype: float

    """

    # get coordinates 
    lon1 = sta1['longitude']
    lat1 = sta1['latitude']
    lon2 = sta2['longitude']
    lat2 = sta2['latitude']

    # calculate distance and return 
    dist,azi,baz = obspy.geodetics.base.gps2dist_azimuth(lat1,lon1,lat2,lon2)
    dist /= 1000.
    return dist,azi,baz


def fft_parameters(fft_para,source_params,inv,Nfft,data_t):
    """ 
    Creates parameter dict for cross-correlations and header info to ASDF.

    :type fft_para: python dictionary.
    :param fft_para: useful parameters used for fft
    :type source_params: `~np.ndarray`
    :param source_params: max_mad,max_std,percent non-zero values of source trace
    :type locs: dict
    :param locs: dict with latitude, elevation_in_m, and longitude of all stations
    :type component: char 
    :param component: component information about the data
    :type Nfft: int
    :param maxlag: number of fft points
    :type data_t: int matrix
    :param data_t: UTC date information
    :return: Auxiliary data parameter dict
    :rtype: dict

    """
    dt = fft_para['dt']
    cc_len = fft_para['cc_len']
    step   = fft_para['step']
    Nt     = data_t.shape[0]

    source_mad,source_std,source_nonzero = source_params[:,0],source_params[:,1],source_params[:,2]
    lon = inv[0][0].longitude
    lat = inv[0][0].latitude
    el  = inv[0][0].elevation
    parameters = {
             'dt':dt,
             'twin':cc_len,
             'step':step,
             'data_t':data_t,
             'nfft':Nfft,
             'nseg':Nt,
             'mad':source_mad,
             'std':source_std,
             'nonzero':source_nonzero,
             'longitude':lon,
             'latitude':lat,
             'elevation_in_m':el}
    return parameters   

def stats_to_dict(stats,stat_type):
    """
    Converts obspy.core.trace.Stats object to dict
    :type stats: `~obspy.core.trace.Stats` object.
    :type source: str
    :param source: 'source' or 'receiver'
    """
    stat_dict = {'{}_network'.format(stat_type):stats['network'],
                 '{}_station'.format(stat_type):stats['station'],
                 '{}_channel'.format(stat_type):stats['channel'],
                 '{}_delta'.format(stat_type):stats['delta'],
                 '{}_npts'.format(stat_type):stats['npts'],
                 '{}_sampling_rate'.format(stat_type):stats['sampling_rate']}
    return stat_dict 

def NCF_denoising(img_to_denoise,Mdate,Ntau,NSV):

	if img_to_denoise.ndim ==2:
		M,N = img_to_denoise.shape
		if NSV > np.min([M,N]):
			NSV = np.min([M,N])
		[U,S,V] = svd(img_to_denoise,full_matrices=False)
		S = scipy.linalg.diagsvd(S,S.shape[0],S.shape[0])
		Xwiener = np.zeros([M,N])
		for kk in range(NSV):
			SV = np.zeros(S.shape)
			SV[kk,kk] = S[kk,kk]
			X = U@SV@V
			Xwiener += wiener(X,[Mdate,Ntau])
			
		denoised_img = wiener(Xwiener,[Mdate,Ntau])
	elif img_to_denoise.ndim ==1:
		M = img_to_denoise.shape[0]
		NSV = np.min([M,NSV])
		denoised_img = wiener(img_to_denoise,Ntau)
		temp = np.trapz(np.abs(np.mean(denoised_img) - img_to_denoise))    
		denoised_img = wiener(img_to_denoise,Ntau,np.mean(temp))

	return denoised_img

if __name__ == "__main__":
    pass