import os
import glob
import datetime
import copy
import matplotlib.pyplot as plt
from numba import jit
import pyasdf
import pandas as pd
import numpy as np
import scipy
from scipy.fftpack import fft,ifft,next_fast_len
from scipy.signal import butter, hilbert, wiener
from scipy.linalg import svd
from obspy.signal.filter import bandpass,lowpass
from obspy.signal.regression import linear_regression
from obspy.signal.invsim import cosine_taper
import obspy
from obspy.signal.util import _npts2nfft
from obspy.core.inventory import Inventory, Network, Station, Channel, Site


def stats2inv(stats,resp=None,filexml=None,locs=None):

    # We'll first create all the various objects. These strongly follow the
    # hierarchy of StationXML files.
    inv = Inventory(networks=[],source="japan_from_resp")

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

def preprocess_raw(st,inv,downsamp_freq,clean_time=True,pre_filt=None,resp=False,respdir=None):
    '''
    pre-process daily stream of data from IRIS server, including:

        - check sample rate is matching (from original process_raw)
        - remove small traces (from original process_raw)
        - remove trend and mean of each trace
        - interpolated to ensure all samples are at interger times of the sampling rate
        - low pass and downsample the data  (from original process_raw)
        - remove instrument response according to the option of resp_option. 
            "inv" -> using inventory information and obspy function of remove_response;
            "spectrum" -> use downloaded response spectrum and interpolate if necessary
            "polezeros" -> use the pole zeros for a crude correction of response
        - trim data to a day-long sequence and interpolate it to ensure starting at 00:00:00.000
    '''

    #----remove the ones with too many segments and gaps------
    if len(st) > 100 or portion_gaps(st) > 0.2:
        print('Too many traces or gaps in Stream: Continue!')
        st=[]
        return st

    #----check sampling rate and trace length----
    st = check_sample(st)
            
    if len(st) == 0:
        print('No traces in Stream: Continue!')
        return st

    sps = int(st[0].stats.sampling_rate)
    #-----remove mean and trend for each trace before merge------
    for ii in range(len(st)):
        if st[ii].stats.sampling_rate != sps:
            st[ii].stats.sampling_rate = sps
        
        #-----set nan values to zeros (it does happens!)-----
        tttindx = np.where(np.isnan(st[ii].data))
        if len(tttindx) >0:
            st[ii].data[tttindx]=0

        tttindx = np.where(np.isinf(st[ii].data))
        if len(tttindx) >0:
            st[ii].data[tttindx]=0
        st[ii].data = np.float32(st[ii].data)
        st[ii].data = scipy.signal.detrend(st[ii].data,type='constant')
        st[ii].data = scipy.signal.detrend(st[ii].data,type='linear')

    st.merge(method=1,fill_value=0)

    if abs(downsamp_freq-sps) > 1E-4:
        #-----low pass filter with corner frequency = 0.9*Nyquist frequency----
        #st[0].data = lowpass(st[0].data,freq=0.4*downsamp_freq,df=sps,corners=4,zerophase=True)
        st[0].data = bandpass(st[0].data,0.01,0.4*downsamp_freq,df=sps,corners=4,zerophase=True)

        #----make downsampling------
        st.interpolate(downsamp_freq,method='weighted_average_slopes')

        delta = st[0].stats.delta
        #-------when starttimes are between sampling points-------
        fric = st[0].stats.starttime.microsecond%(delta*1E6)
        if fric>1E-4:
            st[0].data = segment_interpolate(np.float32(st[0].data),float(fric/delta*1E6))
            #--reset the time to remove the discrepancy---
            st[0].stats.starttime-=(fric*1E-6)

    station = st[0].stats.station

    #-----check whether file folder exists-------
    if resp is not False:
        if resp != 'inv':
            if (respdir is None) or (not os.path.isdir(respdir)):
                raise ValueError('response file folder not found! abort!')

        if resp == 'inv':
            #----check whether inventory is attached----
            if not inv[0][0][0].response:
                raise ValueError('no response found in the inventory! abort!')
            else:
                print('removing response for %s using inv'%st[0])
                st[0].attach_response(inv)
                st[0].remove_response(output="VEL",pre_filt=pre_filt,water_level=60)

        elif resp == 'spectrum':
            print('remove response using spectrum')
            specfile = glob.glob(os.path.join(respdir,'*'+station+'*'))
            if len(specfile)==0:
                raise ValueError('no response sepctrum found for %s' % station)
            st = resp_spectrum(st,specfile[0],downsamp_freq,pre_filt)

        elif resp == 'RESP_files':
            print('using RESP files')
            seedresp = glob.glob(os.path.join(respdir,'RESP.'+station+'*'))
            if len(seedresp)==0:
                raise ValueError('no RESP files found for %s' % station)
            st.simulate(paz_remove=None,pre_filt=pre_filt,seedresp=seedresp[0])

        elif resp == 'polozeros':
            print('using polos and zeros')
            paz_sts = glob.glob(os.path.join(respdir,'*'+station+'*'))
            if len(paz_sts)==0:
                raise ValueError('no polozeros found for %s' % station)
            st.simulate(paz_remove=paz_sts[0],pre_filt=pre_filt)

        else:
            raise ValueError('no such option of resp in preprocess_raw! please double check!')

    #-----fill gaps, trim data and interpolate to ensure all starts at 00:00:00.0------
    if clean_time:
        st = clean_daily_segments(st)

    return st

def portion_gaps(stream):
    '''
    get the accumulated gaps (npts) by looking at the accumulated difference between starttime and endtime,
    instead of using the get_gaps function of obspy object of stream. remove the trace if gap length is 
    more than 30% of the trace size. remove the ones with sampling rate not consistent with max(freq) 
    '''
    #-----check the consistency of sampling rate----

    pgaps=0

    if len(stream)==0:
        return pgaps
        
    else:
        npts = (stream[-1].stats.endtime-stream[0].stats.starttime)*stream[0].stats.sampling_rate
        #----loop through all trace to find gaps----
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

def clean_daily_segments(tr):
    '''
    subfunction to clean the tr recordings. only the traces with at least 0.5-day long
    sequence (respect to 00:00:00.0 of the day) is kept. note that the trace here could
    be of several days recordings, so this function helps to break continuous chunck 
    into a day-long segment from 00:00:00.0 to 24:00:00.0.

    tr: obspy stream object
    '''
    #-----all potential-useful time information-----
    stream_time = tr[0].stats.starttime
    time0 = obspy.UTCDateTime(stream_time.year,stream_time.month,stream_time.day,0,0,0)
    time1 = obspy.UTCDateTime(stream_time.year,stream_time.month,stream_time.day,12,0,0)
    time2 = time1+datetime.timedelta(hours=12)

    #--only keep days with > 0.5-day recordings-
    if stream_time <= time1:
        starttime=time0
    else:
        starttime=time2

    #-----ndays represents how many days from starttime to endtime----
    ndays = round((tr[0].stats.endtime-starttime)/(time2-time0))
    if ndays==0:
        tr=[]
        return tr

    else:
        #-----make a new stream------
        ntr = obspy.Stream()
        ttr = tr[0].copy()
        #----trim a continous segment into day-long sequences----
        for ii in range(ndays):    
            tr[0] = ttr.copy()
            endtime = starttime+datetime.timedelta(days=1)
            tr[0].trim(starttime=starttime,endtime=endtime,pad=True,fill_value=0)

            ntr.append(tr[0])
            starttime = endtime

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


def get_event_list(str1,str2,inc_days):
    '''
    return the event list in the formate of 2010_01_01 by taking
    advantage of the datetime modules
    
    str1: string of starting date -> 2010_01_01
    str2: string of ending date -> 2010_10_11
    '''
    event = []
    date1=str1.split('_')
    date2=str2.split('_')
    y1=int(date1[0])
    m1=int(date1[1])
    d1=int(date1[2])
    y2=int(date2[0])
    m2=int(date2[1])
    d2=int(date2[2])
    
    d1=datetime.datetime(y1,m1,d1)
    d2=datetime.datetime(y2,m2,d2)
    dt=datetime.timedelta(days=inc_days)

    while(d1<d2):
        event.append(d1.strftime('%Y_%m_%d'))
        d1+=dt
    event.append(d2.strftime('%Y_%m_%d'))
    
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

def whiten(data, delta, freqmin, freqmax,Nfft=None):
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

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    if Nfft is None:
        Nfft = next_fast_len(int(data.shape[axis]))

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
        FFTRawSign[:,left:right] = np.exp(1j * np.angle(FFTRawSign[:,left:right]))
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
        FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
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
    
def optimized_cc_parameters(dt,maxlag,method,nhours,lonS,latS,lonR,latR):
    '''
    provide the parameters for computting CC later
    '''
    #dist = get_distance(lonS,latS,lonR,latR)
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
        'ngood':nhours,
        'method':method}
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

@jit('float32[:](float32[:],int16)')
def moving_ave(A,N):
    '''
    Numba compiled function to do running smooth average.
    N is the the half window length to smooth
    A and B are both 1-D arrays (which runs faster compared to 2-D operations)
    '''
    A = np.r_[A[:N],A,A[-N:]]
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

def whiten_smooth(data,dt,freqmin,freqmax,smooth_N):
    '''
    subfunction to make a moving window average on the ampliutde spectrum
    with a tapering on the lower and higher end of the frequency range

    '''
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    #--------frequency information----------
    Nfft = int(next_fast_len(int(data.shape[axis])))
    freqVec = scipy.fftpack.fftfreq(Nfft, d=dt)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]

    #------four frequency parameters-------
    Napod = 100
    left  = J[0]
    right = J[-1]
    low = J[0] - Napod
    if low < 0:
        low = 0
    high = J[-1] + Napod
    if high > Nfft/2:
        high = int(Nfft//2)

    spect = scipy.fftpack.fft(data,Nfft,axis=axis)

    #-----smooth the spectrum and do tapering-----
    if axis:
        spect[:,0:low] *= 0
        #----left tapering---
        spect[:,low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(spect[:,low:left]))
        #-----Pass band-------
        for ii in range(data.shape[0]):
            tave = moving_ave(np.abs(spect[ii,left:right]),smooth_N)
            spect[ii,left:right] = spect[ii,left:right]/tave
        #----right tapering------
        spect[:,right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(spect[:,right:high]))
        spect[:,high:Nfft//2+1] *= 0

        spect[:,-(Nfft//2)+1:] = np.flip(np.conj(spect[:,1:(Nfft//2)]),axis=axis)
    else:
        spect[0:low] *= 0
        #----left tapering---
        spect[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(spect[low:left]))
        #-----Pass band-------
        tave = moving_ave(np.abs(spect[left:right]),smooth_N)
        spect[left:right] = spect[left:right]/tave
        #----right tapering------
        spect[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(spect[right:high]))
        spect[high:Nfft//2] *= 0

        spect[-(Nfft//2)+1:] = np.flip(np.conj(spect[1:(Nfft//2)]),axis=axis)    

    return spect   


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


def pole_zero(inv): 
    """

    Return only pole and zeros of instrument response

    """
    for ii,chan in enumerate(inv[0][0]):
        stages = chan.response.response_stages
        new_stages = []
        for stage in stages:
            if type(stage) == obspy.core.inventory.response.PolesZerosResponseStage:
                new_stages.append(stage)
            elif type(stage) == obspy.core.inventory.response.CoefficientsTypeResponseStage:
                new_stages.append(stage)

        inv[0][0][ii].response.response_stages = new_stages

    return inv

def check_and_phase_shift(trace):
    # print trace
    taper_length = 20.0
    # if trace.stats.npts < 4 * taper_length*trace.stats.sampling_rate:
    # 	trace.data = np.zeros(trace.stats.npts)
    # 	return trace

    dt = np.mod(trace.stats.starttime.datetime.microsecond*1.0e-6,
                trace.stats.delta)
    if (trace.stats.delta - dt) <= np.finfo(float).eps:
        dt = 0
    if dt != 0:
        if dt <= (trace.stats.delta / 2.):
            dt = -dt
        # direction = "left"
        else:
            dt = (trace.stats.delta - dt)
        # direction = "right"
        trace.detrend(type="demean")
        trace.detrend(type="simple")
        taper_1s = taper_length * float(trace.stats.sampling_rate) / trace.stats.npts
        trace.taper(taper_1s)

        n = int(2**nextpow2(len(trace.data)))
        FFTdata = scipy.fftpack.fft(trace.data, n=n)
        fftfreq = scipy.fftpack.fftfreq(n, d=trace.stats.delta)
        FFTdata = FFTdata * np.exp(1j * 2. * np.pi * fftfreq * dt)
        trace.data = np.real(scipy.fftpack.ifft(FFTdata, n=n)[:len(trace.data)])
        trace.stats.starttime += dt
        return trace
    else:
        return trace

def check_sample(stream):
    """
    Returns sampling rate of traces in stream.

    :type stream:`~obspy.core.stream.Stream` object. 
    :param stream: Stream containing one or more day-long trace 
    :return: List of sampling rates in stream

    """
    if len(stream)==0:
        return stream
    else:
        freqs = []	
        for tr in stream:
            freqs.append(int(tr.stats.sampling_rate))

    freq = max(freqs)
    for tr in stream:
        if int(tr.stats.sampling_rate) != freq:
            stream.remove(tr)

    return stream				


def remove_resp(arr,stats,inv):
    """
    Removes instrument response of cross-correlation

    :type arr: numpy.ndarray 
    :type stats: `~obspy.core.trace.Stats` object.
    :type inv: `~obspy.core.inventory.inventory.Inventory`
    :param inv: StationXML file containing response information
    :returns: cross-correlation with response removed
    """	
    
    def arr_to_trace(arr,stats):
        tr = obspy.Trace(arr)
        tr.stats = stats
        tr.stats.npts = len(tr.data)
        return tr

    # prefilter and remove response
    
    st = obspy.Stream()
    if len(arr.shape) == 2:
        for row in arr:
            tr = arr_to_trace(row,stats)
            st += tr
    else:
        tr = arr_to_trace(arr,stats)
        st += tr
    min_freq = 1/tr.stats.npts*tr.stats.sampling_rate
    min_freq = np.max([min_freq,0.005])
    pre_filt = [min_freq,min_freq*1.5, 0.9*tr.stats.sampling_rate, 0.95*tr.stats.sampling_rate]
    st.attach_response(inv)
    st.remove_response(output="VEL",pre_filt=pre_filt) 

    if len(st) > 1: 
        data = []
        for tr in st:
            data.append(tr.data)
        data = np.array(data)
    else:
        data = st[0].data
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


def fft_parameters(dt,cc_len,source_params,locs,component,Nfft,Nt):
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
    source_mad,source_std,source_nonzero = source_params[:,0],\
                         source_params[:,1],source_params[:,2]
    lon,lat,el=locs["longitude"],locs["latitude"],locs["elevation"]
    parameters = {
             'twin':cc_len,
             'mad':source_mad,
             'std':source_std,
             'nonzero':source_nonzero,
             'longitude':lon,
             'latitude':lat,
             'elevation_in_m':el,
             'component':component,
             'nfft':Nfft,
             'nseg':Nt}
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

def Stretching_current(ref, cur, t, dv_range, nbtrial, window, fmin, fmax, tmin, tmax):
    """
    Stretching function: 
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.

    modified based on the script from L. Viens 04/26/2018 (Viens et al., 2018 JGR)

    INPUTS:
        - ref = Reference waveform (np.ndarray, size N)
        - cur = Current waveform (np.ndarray, size N)
        - t = time vector, common to both ref and cur (np.ndarray, size N)
        - dvmin = minimum bound for the velocity variation; example: dvmin=-0.03 for -3% of relative velocity change ('float')
        - dvmax = maximum bound for the velocity variation; example: dvmax=0.03 for 3% of relative velocity change ('float')
        - nbtrial = number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
        - window = vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
        For error computation:
            - fmin = minimum frequency of the data
            - fmax = maximum frequency of the data
            - tmin = minimum time window where the dv/v is computed 
            - tmax = maximum time window where the dv/v is computed 

    OUTPUTS:
        - dv = Relative velocity change dv/v (in %)
        - cc = correlation coefficient between the reference waveform and the best stretched/compressed current waveform
        - cdp = correlation coefficient between the reference waveform and the initial current waveform
        - error = Errors in the dv/v measurements based on Weaver, R., C. Hadziioannou, E. Larose, and M. Camnpillo (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3), 1384?1392

    The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values. 
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .
    """ 
    dvmin = -np.abs(dv_range)
    dvmax = np.abs(dv_range)
    Eps = 1+(np.linspace(dvmin, dvmax, nbtrial))
    cof = np.zeros(Eps.shape,dtype=np.float32)

    # Set of stretched/compressed current waveforms
    for ii in range(len(Eps)):
        nt = t*Eps[ii]
        s = np.interp(x=t, xp=nt, fp=cur[window])
        waveform_ref = ref[window]
        waveform_cur = s
        cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cdp = np.corrcoef(cur[window], ref[window])[0, 1] # correlation coefficient between the reference and initial current waveforms

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
        nt = t*dtfiner[ii]
        s = np.interp(x=t, xp=nt, fp=cur[window])
        waveform_ref = ref[window]
        waveform_cur = s
        ncof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    cc = np.max(ncof) # Find maximum correlation coefficient of the refined  analysis
    dv = 100. * dtfiner[np.argmax(ncof)]-100 # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

    # Error computation based on Weaver, R., C. Hadziioannou, E. Larose, and M. Camnpillo (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3), 1384?1392
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, cc, cdp, error


def smooth(x, window='boxcar', half_win=3):
    """ some window smoothing """
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


def getCoherence(dcs, ds1, ds2):
    # TODO: docsting
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2) > 0))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh

def mwcs_dvv(ref, cur, moving_window_length, slide_step, delta, window, fmin, fmax, tmin, smoothing_half_win=5):
    """
    modified sub-function from MSNoise package by Thomas Lecocq. download from
    https://github.com/ROBelgium/MSNoise/tree/master/msnoise

    combine the mwcs and dv/v functionality of MSNoise into a single function

    ref: The "Reference" timeseries
    cur: The "Current" timeseries
    moving_window_length: The moving window length (in seconds)
    slide_step: The step to jump for the moving window (in seconds)
    delta: The sampling rate of the input timeseries (in Hz)
    window: The target window for measuring dt/t
    fmin: The lower frequency bound to compute the dephasing (in Hz)
    fmax: The higher frequency bound to compute the dephasing (in Hz)
    tmin: The leftmost time lag (used to compute the "time lags array")
    smoothing_half_win: If different from 0, defines the half length of
        the smoothing hanning window.
    :returns: [time_axis,delta_t,delta_err,delta_mcoh]. time_axis contains the
        central times of the windows. The three other columns contain dt, error and
        mean coherence for each window.
    """
    
    ##########################
    #-----part I: mwcs-------
    ##########################
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    window_length_samples = np.int(moving_window_length * delta)
    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    count = 0
    tp = cosine_taper(window_length_samples, 0.85)

    #----does minind really start from 0??-----
    minind = 0
    maxind = window_length_samples

    #-------loop through all sub-windows-------
    while maxind <= len(window):
        cci = cur[window[minind:maxind]]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[window[minind:maxind]]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step*delta)
        maxind += int(slide_step*delta)

        #-------------get the spectrum-------------
        fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
        fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # Calculate the cross-spectrum
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
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, 1. / delta)[:padd // 2]
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
        # forced through the origin
        # weights for the WLS must be the variance !
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

    if maxind > len(cur) + slide_step*delta:
        print("The last window was too small, but was computed")

    delta_t = np.array(delta_t)
    delta_err = np.array(delta_err)
    delta_mcoh = np.array(delta_mcoh)
    time_axis  = np.array(time_axis)

    #####################################
    #-----------part II: dv/v------------
    #####################################
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
        m0, em0 = linear_regression(time_axis[indx], delta_t[indx], w,intercept_origin=True)
    
    else:
        print('not enough points to estimate dv/v')
        m0=0;em0=0

    return np.array([-m0*100,em0*100]).T


def wavg_wstd(data, errors):
    '''
    estimate the weights for doing linear regression in order to get dt/t
    '''

    d = data
    errors[errors == 0] = 1e-6
    w = 1. / errors
    wavg = (d * w).sum() / w.sum()
    N = len(np.nonzero(w)[0])
    wstd = np.sqrt(np.sum(w * (d - wavg) ** 2) / ((N - 1) * np.sum(w) / N))
    return wavg, wstd


def mwcs_dvv1(ref, cur, moving_window_length, slide_step, delta, window, fmin, fmax, tmin, smoothing_half_win=5):
    """
    modified sub-function from MSNoise package by Thomas Lecocq. download from
    https://github.com/ROBelgium/MSNoise/tree/master/msnoise

    combine the mwcs and dv/v functionality of MSNoise into a single function

    ref: The "Reference" timeseries
    cur: The "Current" timeseries
    moving_window_length: The moving window length (in seconds)
    slide_step: The step to jump for the moving window (in seconds)
    delta: The sampling rate of the input timeseries (in Hz)
    window: The target window for measuring dt/t
    fmin: The lower frequency bound to compute the dephasing (in Hz)
    fmax: The higher frequency bound to compute the dephasing (in Hz)
    tmin: The leftmost time lag (used to compute the "time lags array")
    smoothing_half_win: If different from 0, defines the half length of
        the smoothing hanning window.
    :returns: [time_axis,delta_t,delta_err,delta_mcoh]. time_axis contains the
        central times of the windows. The three other columns contain dt, error and
        mean coherence for each window.
    """
    
    ##################################################################
    #-------------------------part I: mwcs----------------------------
    ##################################################################
    delta_t    = []
    delta_err  = []
    delta_mcoh = []
    time_axis  = []

    window_length_samples = np.int(moving_window_length * delta)
    padd  = int(2 ** (nextpow2(window_length_samples) + 2))
    tp = cosine_taper(window_length_samples, 0.85)

    #----make usage of both lags----
    flip_flag = [0,1]
    for iflip in flip_flag:

        #----indices of the moving window-----
        count  = 0
        minind = 0
        maxind = window_length_samples

        if iflip:
            cur = np.flip(cur,axis=0)
            ref = np.flip(ref,axis=0)

        #-------loop through all sub-windows-------
        while maxind <= len(window):
            cci = cur[window[minind:maxind]]
            cci = scipy.signal.detrend(cci, type='linear')
            cci *= tp

            cri = ref[window[minind:maxind]]
            cri = scipy.signal.detrend(cri, type='linear')
            cri *= tp

            minind += int(slide_step*delta)
            maxind += int(slide_step*delta)

            #-------------get the spectrum-------------
            fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
            fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

            fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
            fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

            # Calculate the cross-spectrum
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
            freq_vec = scipy.fftpack.fftfreq(len(X) * 2, 1. / delta)[:padd // 2]
            index_range = np.argwhere(np.logical_and(freq_vec >= fmin,freq_vec <= fmax))

            # Get Coherence and its mean value
            coh  = getCoherence(dcs, dref, dcur)
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
            # forced through the origin
            # weights for the WLS must be the variance !
            m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())

            delta_t.append(m)

            # print phi.shape, v.shape, w.shape
            e    = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
            s2x2 = np.sum(v ** 2 * w ** 2)
            sx2  = np.sum(w * v ** 2)
            e    = np.sqrt(e * s2x2 / sx2 ** 2)

            delta_err.append(e)
            delta_mcoh.append(np.real(mcoh))
            if iflip:
                time_axis.append((tmin+moving_window_length/2.+count*slide_step)*-1)
            else:
                time_axis.append(tmin+moving_window_length/2.+count*slide_step)
            count += 1

            del fcur, fref
            del X
            del freq_vec
            del index_range
            del w, v, e, s2x2, sx2, m, em

        if maxind > len(cur) + slide_step*delta:
            print("The last window was too small, but was computed")

    delta_t    = np.array(delta_t)
    delta_err  = np.array(delta_err)
    delta_mcoh = np.array(delta_mcoh)
    time_axis  = np.array(time_axis)

    tt = np.arange(0,len(ref))*delta
    plt.subplot(211)
    plt.scatter(time_axis,delta_t)
    plt.subplot(212)
    plt.plot(tt,ref,'r-');plt.plot(tt,cur,'g-')
    plt.show()

    ##################################################################
    #--------------------------part II: dv/v--------------------------
    ##################################################################

    #-----some default parameters used in MSNoise-------
    delta_mincho = 0.65
    delta_maxerr = 0.1
    delta_maxdt  = 0.1
    indx1 = np.where(delta_mcoh>delta_mincho)
    indx2 = np.where(delta_err<delta_maxerr)
    indx3 = np.where(delta_t<delta_maxdt)

    #-----find good dt measurements-----
    indx = np.intersect1d(indx1,indx2)
    indx = np.intersect1d(indx,indx3)

    #-----at least 3 points for the linear regression------
    if len(indx) >2:

        #----estimate weight for regression----
        w = 1/delta_err[indx]
        w[~np.isfinite(w)] = 1.0

        #---------do linear regression-----------
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis[indx], delta_t[indx], w,intercept_origin=True)
    
    else:
        print('not enough points to estimate dv/v')
        m0=np.nan;em0=np.nan

    return np.array([-m0*100,em0*100]).T

def computeErrorFunction(u1, u0, nSample, lag, norm='L2'):
    """
    USAGE: err = computeErrorFunction( u1, u0, nSample, lag )
    
    INPUT:
        u1      = trace that we want to warp; size = (nsamp,1)
        u0      = reference trace to compare with: size = (nsamp,1)
        nSample = numer of points to compare in the traces
        lag     = maximum lag in sample number to search
        norm    = 'L2' or 'L1' (default is 'L2')
    OUTPUT:
        err = the 2D error function; size = (nsamp,2*lag+1)
    
    The error function is equation 1 in Hale, 2013. You could uncomment the
    L1 norm and comment the L2 norm if you want on Line 29
    
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
    USAGE: d = accumulation_diw_mod( dir, err, nSample, lag, b )

    INPUT:
        dir = accumulation direction ( dir > 0 = forward in time, dir <= 0 = backward in time)
        err = the 2D error function; size = (nsamp,2*lag+1)
        nSample = numer of points to compare in the traces
        lag = maximum lag in sample number to search
        b = strain limit (integer value >= 1)
    OUTPUT:
        d = the 2D distance function; size = (nsamp,2*lag+1)
    
    The function is equation 6 in Hale, 2013.

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
    USAGE: stbar = backtrackDistanceFunction( dir, d, err, lmin, b )

    INPUT:
        dir   = side to start minimization ( dir > 0 = front, dir <= 0 =  back)
        d     = the 2D distance function; size = (nsamp,2*lag+1)
        err   = the 2D error function; size = (nsamp,2*lag+1)
        lmin  = minimum lag to search over
        b     = strain limit (integer value >= 1)
    OUTPUT:
        stbar = vector of integer shifts subject to |u(i)-u(i-1)| <= 1/b

    The function is equation 2 in Hale, 2013.

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


def computeDTWerror( Aerr, u, lag0):
    """

    Compute the accumulated error along the warping path for Dynamic Time Warping.

    USAGE: function error = computeDTWerror( Aerr, u, lag0 )

    INPUT:
        Aerr = error MATRIX (equation 13 in Hale, 2013)
        u    = warping function (samples) VECTOR
        lag0 = value of maximum lag (samples) SCALAR

    Written by Dylan Mikesell
    Last modified: 25 February 2015
    Translated to python by Tim Clements (17 Aug. 2018)
    """

    npts = len(u)

    if Aerr.shape[0] != npts:
        print('Funny things with dimensions of error matrix: check inputs.')
        Aerr = Aerr.T

    error = 0
    for ii in range(npts):
        idx = lag0 + 1 + u[ii] # index of lag 
        error = error + Aerr[ii,idx]

    return error 

if __name__ == "__main__":
    pass

