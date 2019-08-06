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
import noise_module
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.filter import bandpass,lowpass
from obspy.core.inventory import Inventory, Network, Station, Channel, Site


'''
Core functions to be called by the main NoisePy scripts directly

by: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
    Marine Denolle (mdenolle@fas.harvard.edu)
'''

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

    by Chengxin Jiang
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
    st = noise_module.check_sample_gaps(st,date_info)
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
            st[0].data = noise_module.segment_interpolate(np.float32(st[0].data),float(fric/(delta*1E6)))
            #--reset the time to remove the discrepancy---
            st[0].stats.starttime-=(fric*1E-6)

    # remove traces of too small length
    #if st[0].stats.npts < 0.3*(date_info['endtime']-date_info['starttime'])*st[0].stats.sampling_rate:
    #    st = []
    #    return st

    # options to remove instrument response
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
            st = noise_module.resp_spectrum(st,specfile[0],samp_freq,pre_filt)

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

    # trim data
    ntr = obspy.Stream()
    # trim a continous segment into user-defined sequences
    st[0].trim(starttime=date_info['starttime'],endtime=date_info['endtime'],pad=True,fill_value=0)
    ntr.append(st[0])

    return ntr


def stats2inv(stats,resp=None,filexml=None,locs=None):

    #  Creates and inventory given the stats parameters in an obspy stream.
    # INPUT:
	
    inv = Inventory(networks=[],source="homegrown")

    if locs is None:
        net = Network(
            # This is the network code according to the SEED standard.
            code=stats.network,
            stations=[],
            description="Marine created from SAC and resp files",
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
            stations=[],
            description="Marine created from SAC and resp files",
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
    if resp is not None:
        print('i dont have the response')

    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)

    # And finally write it to a StationXML file. We also force a validation against
    # the StationXML schema to ensure it produces a valid StationXML file.
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


def optimized_cut_trace_make_statis(fc_para,source,flag):
    '''
    cut continous noise data into user-defined segments, estimate the statistics of 
    each segment and keep timestamp for later use.

    fft_para: dictionary containing all useful variables for the fft step.
    source: obspy stream of noise data.
    flag: boolen variable to output intermediate variables or not.
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

    # statistic to detect segments that may be associated with earthquakes
    all_madS = noise_module.mad(data)	            # median absolute deviation over all noise window
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
    dataS = noise_module.demean(dataS)
    dataS = noise_module.detrend(dataS)
    dataS = noise_module.taper(dataS)

    return trace_stdS,dataS_t,dataS


def noise_processing(fft_para,dataS,flag):
    '''
    perform time domain and frequency normalization according to user requirements. 
    Note that there are discussions in the litterature on noise cross correlation processing
    (REFs)
    This may not be necessary for coherency and deconvolution (Prieto et al, 2008, 2009; Denolle et al, 2013)

    # INPUT VARIABLES:
    fft_para: dictionary containing all useful variables used for fft
    dataS: data matrix containing all segmented noise data
    flag: boolen variable to output intermediate variables or not
    # OUTPUT VARIABLES:
    source_white: data matrix of processed Fourier spectra
    '''
    # load parameters first
    time_norm   = fft_para['time_norm']
    to_whiten   = fft_para['to_whiten']
    smooth_N    = fft_para['smooth_N']
    N = dataS.shape[0]

    #------to normalize in time or not------
    if time_norm:
        t0=time.time()   

        if time_norm == 'one_bit': 	# sign normalization
            white = np.sign(dataS)
        elif time_norm == 'running_mean': # running mean: normalization over smoothed absolute average           
            white = np.zeros(shape=dataS.shape,dtype=dataS.dtype)
            for kkk in range(N):
                white[kkk,:] = dataS[kkk,:]/noise_module.moving_ave(np.abs(dataS[kkk,:]),smooth_N)

        t1=time.time()
        if flag:
            print("temporal normalization takes %f s"%(t1-t0))
    else:	# don't normalize
        white = dataS

    #-----to whiten or not------
    if to_whiten:
        t0=time.time()
        source_white = noise_module.whiten(white,fft_para)	# whiten and return FFT
        t1=time.time()
        if flag:
            print("spectral whitening takes %f s"%(t1-t0))
    else:
        Nfft = int(next_fast_len(int(dataS.shape[1])))
        source_white = scipy.fftpack.fft(white, Nfft, axis=1) # return FFT
    
    return source_white


def smooth_source_spect(cc_para,fft1):
    '''
    Smoothes the amplitude spectrum of a 2D matrix of Fourier spectra.
    Used to speed up processing in correlation.

    input cc_para: dictionary containing useful cc parameters
          fft1: complex matrix containing source spectrum
    output sfft1: complex numpy array with normalized spectrum
    '''
    cc_method = cc_para['cc_method']
    smoothspect_N = cc_para['smoothspect_N']

    if cc_method == 'deconv':
        
        #-----normalize single-station cc to z component-----
        temp = noise_module.moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = np.conj(fft1)/temp**2
        except ValueError:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'coherency':
        temp = noise_module.moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = np.conj(fft1)/temp
        except ValueError:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'raw':
        sfft1 = np.conj(fft1)
    
    return sfft1

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
    substack= D['substack']                                                          
    substack_len  = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    nwin  = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size,)*fft2.reshape(fft2.size,)

    if method == "coherency":
        temp = noise_module.moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)             
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    #--------------- remove outliers in frequency domain -------------------
    # [reduce the number of IFFT by pre-selecting good windows before substack]
    freq = scipy.fftpack.fftfreq(Nfft, d=dt)[:Nfft2]
    i1 = np.where( (freq>=freqmin) & (freq <= freqmax))[0]

    # this creates the residuals between each window and their median
    med = np.log10(np.median(corr[:,i1],axis=0))
    r   = np.log10(corr[:,i1]) - med
    ik  = np.zeros(nwin,dtype=np.int)
    # find time window of good data
    for i in range(nwin):
        if np.any( (r[i,:]>=med-10) & (r[i,:]<=med+10) ):ik[i]=i
    ik1 = np.nonzero(ik)
    ik=ik[ik1]

    if substack:
        if substack_len == cc_len:
            # choose to keep all fft data for a day
            s_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)   # stacked correlation
            ampmax = np.zeros(nwin,dtype=np.float32)
            n_corr = np.zeros(nwin,dtype=np.int16)                  # number of correlations for each substack
            t_corr = dataS_t                                        # timestamp
            crap   = np.zeros(Nfft,dtype=np.complex64)
            for i in range(len(ik)): 
                n_corr[ik[i]]= 1           
                crap[:Nfft2] = corr[ik[i],:]
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[ik[i],:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

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
                itime = np.where( (dataS_t[ik] >= tstart) & (dataS_t[ik] < tstart+substack_len) )[0]  
                if len(ik[itime])==0:tstart+=substack_len;continue
                
                crap[:Nfft2] = np.mean(corr[ik[itime],:],axis=0)   # linear average of the correlation 
                crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2)+1:]=np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
                crap[0]=complex(0,0)
                s_corr[istack,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(ik[itime])           # number of windows stacks
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
        n_corr = len(ik)
        s_corr = np.zeros(Nfft,dtype=np.float32)
        t_corr = dataS_t[0]
        crap   = np.zeros(Nfft,dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr[ik,:],axis=0)
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
        'time':tcorr}
    return parameters


def load_pfiles(pfiles):
    '''
    read the dictionary containing all station-pair information for the cross-correlation data
    that is saved in ASDF format, and merge them into one sigle array for stacking purpose. 

    input pfiles: the file names containing all information
    output: an array of all station-pair information for the cross-correlations
    '''
    # get all unique path list
    paths_all = []
    for ii in range(len(pfiles)):
        with pyasdf.ASDFDataSet(pfiles[ii],mpi=False,mode='r') as pds:
            try:
                tpath = pds.auxiliary_data.list()
            except Exception:
                continue
        paths_all = list(set(paths_all+tpath))
    return paths_all
    

def do_stacking(cc_array,cc_time,cc_ngood,f_substack_len,stack_para):
    '''
    stacks the cross correlation data according to the interval of substack_len

    input variables:
    cc_array: 2D numpy float32 matrix containing all segmented cross-correlation data
    cc_time: 1D numpy array of all timestamp information for each segment of cc_array
    f_substack_len: length of time intervals for sub-stacking
    smethod: stacking method, chosen between linear and pws

    return variables:
    '''
    # do substacking and output them
    samp_freq = stack_para['samp_freq']
    smethod   = stack_para['stack_method']
    cc_len    = stack_para['cc_len']
    substack_len = stack_para['substack_len']

    npts = cc_array.shape[1]
    nwin = cc_array.shape[0]

    if cc_time[-1]<=cc_time[0]:
        s_corr=[];t_corr=[];n_corr=[]
        return s_corr,t_corr,n_corr

    # final substacking
    if f_substack_len:
        if f_substack_len==substack_len:
            s_corr = cc_array 
            n_corr = cc_ngood             
            t_corr = cc_time            

            # remove abnormal data
            ampmax = np.max(s_corr,axis=1)
            tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
            s_corr = s_corr[tindx,:]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = cc_time[-1]-cc_time[0]+cc_len            # total duration of what we have now
            tstart = cc_time[0]

            nstack = int(np.floor(Ttotal/f_substack_len))
            ampmax = np.zeros(nstack,dtype=np.float32)
            s_corr = np.zeros(shape=(nstack,npts),dtype=np.float32)
            n_corr = np.zeros(nstack,dtype=np.int)
            t_corr = np.zeros(nstack,dtype=np.float)                                            

            # remove abnormal data
            ampmax = np.zeros(nwin,dtype=np.float32)
            ampmax = np.max(cc_array,axis=1)
            indx = np.where((ampmax<20*np.median(ampmax)) & (ampmax>0) )[0]  

            for istack in range(nstack):                                                                   
                # find the indexes of all of the windows that start or end within 
                itime  = np.where( (cc_time[indx] >= tstart) & (cc_time[indx] < tstart+f_substack_len) )[0] 
                if not len(itime):tstart+=f_substack_len;continue
                ik = indx[itime]

                if smethod == 'linear':
                    s_corr[istack] = np.mean(cc_array[ik,:],axis=0)    # linear average of the correlation
                elif smethod == 'pws':
                    s_corr[istack] = noise_module.pws(cc_array[ik,:],samp_freq) 
                n_corr[istack] = np.sum(cc_ngood[ik])             # number of windows stacks
                t_corr[istack] = tstart               # save the time stamps
                tstart += f_substack_len
                #print('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            iindx = np.where(n_corr>0)[0]
            s_corr = s_corr[iindx,:]
            t_corr = t_corr[iindx]
            n_corr = n_corr[iindx]

    else:
        # do all averaging
        s_corr = np.zeros(npts,dtype=np.float32)
        n_corr = 1
        t_corr = cc_time[0]

        # remove abnormal data
        ampmax = np.zeros(cc_array.shape[0],dtype=np.float32)
        ampmax = np.max(cc_array,axis=1)
        indx = np.where((ampmax<20*np.median(ampmax)) & (ampmax>0) )[0]

        if smethod == 'linear':
            s_corr = np.mean(cc_array[indx],axis=0)
        elif smethod == 'pws':
            s_corr = noise_module.pws(cc_array[indx],samp_freq) 
        n_corr = np.sum(cc_ngood[indx])
    
    return s_corr,t_corr,n_corr

def do_rotation(sfile,stack_para,locs,flag):
    '''
    function to transfer from a E-N-Z coordinate into a R-T-Z system

    input variables:
    sfiles:     all stacked files in ASDF format
    stack_para: dict containing all parameters for stacking
    locs:       dict containing station angle info
    flag:       boolen variables to show intermeidate files
    '''
    # load useful variables
    sta_list = list(locs.iloc[:]['station'])
    angles   = list(locs.iloc[:]['angle'])
    correction = stack_para['correction']

    # get station info from the name of ASDF file
    staS  = sfile.split('/')[-1].split('_')[1].split('.')[1]
    staR  = sfile.split('/')[-1].split('_')[2].split('.')[1]
    ind   = sta_list.index(staS)
    acorr = angles[ind]
    ind   = sta_list.index(staR)
    bcorr = angles[ind]

    # define useful variables
    rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    pi = 3.1415926

    if flag:
        print('doing matrix rotation now!')

    # load useful parameters from asdf files 
    with pyasdf.ASDFDataSet(sfile,mpi=False) as st:
        dtypes = st.auxiliary_data.list()
        if not len(dtypes): print('no data in %s, return to main function'%sfile);return

        # loop through each time chunck
        for itype in dtypes:
            comp_list = st.auxiliary_data[itype].list()

            if len(comp_list) < 9:
                print('contine! seems no 9 components ccfs available');continue

            # load parameter dic
            parameters = st.auxiliary_data[itype][comp_list[0]].parameters
            azi = parameters['azi']
            baz = parameters['baz']
            data = st.auxiliary_data[itype][comp_list[0]].data[:]
            npts = data.shape[0]

            #---angles to be corrected----
            if correction:
                cosa = np.cos((azi+acorr)*pi/180)
                sina = np.sin((azi+acorr)*pi/180)
                cosb = np.cos((baz+bcorr)*pi/180)
                sinb = np.sin((baz+bcorr)*pi/180)
            else:
                cosa = np.cos(azi*pi/180)
                sina = np.sin(azi*pi/180)
                cosb = np.cos(baz*pi/180)
                sinb = np.sin(baz*pi/180)

            tcorr = np.zeros(shape=(9,npts),dtype=np.float32)
            for ii,icomp in enumerate(comp_list):
                tcorr[ii] = st.auxiliary_data[itype][icomp].data[:]

            #------9 component tensor rotation 1-by-1------
            for jj in range(len(rtz_components)):
                
                if jj==0:
                    crap = -cosb*tcorr[7]-sinb*tcorr[6]
                elif jj==1:
                    crap = sinb*tcorr[7]-cosb*tcorr[6]
                elif jj==2:
                    crap = tcorr[8]
                    continue
                elif jj==3:
                    crap = -cosa*cosb*tcorr[4]-cosa*sinb*tcorr[3]-sina*cosb*tcorr[1]-sina*sinb*tcorr[0]
                elif jj==4:
                    crap = cosa*sinb*tcorr[4]-cosa*cosb*tcorr[3]+sina*sinb*tcorr[1]-sina*cosb*tcorr[0]
                elif jj==5:
                    crap = cosa*tcorr[5]+sina*tcorr[2]
                elif jj==6:
                    crap = sina*cosb*tcorr[4]+sina*sinb*tcorr[3]-cosa*cosb*tcorr[1]-cosa*sinb*tcorr[0]
                elif jj==7:
                    crap = -sina*sinb*tcorr[4]+sina*cosb*tcorr[3]+cosa*sinb*tcorr[1]-cosa*cosb*tcorr[0]
                else:
                    crap = -sina*tcorr[5]+cosa*tcorr[2]

                #------save the time domain cross-correlation functions-----
                data_type = itype
                path = rtz_components[jj]
                st.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
