import os
import scipy
import obspy
import pycwt
import pyasdf
import numpy as np
import monitor_modules
from obspy.signal.filter import bandpass
from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression

'''
a compilation of all available core functions for computing phase delays on ambient noise interferometry

quick index of the methods:
1) stretching (time stretching method: ref)
2) mwcs_dvv (Moving Window Cross Spectrum method: ref)
3) dtw_dvv (Dynamic Time Warping method: ref)
'''
def load_waveforms(sfile,para):
    '''
    functions to load targeted cross-correlation functions (CCFs) from ASDF file; trim and filter the CCFs
    and return for later dv-v measurements

    PARAMETERS:
    ----------------
    sfile: ASDF file for one-station pair with stacked and substacked CCFs
    para: dictionary containing all useful variables to window data

    RETURNS:
    ----------------
    ref: reference waveform
    cur: array containing all current waveforms 
    '''
    # load useful variables
    twin = para['twin']
    comp = para['ccomp']
    fband= para['fband']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(fband)
    fmax = np.max(fband)
    norm_flag = para['norm_flag']

    with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
        slist = ds.auxiliary_data.list()

        # some common variables
        try:
            delta = ds.auxiliary_data[slist[0]][comp].parameters['dt']
            maxlag= ds.auxiliary_data[slist[0]][comp].parameters['maxlag']
        except Exception as e:
            raise ValueError('cannot open %s to read'%sfile)

        if delta != dt:


        # time axis
        window = np.arange(int(tmin/delta),int(tmax/delta))+int(maxlag/delta)
        tvec = np.arange(-maxlag,maxlag+1)*delta
        npts = tvec.size
        nstacks = len(slist)

        # prepare data matrix for later loading 
        data  = np.zeros((nstacks,npts),dtype=np.float32)
        flag  = np.zeros(nstacks,dtype=np.int16)

        # loop through each stacked segment
        for ii,dtype in enumerate(slist):
            try:
                tdata = ds.auxiliary_data[dtype][comp].data[:]
                flag[ii] = 1
            except Exception:
                continue
            
            data[ii] = np.float32(bandpass(tdata,fmin,fmax,int(1/delta),corners=4, zerophase=True))
            data[ii] /= max(data[ii])           # whether to normalize waveform or not

        indx = np.where(flag==1)
        data = data[indx]
        del flag
    
    ref = data[0]
    data = data[1:]
    return ref,data,para

def stretching(ref, cur, dv_range, nbtrial, para):
    
    """
    Stretching function: 
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.

    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    dvmin: minimum bound for the velocity variation; example: dvmin=-0.03 for -3% of relative velocity change ('float')
    dvmax: maximum bound for the velocity variation; example: dvmax=0.03 for 3% of relative velocity change ('float')
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
    fband= para['fband']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(fband)
    fmax = np.max(fband)
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
        s = np.interp(x=t, xp=nt, fp=cur)
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

    return dv, cc, cdp, error


def mwcs_dvv(ref, cur, moving_window_length, slide_step, delta, window, fmin, fmax, tmin, smoothing_half_win=5):
    """
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
    
    Originally from MSNoise package by Thomas Lecocq. download from https://github.com/ROBelgium/MSNoise/tree/master/msnoise

    Modified by Chengxin Jiang
    """
    
    ##########################
    #-----part I: mwcs-------
    ##########################
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    window_length_samples = np.int(moving_window_length * delta)
    padd = int(2 ** (monitor_modules.nextpow2(window_length_samples) + 2))
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
            dcur = np.sqrt(monitor_modules.smooth(fcur2, window='hanning',half_win=smoothing_half_win))
            dref = np.sqrt(monitor_modules.smooth(fref2, window='hanning',half_win=smoothing_half_win))
            X = monitor_modules.smooth(X, window='hanning',half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, 1. / delta)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= fmin,freq_vec <= fmax))

        # Get Coherence and its mean value
        coh = monitor_module.getCoherence(dcs, dref, dcur)
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


def WCC_dvv(ref, cur, moving_window_length, slide_step, delta, window, tmin):
    """
    Windowed cross correlation (WCC) for dt or dv/v mesurement (Snieder et al. 2012)

    TO DO:
    compare this with MWCS method
    
    Parameters:
    -----------
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
        
    Written by Congcong Yuan (1 July, 2019)
    """ 
    ##########################
    #-----part I: wcc-------
    ##########################
    delta_t = []
    delta_t_coef = []
    time_axis = []

    window_length_samples = np.int(moving_window_length * delta)
#    padd = int(2 ** (nextpow2(window_length_samples) + 2))
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

        #-------- normalize signals before cross correlation -----------
        cci = (cci - cci.mean()) / cci.std()
        cri = (cri - cri.mean()) / cri.std()
        
        #-------- get maximum correlation coefficient and its index --------
        cc2 = np.correlate(cci, cri, mode='same')
        cc2 = cc2 / np.sqrt((cci**2).sum() * (cri**2).sum())
            
        imaxcc2 = np.where(cc2==np.max(cc2))[0]
        maxcc2 = np.max(cc2)
        
        #-------- get the time shift -------------
        m = (imaxcc2-((maxind-minind)//2))/delta
        delta_t.append(m)
        delta_t_coef.append(maxcc2)
       
        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

    del cci, cri, cc2, imaxcc2, maxcc2
    del m

    if maxind > len(cur) + slide_step*delta:
        print("The last window was too small, but was computed")

    delta_t = np.array(delta_t)
    delta_t_coef =  np.array(delta_t_coef)
    time_axis  = np.array(time_axis)

    #####################################
    #-----------part II: dv/v------------
    #####################################

    if count >2:
        
        #----estimate weight for regression----
        w = np.ones(count)

        #---------do linear regression-----------
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis.flatten(), delta_t.flatten(), w.flatten(),intercept_origin=True)
    
    else:
        print('not enough points to estimate dv/v')
        m0=0;em0=0

    return np.array([-m0*100,em0*100]).T  


def dtw_dvv(cur, ref, t, window, maxLag=50, b=5, direction=1):
    """
    Dynamic time warping to be used for dv/v estimation.
    
    Parameters:
        :param cur : current signal, (np.array, size N)
        :param ref : reference signal, (np.array, size N)
        :param t : time vector, common to both ref and cur (np.ndarray, size N)
        :param window : vector of the indices of the cur and ref windows 
                        on which you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
        :param maxLag : max number of points to search forward and backward. 
                        Suggest setting it larger if window is set larger.
        :param b : b-value to limit strain, which is to limit the maximum velocity perturbation. 
                   See equation 11 in (Mikesell et al. 2015)
    Return:
        :-m0 : estimated dv/v
        :em0 : error of dv/v estimation
        
    """
    
    # setup other parameters
    dt = np.mean(np.diff(t)) # sampling rate
    npts = len(window) # number of time samples
    
    tvect = t[window]
    # ============================= part 1: get time shifts ================================
    # compute error function
    # compute error function over lags, which is independent of strain limit 'b'.
    err = monitor_modules.computeErrorFunction( cur[window], ref[window], npts, maxLag ) 
    
    # direction to accumulate errors (1=forward, -1=backward)
    # it is instructive to flip the sign of +/-1 here to see how the function
    # changes as we start the backtracking on different sides of the traces.
    # Also change 'b' to see how this influences the solution for stbar. You
    # want to make sure you're doing things in the proper directions in each
    # step!!!
    dist  = monitor_modules.accumulateErrorFunction( direction, err, npts, maxLag, b )
    stbar = monitor_modules.backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )
    
    stbarTime = stbar * dt   # convert from samples to time
    #tvect2     = tvect + stbarTime # make the warped time axis
    
    # ============================ part 2: get dvv ==========================================
    if npts >2:

        #----estimate weight for regression----
        w = np.ones(npts)
    
        #---------do linear regression-----------
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(tvect.flatten(), stbarTime.flatten(), w.flatten(),intercept_origin=True)

    else:
        print('not enough points to estimate dv/v')
        m0=0;em0=0
    
    return np.array([-m0*100,em0*100]).T


def wxs_allfreq(cur,ref,t,twindow,fwindow, dj=1/12, s0=-1, J=-1, sig=False, wvn='morlet',unwrapflag=False,nwindow=1.5,windowflag=True):
    """
    Compute dt or dv/v in time and frequency domain from wavelet cross spectrum (wxs).
    for all frequecies in an interest range
    
    Parameters
    --------------
    :type cur: :class:`~numpy.ndarray`
    :param cur: 1d array. Cross-correlation measurements.
    :type ref: :class:`~numpy.ndarray`
    :param ref: 1d array. The reference trace.
    :type t: :class:`~numpy.ndarray`
    :param t: 1d array. Cross-correlation measurements.
    :param twindow: 1d array. [earlist time, latest time] time window limit
    :param fwindow: 1d array. [lowest frequncy, highest frequency] frequency window limit
    :params, dj, s0, J, sig, wvn, refer to function 'wavelet.wct'
    :unwrapflag: True - unwrap phase delays. Default is False
    :nwindow: the times of current period/frequency, which will be time window if windowflag is False 
    :windowflag: if True, the given window 'twindow' will be used, 
                 otherwise, the current period*nwindow will be used as time window
    
    Originally written by Tim Clements (1 March, 2019)
    Modified by Congcong Yuan (30 June, 2019) based on (Mao et al. 2019).
    """
    
    ##################################################################
    #------------------part I: cross spectrum analysis----------------
    ##################################################################
    
    fs = 1/np.mean(np.diff(t))
    fmin, fmax = fwindow[0], fwindow[1]
    
    # perform cross coherent analysis, modified from function 'wavelet.cwt'
    WCT, aWCT, coi, freq, sig = pycwt.wct(cur, ref, 1/fs, dj=dj, s0=s0, J=J, sig=sig, wavelet=wvn, normalize=True)
    
    if unwrapflag:
        phase = np.unwrap(aWCT,axis=-1) # axis=0, upwrap along time; axis=-1, unwrap along frequency
    else:
        phase=aWCT
    
    # convert phase delay to time delay
    delta_t = phase / (2*np.pi*freq[:,None]) # normalize phase by (2*pi*frequency) 

    # zero out data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        print('Error: please input right frequency limits in the frequenct window!')
    else:
        freq_indout = np.where((freq < fmin) | (freq > fmax))[0]
        delta_t[freq_indout, :] = 0.
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]
        
    
    ##################################################################
    #--------------------------part II: dv/v--------------------------
    ##################################################################
    dvv, err = np.zeros(freq.shape), np.zeros(freq.shape)
    if windowflag: #using fixed time window
        tmin, tmax = twindow[0], twindow[1]  
    
        if (tmin < np.min(t)) | (tmax> np.max(t)) | (tmax <= tmin):
            print('Error: please input right time limits in the time window!')
        else:
            # truncate data with the time window
            time_ind = np.where((t >= tmin) & (t < tmax))[0]
            wt = t[time_ind]
            
        #-----at least 3 points for the linear regression------
        for ifreq in freq_indin:
            if len(wt)>2:
                if not np.any(delta_t[ifreq,time_ind]):
                    continue
                #---- use WXA as weight for regression----
                # w = 1.0 / (1.0 / (WCT[ifreq,:] ** 2) - 1.0)
                # w[WCT[ifreq,time_ind] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
                # w = np.sqrt(w * np.sqrt(WXA[ifreq,time_ind]))
                # w = np.real(w)
                w = 1/WCT[ifreq,time_ind]
                w[~np.isfinite(w)] = 1.0
                
                #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
                m, em = linear_regression(wt, delta_t[ifreq,time_ind], w, intercept_origin=True)
                dvv[ifreq], err[ifreq] = m, em
            else:
                print('not enough points to estimate dv/v')
                dvv[ifreq], err[ifreq]=np.nan, np.nan    
        
    else: #using dynamic time window
        for ifreq in freq_indin:
            tmin, tmax = twindow[0], twindow[0]+nwindow*(1./freq[ifreq])  
        
            if (tmin < np.min(t)) | (tmax> np.max(t)) | (tmax <= tmin):
                print('Error: please input right time limits in the time window!')
            else:
                # truncate data with the time window
                time_ind = np.where((t >= tmin) & (t < tmax))[0]
                wt = t[time_ind]
                
            if len(wt)>2:
                if not np.any(delta_t[ifreq,time_ind]):
                    continue
                #---- use WXA as weight for regression----
                # w = 1.0 / (1.0 / (WCT[ifreq,:] ** 2) - 1.0)
                # w[WCT[ifreq,time_ind] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
                # w = np.sqrt(w * np.sqrt(WXA[ifreq,time_ind]))
                # w = np.real(w)
                w = 1/WCT[ifreq,time_ind]
                w[~np.isfinite(w)] = 1.0
                
                #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
                m, em = linear_regression(wt, delta_t[ifreq,time_ind], w, intercept_origin=True)
                dvv[ifreq], err[ifreq] = m, em
            else:
                print('not enough points to estimate dv/v')
                dvv[ifreq], err[ifreq]=np.nan, np.nan

    del cur, ref, twindow, fwindow
    del WCT, aWCT, coi, sig
    del phase, delta_t, freq_indout, freq_indin
    del t, w, wt, m, em
            
    return freq, dvv*100, err*100

def wts_allfreq(cur,ref,t,twindow,fwindow, maxdv=0.1, ndv=100, dj=1/12, s0=-1, J=-1, wvn='morlet',normalize=True,nwindow=1.5,windowflag=True):
    """
    Apply stretching method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range
    
    Parameters
    --------------
    :type cur: :class:`~numpy.ndarray`
    :param cur: 1d array. Cross-correlation measurements.
    :type ref: :class:`~numpy.ndarray`
    :param ref: 1d array. The reference trace.
    :type t: :class:`~numpy.ndarray`
    :param t: 1d array. Cross-correlation measurements.
    :param twindow: 1d array. [earlist time, latest time] time window limit
    :param fwindow: 1d array. [lowest frequncy, highest frequency] frequency window limit
    :params, dj, s0, J, wvn, refer to function 'wavelet.cwt'
    :normalize: True - normalize signals before stretching. Default is True
    :param maxdv: Velocity relative variation range [-maxdv, maxdv](100%)
    :param ndv : Number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
    :nwindow: the times of current period/frequency, which will be time window if windowflag is False 
    :windowflag: if True, the given window 'twindow' will be used, 
                 otherwise, the current period*nwindow will be used as time window
    
    Written by Congcong Yuan (30 Jun, 2019)  
    """
    
    dt = np.mean(np.diff(t))
    fmin, fmax = fwindow[0], fwindow[1]
    
    # apply cwt on two traces
    cwt1, sj, freq, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)
    
    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)
    
    # zero out data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        print('Error: please input right frequency limits in the frequenct window!')
    else:
        freq_indout = np.where((freq < fmin) | (freq > fmax))[0]
        rcwt1[freq_indout, :], rcwt2[freq_indout, :] = 0., 0.
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]

        # Use stretching method to extract dvv
        nfreq=len(freq)
        dvv, cc, cdp, err = np.zeros([nfreq,],dtype=np.float32), np.zeros([nfreq,],dtype=np.float32),\
        np.zeros([nfreq,],dtype=np.float32),np.zeros([nfreq,],dtype=np.float32)  
        
        for ifreq in freq_indin:
            if windowflag:
                tmin, tmax = twindow[0], twindow[1]
                if (tmin < np.min(t)) | (tmax> np.max(t)) | (tmax <= tmin):
                    print('Error: please input right time limits in the time window!')
                else:
                    # truncate data with the time window
                    time_ind = np.where((t >= tmin) & (t < tmax))[0] 
            
            else:
                tmin, tmax = twindow[0], twindow[0]+nwindow*(1./freq[ifreq])
                if (tmin < np.min(t)) | (tmax> np.max(t)) | (tmax <= tmin):
                    print('Error: please input right time limits in the time window!')
                else:
                    # truncate data with the time window
                    time_ind = np.where((t >= tmin) & (t < tmax))[0]
            
            # prepare time axis and its indices
            wt = t[time_ind]
            it = np.array(np.arange(0, len(wt)))
            # prepare windowed data                
            wcwt1, wcwt2 = rcwt1[ifreq, time_ind], rcwt2[ifreq, time_ind] 
            # Normalizes both signals, if appropriate.
            if normalize:
                ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
                ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
            else:
                ncwt1 = wcwt1
                ncwt2 = wcwt2
                
            # run stretching
            # dv, c1, c2, error = Stretching_current(ncwt2[ifreq,], ncwt1[ifreq,], wt, maxdv, ndv, it, freq[ifreq], freq[ifreq-1], tmin, tmax)
            dv, c1, c2, error = stretching(ncwt2, ncwt1, wt, maxdv, ndv, it, fmin, fmax, tmin, tmax)
            dvv[ifreq], cc[ifreq], cdp[ifreq], err[ifreq]=dv, c1, c2, error     
    
    del cur, ref, twindow, fwindow
    del cwt1, cwt2, rcwt1, rcwt2, ncwt1, ncwt2, wcwt1, wcwt2, coi, sj
    del time_ind, freq_indout, freq_indin
    del t, it, wt
    
    return freq, dvv, cc, cdp, err


def wtdtw_allfreq(cur,ref,t,twindow,fwindow, maxLag=50, b=5, direction=1, dj=1/12, s0=-1, J=-1, wvn='morlet',normalize=True,nwindow=1.5,windowflag=True):
    """
    Apply dynamic time warping method to continuous wavelet transformation (CWT) of signals
    for all frequecies in an interest range
    
    Parameters
    --------------
    :type cur: :class:`~numpy.ndarray`
    :param cur: 1d array. Cross-correlation measurements.
    :type ref: :class:`~numpy.ndarray`
    :param ref: 1d array. The reference trace.
    :type t: :class:`~numpy.ndarray`
    :param t: 1d array. Cross-correlation measurements.
    :param twindow: 1d array. [earlist time, latest time] time window limit
    :param fwindow: 1d array. [lowest frequncy, highest frequency] frequency window limit
    :params, dj, s0, J, wvn, refer to function 'wavelet.cwt'
    :normalize: True - normalize signals before stretching. Default is True
    :param maxLag : max number of points to search forward and backward. 
                Suggest setting it larger if window is set larger.
    :param b : b-value to limit strain, which is to limit the maximum velocity perturbation. 
               See equation 11 in (Mikesell et al. 2015)
    :nwindow: the times of current period/frequency, which will be time window if windowflag is False 
    :windowflag: if True, the given window 'twindow' will be used, 
                 otherwise, the current period*nwindow will be used as time window
    
    Written by Congcong Yuan (30 Jun, 2019)
    """
    
    dt = np.mean(np.diff(t))
    fmin, fmax = fwindow[0], fwindow[1]
 
    # apply cwt on two traces
    cwt1, sj, freq, coi, _, _ = pycwt.cwt(cur, dt, dj, s0, J, wvn)
    cwt2, sj, freq, coi, _, _ = pycwt.cwt(ref, dt, dj, s0, J, wvn)
    
    # extract real values of cwt
    rcwt1, rcwt2 = np.real(cwt1), np.real(cwt2)
    
    # zero out cone of influence and data outside frequency band
    if (fmax> np.max(freq)) | (fmax <= fmin):
        print('Error: please input right frequency limits in the frequenct window!')
    else:
        freq_indout = np.where((freq < fmin) | (freq > fmax))[0]
        rcwt1[freq_indout, :], rcwt1[freq_indout, :] = 0., 0.
        freq_indin = np.where((freq >= fmin) & (freq <= fmax))[0]
        
        # Use stretching method to extract dvv
        nfreq=len(freq)
        dvv, err = np.zeros([nfreq,],dtype=np.float32), np.zeros([nfreq,],dtype=np.float32)   
        for ifreq in freq_indin:
                    
            if windowflag:
                tmin, tmax = twindow[0], twindow[1]
                
                if (tmin < np.min(t)) | (tmax> np.max(t)) | (tmax <= tmin):
                    print('Error: please input right time limits in the time window!')
                else:
                    # truncate data with the time window
                    time_ind = np.where((t >= tmin) & (t < tmax))[0]      
            else:
                tmin, tmax = twindow[0], twindow[0]+nwindow*(1./freq[ifreq])
                
                if (tmin < np.min(t)) | (tmax> np.max(t)) | (tmax <= tmin):
                    print('Error: please input right time limits in the time window!')
                else:
                    # truncate data with the time window
                    time_ind = np.where((t >= tmin) & (t < tmax))[0]       
                    
            # prepare time axis and its indices
            wt = t[time_ind]
            it = np.array(np.arange(0, len(wt)))
            # prepare windowed data 
            wcwt1, wcwt2 = rcwt1[ifreq, time_ind], rcwt2[ifreq, time_ind]
            # Normalizes both signals, if appropriate.
            if normalize:
                ncwt1 = (wcwt1 - wcwt1.mean()) / wcwt1.std()
                ncwt2 = (wcwt2 - wcwt2.mean()) / wcwt2.std()
            else:
                ncwt1 = wcwt1
                ncwt2 = wcwt2
            # run dtw
            dv, error = dtw_dvv(ncwt2, ncwt1, wt, it, maxLag, b, direction)
            dvv[ifreq], err[ifreq]=dv, error     
    
    del cur, ref, twindow, fwindow
    del cwt1, cwt2, rcwt1, rcwt2, ncwt1, ncwt2, wcwt1, wcwt2, coi, sj
    del time_ind, freq_indout, freq_indin
    del t, it, wt
    
    return freq, dvv, err

if __name__ == "__main__":
    pass
