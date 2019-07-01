import os
import scipy
import obspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.regression import linear_regression
from obspy.signal.invsim import cosine_taper

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
