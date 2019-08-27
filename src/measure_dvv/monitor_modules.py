import os
import scipy
import pycwt
import datetime
import numpy as np
from obspy.core.util.base import _get_function_from_entry_point


def detrend(data):
    '''
    remove the trend of the signal based on QR decomposion
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
    remove the mean of the signal
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
    apply a cosine taper using obspy functions
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


def nextpow2(x):
    """
    Returns the next power of 2 of x.

    :type x: int 
    :returns: the next power of 2 of x

    """

    return int(np.ceil(np.log2(np.abs(x)))) 


def getCoherence(dcs, ds1, ds2):
    # TODO: docsting
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2) > 0))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh


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


def wct_modified(y1, y2, dt, dj=1/12, s0=-1, J=-1, sig=True,
        significance_level=0.95, wavelet='morlet', normalize=True, **kwargs):
    """Wavelet coherence transform (WCT).

    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.

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

    Returns
    -------
    TODO: Something TBA and TBC

    See also
    --------
    cwt, xwt

    """
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

    return WXS, WXA, WCT, aWCT, coi, freq, sig