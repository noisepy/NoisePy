import sys
import pyasdf
import matplotlib
import numpy as np 
import pandas as pd 
import noise_module
import matplotlib.pyplot as plt 
from obspy.signal.filter import bandpass
from mpl_toolkits.axes_grid1 import make_axes_locatable

#----mute the warning----
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

# common parameters
tmin   = 20
tmax   = 50
maxLag = 20     # max nuber of points to search forward and backward
dt = 0.05       # time intervals
b = 5           # b-value to limit strain

#------load u0 and u1---------
h5file = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK1/E.ABHM/E.ABHM_E.OHSM.h5'
comp   = 'ZZ'
fmin   = 0.1
fmax   = 0.3
ds = pyasdf.ASDFDataSet(h5file,mode='r')
slist = ds.auxiliary_data.list()

#------loop through the reference waveforms------
if slist[0]== 'Allstacked':

    #------useful parameters from ASDF file------
    delta = ds.auxiliary_data[slist[0]][comp].parameters['dt']
    lag   = ds.auxiliary_data[slist[0]][comp].parameters['lag']

    #--------index for the data---------
    indx1 = int((lag+tmin)/delta)
    indx2 = int((lag+tmax)/delta)

    #-------------prepare the data matrix-----------------
    tdata = ds.auxiliary_data[slist[5]][comp].data[indx1:indx2+1]
    tdata = bandpass(tdata,fmin,fmax,int(1/dt),corners=4, zerophase=True)
    u0 = tdata/max(tdata)

    tdata = ds.auxiliary_data[slist[15]][comp].data[indx1:indx2+1]
    tdata = bandpass(tdata,fmin,fmax,int(1/dt),corners=4, zerophase=True)
    u1 = tdata/max(tdata) 
    del ds

#----------parepare other parameters-----------
lvec = np.arange(-maxLag,maxLag+1)*dt # lag array for plotting below
#st, u0, u1 = df['st'].values, df['u0'].values, df['u1'].values
npts = len(u0) # number of samples
tvec   = np.arange(npts) * dt # make the time axis

#-----plot the waveforms-----
plt.plot(tvec,u0,'r-')
plt.plot(tvec,u1,'b-')
plt.xlabel('Time [s]')
plt.ylabel('Normalized Amp')
plt.legend(['u0','u1'],loc='upper right')

# compute error function
err = noise_module.computeErrorFunction( u1, u0, npts, maxLag ) # compute error function over lags, which is independent of strain limit 'b'.

#------plot error functions-----
plt.figure(figsize=(10,10))
plt.imshow(np.flipud(np.log10(err.T + 1e-16)),aspect='auto',cmap=plt.cm.gray,extent=[tvec[0],tvec[-1],lvec[-1],lvec[0]])
plt.xlabel('Time [s]')
plt.ylabel('Lag')
plt.title('Error Function')
plt.colorbar()
plt.tight_layout()
plt.show()

direction = 1
# direction to accumulate errors (1=forward, -1=backward)
# it is instructive to flip the sign of +/-1 here to see how the function
# changes as we start the backtracking on different sides of the traces.
# Also change 'b' to see how this influences the solution for stbar. You
# want to make sure you're doing things in the proper directions in each
# step!!!

dist  = noise_module.accumulateErrorFunction( direction, err, npts, maxLag, b )
stbar = noise_module.backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )

stbarTime = stbar * dt   # convert from samples to time
tvec2     = tvec + stbarTime # make the warped time axis

# plot the results
fig, ax = plt.subplots(2,1,figsize=(20,10))
dist_mat = ax[0].imshow(dist.T,aspect='auto',extent=[tvec[0],tvec[-1],lvec[-1],lvec[0]])
ax[0].set_title('Distance function')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel(r'$\tau$ [s]')
ax[0].invert_yaxis()  
cax = fig.add_axes([0.65, 0.5, 0.3, 0.02])
fig.colorbar(dist_mat,cax=cax,orientation='horizontal')
# plot real shifts against estimated shifts
#ax[1].plot(tvec,stTime,'ko',label='Actual')
ax[1].plot(tvec,stbarTime,'r+',label='Estimated') 
ax[1].legend(fontsize=12)
ax[1].set_title('Estimated shifts')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel(r'$\tau$ [s]')
ax[1].set_xlim([tvec[0],tvec[-1]])
plt.autoscale(enable=True, tight=True)
fig.tight_layout()

# plot the input traces 
fig, ax = plt.subplots(2,1,figsize=(20,10),sharex=True)
ax[0].plot(tvec,u0,'b',label='Raw')
ax[0].plot(tvec,u1,'r--',label='Shifted')
ax[0].legend(loc='best',frameon=False)
ax[0].set_title('Input traces for dynamic time warping')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude (a.u.)')
# plot warped trace to compare
ax[1].plot(tvec,u0,'b',label='Raw')
ax[1].plot(tvec2,u1,'r--',label='Shifted')
ax[1].legend(loc='best',frameon=False)
ax[1].set_title('Output traces for dynamic time warping')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude (a.u.)')
plt.tight_layout()

# Apply dynamic time warping in both directions to smooth. (Follwoing example in Hale 2013)
dist1 = noise_module.accumulateErrorFunction( -1, err, npts, maxLag, b ) # forward accumulation to make distance function
dist2 = noise_module.accumulateErrorFunction( 1, err, npts, maxLag, b ); # backwward accumulation to make distance function

dist  = dist1 + dist2 - err; # add them and remove 'err' to not count twice (see Hale's paper)
stbar = noise_module.backtrackDistanceFunction( -1, dist, err, -maxLag, b ); # find shifts
# !! Notice now that you can backtrack in either direction and get the same
# result after you smooth the distance function in this way.

# plot the results
stbarTime = stbar * dt      # convert from samples to time
tvec2     = tvec + stbarTime # make the warped time axis

fig, ax = plt.subplots(2,1,figsize=(20,10))
dist_mat = ax[0].imshow(dist.T,aspect='auto',extent=[tvec[0],tvec[-1],lvec[-1],lvec[0]])
ax[0].set_title('Distance function')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel(r'$\tau$ [s]')
ax[0].invert_yaxis()  
cax = fig.add_axes([0.65, 0.5, 0.3, 0.02])
fig.colorbar(dist_mat,cax=cax,orientation='horizontal')
# plot real shifts against estimated shifts
#ax[1].plot(tvec,stTime,'ko',label='Actual')
ax[1].plot(tvec,stbarTime,'r+',label='Estimated') 
ax[1].legend(fontsize=12)
ax[1].set_title('Estimated shifts')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel(r'$\tau$ [s]')
ax[1].set_xlim([tvec[0],tvec[-1]])
plt.autoscale(enable=True, tight=True)
fig.tight_layout()

fig,ax = plt.subplots(2,1,figsize=(20,10))
ax[0].plot(tvec,u0,'b',label='Raw')
ax[0].plot(tvec,u1,'r--',label='Shifted')
ax[0].legend(loc='best',frameon=False)
ax[0].set_title('Input traces for dynamic time warping')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude (a.u.)')
# plot warped trace to compare
ax[1].plot(tvec,u0,'b',label='Raw')
ax[1].plot(tvec2,u1,'r--',label='Shifted')
ax[1].legend(loc='best',frameon=False)
ax[1].set_title('Output traces for dynamic time warping')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude (a.u.)')
plt.tight_layout()

plt.show()