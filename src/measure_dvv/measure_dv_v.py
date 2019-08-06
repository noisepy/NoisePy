import matplotlib.pyplot as plt
import core_functions
import numpy as np 
import matplotlib
import pyasdf
import scipy
import glob
import sys
import os

'''
a compilation of 6 different methods to estimate dv/v values for stacked waveforms 
in ASDF files. the 6 methods are 1) stretch, 2) MWCS, 3) DTW, 4) wct, 5) wts and 6) wtdtw. 

update to add the comparision of the resulted dv/v from positive lag, negative lag 
and the averaged waveforms for the strecthing part
'''

##############################################
############# parameter section ##############
##############################################

# take one station-pair as an example here
sfile = '/Users/chengxin/Documents/NoisePy_example/SCAL/STACK/CI.ADO.00/linear_CI.ADO.00_CI.CCC.00.h5'

# targeted window and frequency range
twin    = [20,100]                          # targeted time window for waveform monitoring (could be both lags)
fband   = [0.3,1]                           # targeted frequency band for waveform monitoring
dt      = 0.05                              # proposed time interval for the targeted data
ccomp   = 'ZZ'                              # measurements on which cross-component
onelag  = False                             # make measurement one one lag or two 
norm_flag = True                            # whether to normalize the cross-correlation waveforms

# save parameters as a dic
para = {'twin':twin,'fband':fband,'ccomp':ccomp,'onelag':onelag,'norm_flag':norm_flag}

# variables for stretching method
epsilon = 0.01                              # limit for the dv/v range (*100 to get range in %)
nbtrial = 50                                # number of increment of dt [-epsilon,epsilon] for the streching method

# variables for MWCS
movewin_length = 3*int(1/np.min(fband))     # length of the moving window  
sstep = 0.2*movewin_length                  # sliding step for moving window

if movewin_length > 2*(np.max(twin)-np.min(twin)):
    raise IOError('twin is set too small for MWCS')

# variables for DTW


###############################################
############ load data from ASDF ##############
###############################################

# load the data 
ref,data,para = core_functions.load_waveforms(sfile,para)

# loop through each substacks
for ii in range(len(data)):
    cur = data[ii]

    # plug into functions for dv/v measurements
    dv1,cc,cdp,error1 = core_functions.stretching(ref,cur,epsilon,nbtrial,para)
    dv2,error2 = core_functions.mwcs_dvv(ref,cur,movewin_length,sstep,para)
    dv3,error3 = core_functions.WCC_dvv(ref,cur,movewin_length,sstep,para)

    # cluster the info for plotting

###############################################
######### collect data for plotting ###########
###############################################
# plot settings 
font = {'family':'normal','weight':'bold','size':10}
matplotlib.rc('font', **font)