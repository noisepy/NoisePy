import os
import glob
import sys
import obspy
import time
import noise_module
import numpy as np
import pandas as pd
from obspy.io.sac.sactrace import SACTrace
import pyasdf
from mpi4py import MPI

'''
this script outputs the stacked cross-correlation functions into SAC traces
'''




for icompS in range(len(compS)):
    for icompR in range(len(compR)):
        if nflag[icompS*3+icompR] >0:
            temp = netS+'.'+source+'_'+netR+'.'+receiver+'_'+compS[icompS]+'_'+compR[icompR]+'.SAC'
            filename = os.path.join(STACKDIR,source,temp)
            sac = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,\
                delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=ncorr[icompS*3+icompR,:])
            sac.write(filename,byteorder='big')