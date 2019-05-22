import os
import sys
import glob
import obspy
import pyasdf
import numpy as np
from obspy.io.sac.sactrace import SACTrace

'''
this script outputs the stacked cross-correlation functions into SAC traces
'''

#------absolute path-------
STACKDIR = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK1/E.ABHM'
ALLFILES = glob.glob(os.path.join(STACKDIR,'*.h5'))
COMP_OUT = ['ZZ','TT','RR']
dtype    = 'Allstacked'

nfiles = len(ALLFILES)
if not os.path.isdir(os.path.join(STACKDIR,'STACK_SAC')):
    os.mkdir(os.path.join(STACKDIR,'STACK_SAC'))

#----loop through station pairs----
for ii in range(nfiles):

    with pyasdf.ASDFDataSet(ALLFILES[ii],mode='r') as ds:
        
        #-----get station info from file name-----
        fname = ALLFILES[ii].split('/')[-1]
        staS = fname.split('_')[0].split('.')[1]
        netS = fname.split('_')[0].split('.')[0]
        staR = fname.split('_')[1].split('.')[1]
        netR = fname.split('_')[1].split('.')[0]

        #-----read data information-------
        slist = ds.auxiliary_data.list()
        rlist = ds.auxiliary_data[slist[0]].list()
        maxlag= ds.auxiliary_data[slist[0]][rlist[0]].parameters['lag']
        dt    = ds.auxiliary_data[slist[0]][rlist[0]].parameters['dt']
        slat  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['latS']
        slon  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['lonS']
        rlat  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['latR']
        rlon  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['lonR']

        if dtype in slist:
            for icomp in range(len(COMP_OUT)):
                comp = COMP_OUT[icomp]

                if comp in rlist:
                    corr = ds.auxiliary_data[dtype][comp].data[:]
                    temp = netS+'.'+staS+'_'+netR+'.'+staR+'_'+comp+'.SAC'
                    filename = os.path.join(STACKDIR,'STACK_SAC',temp)
                    sac = SACTrace(nzyear=2000,nzjday=1,nzhour=0,nzmin=0,nzsec=0,nzmsec=0,b=-maxlag,\
                        delta=dt,stla=rlat,stlo=rlon,evla=slat,evlo=slon,data=corr)
                    sac.write(filename,byteorder='big')