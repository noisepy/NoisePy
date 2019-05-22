import os
import sys
import time
import glob
import pyasdf
import numpy as np
import pandas as pd
import noise_module
from mpi4py import MPI

'''
this script stacks the cross-correlation functions according to the parameter of stack_days, 
which allows to explore the stability of the stacked ccfs for monitoring
'''

t0=time.time()

#-------------absolute path of working directory-------------
rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
CCFDIR = os.path.join(rootpath,'CCF')
FFTDIR = os.path.join(rootpath,'FFT')
STACKDIR = os.path.join(rootpath,'STACK')

#------------make correction due to mis-orientation of instruments---------------
corrfile = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/rotation/meso_angles.dat'
locs     = pd.read_csv(corrfile)
sta_list = list(locs.iloc[:]['station'])
angles   = list(locs.iloc[:]['angle'])


#---control variables---
flag = False
do_rotation   = True
one_component = False
stack_days = 5

maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq
pi = 3.141593

#----parameters to estimate SNR----
snr_parameters = {
    'freqmin':0.08,
    'freqmax':6,
    'steps': 15,
    'minvel': 0.5,
    'maxvel': 10,
    'noisewin':100}

#--------------9-component Green's tensor------------------
if not one_component:
    enz_components = ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']

    #---!!!each element is corresponding to each other in 2 systems!!!----
    if do_rotation:
        rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
else:
    enz_components = ['ZZ']


#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    #----check the directory of STACK----
    if os.path.exists(STACKDIR)==False:
        os.mkdir(STACKDIR)

    #------a way to keep same station-pair orders as S2-------
    sfiles = sorted(glob.glob(os.path.join(FFTDIR,'*.h5')))
    sta = []
    for ifile in sfiles:
        temp = ifile.split('/')[-1]
        ista = temp.split('.')[1]
        inet = temp.split('.')[0]

        #--------make directory for storing stacked data------------
        if not os.path.exists(os.path.join(STACKDIR,inet+'.'+ista)):
            os.mkdir(os.path.join(STACKDIR,inet+'.'+ista))
        sta.append(inet+'.'+ista)

    #-------make station pairs based on list--------        
    pairs= noise_module.get_station_pairs(sta)
    ccfs = sorted(glob.glob(os.path.join(CCFDIR,'*.h5')))
    splits = len(pairs)
else:
    pairs,ccfs,splits=[None for _ in range(3)]

#---------broadcast-------------
pairs  = comm.bcast(pairs,root=0)
ccfs   = comm.bcast(ccfs,root=0)
splits = comm.bcast(splits,root=0)
extra  = splits % size


#-----loop I: source station------
for ii in range(rank,splits+size-extra,size):
    
    if ii<splits:

        source,receiver = pairs[ii][0],pairs[ii][1]

        #----corr records every 10 days; ncorr records all days----
        corr  = np.zeros((len(enz_components),int(2*maxlag/dt)+1),dtype=np.float32)
        ncorr = np.zeros((len(enz_components),int(2*maxlag/dt)+1),dtype=np.float32)
        num1  = np.zeros(len(enz_components),dtype=np.int16)
        num2  = np.zeros(len(enz_components),dtype=np.int16)

        #-----source information-----
        staS = source.split('.')[1]
        netS = source.split('.')[0]

        #-----receiver information------
        staR = receiver.split('.')[1]
        netR = receiver.split('.')[0]

        #------keep a track of the starting date-----
        date_s = ccfs[0].split('/')[-1].split('.')[0]
        date_s = date_s.replace('_','')

        #-----loop through each day----
        for iday in range(len(ccfs)):
            if flag:
                print("source %s receiver %s at day %s" % (source,receiver,ccfs[iday].split('/')[-1]))

            fft_h5 = ccfs[iday]
            with pyasdf.ASDFDataSet(fft_h5,mpi=False,mode='r') as ds:

                #-------data types for source A--------
                data_types = ds.auxiliary_data.list()
                slist = np.array([s for s in data_types if staS in s])

                #---in case no such source-----
                if len(slist)==0:
                    print("no source %s at %dth day! continue" % (staS,iday))
                    continue

                for data_type in slist:
                    paths = ds.auxiliary_data[data_type].list()

                    #-------find the correspoinding receiver--------
                    rlist = np.array([r for r in paths if staR in r])
                    if len(rlist)==0:
                        print("no receiver %s for source %s at %dth day! continue" % (staR,staS,iday))
                        continue

                    if flag:
                        print('found the station-pair at %dth day' % iday)

                    #----------------copy the parameter information---------------
                    parameters  = ds.auxiliary_data[data_type][rlist[0]].parameters
                    for path in rlist:

                        #--------cross component-------
                        ccomp = data_type[-1]+path[-1]

                        #------put into a 2D matrix----------
                        tindx  = enz_components.index(ccomp)
                        corr[tindx] += ds.auxiliary_data[data_type][path].data[:]
                        ncorr[tindx]+= ds.auxiliary_data[data_type][path].data[:]
                        num1[tindx] += 1
                        num2[tindx] += 1

            #------stack every n(10) day or whatever is left-------
            if (iday+1)%stack_days==0:

                #------keep a track of ending date for stacking------
                date_e = ccfs[iday].split('/')[-1].split('.')[0]
                date_e = date_e.replace('_','')

                if flag:
                    print('write the stacked data to ASDF between %s and %s' % (date_s,date_e))

                #------------------output path and file name----------------------
                stack_h5 = os.path.join(STACKDIR,source+'/'+source+'_'+receiver+'.h5')
                crap   = np.zeros(int(2*maxlag/dt)+1,dtype=np.float32)

                #------in case it already exists------
                if not os.path.isfile(stack_h5):
                    with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
                        pass 

                with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:

                    #-----loop through all E-N-Z components-----
                    for ii in range(len(enz_components)):
                        icomp = enz_components[ii]

                        #------do average-----
                        if num1[ii]==0:
                            print('station-pair %s_%s no data in %d days for component of %s: filling zero' % (source,receiver,stack_days,icomp))
                        else:
                            corr[ii] = corr[ii]/num1[ii]

                        if flag:
                            print('estimate the SNR of component %s for %s_%s in E-N-Z system' % (enz_components[ii],source,receiver))
                        #--------evaluate the SNR of the signal at target period range-------
                        new_parameters = noise_module.get_SNR(corr[ii],snr_parameters,parameters)

                        #------save the time domain cross-correlation functions-----
                        data_type = 'F'+date_s+'T'+date_e
                        path = icomp
                        crap = corr[ii]
                        stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=new_parameters)

                        if not do_rotation:
                            #----reset----
                            corr[ii] = 0
                            num1[ii] = 0

                    #-------do rotation here if needed---------
                    if do_rotation:
                        if flag:
                            print('doing matrix rotation now!')

                        #---read azi, baz info---
                        azi = parameters['azi']
                        baz = parameters['baz']

                        #---angles to be corrected----
                        ind = sta_list.index(staS)
                        acorr = angles[ind]
                        ind = sta_list.index(staR)
                        bcorr = angles[ind]
                        cosa = np.cos((azi+acorr)*pi/180)
                        sina = np.sin((azi+acorr)*pi/180)
                        cosb = np.cos((baz+bcorr)*pi/180)
                        sinb = np.sin((baz+bcorr)*pi*180)

                        #------9 component tensor rotation 1-by-1------
                        for ii in range(len(rtz_components)):
                            
                            if ii==0:
                                crap = -cosb*corr[7]-sinb*corr[6]
                            elif ii==1:
                                crap = sinb*corr[7]-cosb*corr[6]
                            elif ii==2:
                                crap = corr[8]
                                continue
                            elif ii==3:
                                crap = -cosa*cosb*corr[4]-cosa*sinb*corr[3]-sina*cosb*corr[1]-sina*sinb*corr[0]
                            elif ii==4:
                                crap = cosa*sinb*corr[4]-cosa*cosb*corr[3]+sina*sinb*corr[1]-sina*cosb*corr[0]
                            elif ii==5:
                                crap = cosa*corr[5]+sina*corr[2]
                            elif ii==6:
                                crap = sina*cosb*corr[4]+sina*sinb*corr[3]-cosa*cosb*corr[1]-cosa*sinb*corr[0]
                            elif ii==7:
                                crap = -sina*sinb*corr[4]+sina*cosb*corr[3]+cosa*sinb*corr[1]-cosa*cosb*corr[0]
                            else:
                                crap = -sina*corr[5]+cosa*corr[2]

                            if flag:
                                print('estimate the SNR of component %s for %s_%s in R-T-Z system' % (rtz_components[ii],source,receiver))
                            #--------evaluate the SNR of the signal at target period range-------
                            new_parameters = noise_module.get_SNR(crap,snr_parameters,parameters)

                            #------save the time domain cross-correlation functions-----
                            data_type = 'F'+date_s+'T'+date_e
                            path = rtz_components[ii]
                            stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=new_parameters)

                        #----reset----
                        corr[ii] = 0
                        num1[ii] = 0

                if iday != len(ccfs)-1:        
                    date_s = ccfs[iday+1].split('/')[-1].split('.')[0]
                    date_s = date_s.replace('_','')

        #--------------now stack all of the days---------------
        with pyasdf.ASDFDataSet(stack_h5,mpi=False) as stack_ds:
            for ii in range(len(enz_components)):
                icomp = enz_components[ii]

                #------do average here--------
                if num2[ii]==0:
                    print('station-pair %s_%s no data in at all for components %s: filling zero' % (source,receiver,icomp))
                else:
                    ncorr[ii] = ncorr[ii]/num2[ii]
                
                #--------evaluate the SNR of the signal at target period range-------
                new_parameters = noise_module.get_SNR(ncorr[ii],snr_parameters,parameters)

                #------save the time domain cross-correlation functions-----
                data_type = 'Allstacked'
                path = icomp
                crap = ncorr[ii]
                stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=new_parameters)

            #----do rotation-----
            if do_rotation:

                #------9 component tensor rotation 1-by-1------
                for ii in range(len(rtz_components)):
                    
                    if ii==0:
                        crap = -cosb*ncorr[7]-sinb*ncorr[6]
                    elif ii==1:
                        crap = sinb*ncorr[7]-cosb*ncorr[6]
                    elif ii==2:
                        crap = ncorr[8]
                        continue
                    elif ii==3:
                        crap = -cosa*cosb*ncorr[4]-cosa*sinb*ncorr[3]-sina*cosb*ncorr[1]-sina*sinb*ncorr[0]
                    elif ii==4:
                        crap = cosa*sinb*ncorr[4]-cosa*cosb*ncorr[3]+sina*sinb*ncorr[1]-sina*cosb*ncorr[0]
                    elif ii==5:
                        crap = cosa*ncorr[5]+sina*ncorr[2]
                    elif ii==6:
                        crap = sina*cosb*ncorr[4]+sina*sinb*ncorr[3]-cosa*cosb*ncorr[1]-cosa*sinb*ncorr[0]
                    elif ii==7:
                        crap = -sina*sinb*ncorr[4]+sina*cosb*ncorr[3]+cosa*sinb*ncorr[1]-cosa*cosb*ncorr[0]
                    else:
                        crap = -sina*ncorr[5]+cosa*ncorr[2]

                    #--------evaluate the SNR of the signal at target period range-------
                    new_parameters = noise_module.get_SNR(crap,snr_parameters,parameters)

                    #------save the time domain cross-correlation functions-----
                    data_type = 'Allstacked'
                    path = rtz_components[ii]
                    stack_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=new_parameters)


t1=time.time()
print('S3 takes '+str(t1-t0)+' s')

#---ready to exit---
comm.barrier()
if rank == 0:
    sys.exit()
