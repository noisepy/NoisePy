import os
import math
import time
import glob
import obspy
import pyasdf
import pandas as pd
import numpy as np


'''
this script rotates the 9-component Green's tensor from E-N-Z system to R-T-Z system

ideas: create a new stacking script to stack a certain number of days and store data 
in ASDF file format 
'''

ttt0=time.time()

#----absolute path-------
rootpath = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW'
CCFDIR   = os.path.join(rootpath,'CCF')
STACKDIR = os.path.join(rootpath,'STACK')
ROTDIR   = os.path.join(rootpath,'ROT')
ccfs     = sorted(glob.glob(os.path.join(CCFDIR,'2010_12_17.h5')))

#------------make correction due to mis-orientation of instruments---------------
corrfile = '/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/rotation/meso_angles.dat'
locs     = pd.read_csv(corrfile)
sta_list = list(locs.iloc[:]['station'])
angles   = list(locs.iloc[:]['angle'])

#----some common parameters-----
pi   = 3.14159

if not os.path.isdir(ROTDIR):
    os.mkdir(ROTDIR)

for iday in range(len(ccfs)):
    iname = ccfs[iday].split('/')[-1]
    with pyasdf.ASDFDataSet(ccfs[iday],mpi=False,mode='r') as ds:
        slist = ds.auxiliary_data.list()
        
        t00=time.time()
        #------for each source------
        for isource in range(len(slist)//3):

            #------this is dangeous because some components for one station may be missing------
            sE = slist[isource*3]
            sN = slist[isource*3+1]
            sZ = slist[isource*3+2]
            #----station info------
            ssta = sE.split('s')[1]
            snet = sE.split('s')[0]
            rlist = ds.auxiliary_data[sE].list()
            #------correction angles-------
            ind = sta_list.index(ssta)
            acorr = angles[ind]
            
            #------for each receiver-------
            for ireceiver in range(len(rlist)//3):
                rE = rlist[ireceiver*3]
                rN = rlist[ireceiver*3+1]
                rZ = rlist[ireceiver*3+2]
                
                #-----station info------
                rsta = rE.split('s')[1]
                rnet = rE.split('s')[0]

                #-----correction angles-------
                ind = sta_list.index(rsta)
                bcorr = angles[ind]

                #-----remember to write azi, baz, distance into parameters-----
                azi = ds.auxiliary_data[sE][rE].parameters['azi']
                baz = ds.auxiliary_data[sE][rE].parameters['baz']
                parameters = ds.auxiliary_data[sE][rE].parameters

                #------construct the 9 component green's tensor------
                cosa = math.cos((azi+acorr)*pi/180)
                sina = math.sin((azi+acorr)*pi/180)
                cosb = math.cos((baz+bcorr)*pi/180)
                sinb = math.sin((baz+bcorr)*pi*180)

                ee = ds.auxiliary_data[sE][rE].data[:]
                en = ds.auxiliary_data[sE][rN].data[:]
                ez = ds.auxiliary_data[sE][rZ].data[:]
                ne = ds.auxiliary_data[sN][rE].data[:]
                nn = ds.auxiliary_data[sN][rN].data[:]
                nz = ds.auxiliary_data[sN][rZ].data[:]
                ze = ds.auxiliary_data[sZ][rE].data[:]
                zn = ds.auxiliary_data[sZ][rN].data[:]
                zz = ds.auxiliary_data[sZ][rZ].data[:]
                
                #------------for all 9 components-------------
                npts  = len(zz)
                Gn    = np.zeros(shape=(9,npts),dtype=np.float32)
                Gn[0] = -cosb*zn-sinb*ze
                Gn[1] = sinb*zn-cosb*ze
                Gn[2] = zz
                Gn[3] = -cosa*cosb*nn-cosa*sinb*ne-sina*cosb*en-sina*sinb*ee
                Gn[4] = cosa*sinb*nn-cosa*cosb*ne+sina*sinb*en-sina*cosb*ee
                Gn[5] = cosa*nz+sina*ez
                Gn[6] = sina*cosb*nn+sina*sinb*ne-cosa*cosb*en-cosa*sinb*ee
                Gn[7] = -sina*sinb*nn+sina*cosb*ne+cosa*sinb*en-cosa*cosb*ee
                Gn[8] = -sina*nz+cosa*ez
                
                rotated_h5 = os.path.join(ROTDIR,iname)
                #-------write into a new file-------
                if not os.path.isfile(rotated_h5):
                    with pyasdf.ASDFDataSet(rotated_h5,mpi=False) as rot_ds:
                        pass 

                with pyasdf.ASDFDataSet(rotated_h5,mpi=False) as rot_ds:

                    #------save the time domain cross-correlation functions-----
                    data_type = snet+'s'+ssta+'s'+rnet+'s'+rsta
                    path = 'ZR'
                    crap = Gn[0]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    path = 'ZT'
                    crap = Gn[1]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    path = 'ZZ'
                    crap = Gn[2]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    path = 'RR'
                    crap = Gn[3]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    path = 'RT'
                    crap = Gn[4]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    path = 'RZ'
                    crap = Gn[5]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    path = 'TR'
                    crap = Gn[6]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    path = 'TT'
                    crap = Gn[7]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)
                    path = 'TZ'
                    crap = Gn[8]
                    rot_ds.add_auxiliary_data(data=crap, data_type=data_type, path=path, parameters=parameters)

        t10=time.time()
        print('each day takes %6.2f s' %(t10-t00))

ttt1 = time.time()
print('rotation takes %8.2f s in total' % (ttt1-ttt0))