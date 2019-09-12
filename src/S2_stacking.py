import sys
import time
import obspy
import pyasdf
import os, glob
import datetime
import numpy as np
import noise_module
import pandas as pd
from mpi4py import MPI

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
Stacking script of NoisePy:
    1) read the saved cross-correlation data to do sub-stacks (if needed) and all-time averaging;
    2) two options for the stacking process: linear and phase weighted stacking (pws);
    3) save outputs in ASDF or SAC format depend on user's choice;
    4) rotation from a E-N-Z to R-T-Z system if needed.

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@fas.harvard.edu)

Note: 
    1) assuming 3 components are E-N-Z 
    2) auto-correlation is not kept in the stacking due to the fact that it has only 6 cross-component.
    this tends to mess up the orders of matrix that stores the CCFs data
'''

tt0=time.time()

########################################
#########PARAMETER SECTION##############
########################################

# absolute path parameters
rootpath  = '/Volumes/Chengxin/LV_monitor'                       # root path for this data processing
CCFDIR    = os.path.join(rootpath,'CCF')                    # dir where CC data is stored
STACKDIR  = os.path.join(rootpath,'STACK') 
locations = os.path.join(rootpath,'station.lst')            # station info including network,station,channel,latitude,longitude,elevation
if not os.path.isfile(locations): 
    raise ValueError('Abort! station info is needed for this script')

# load fc_para parameters from Step1
fc_metadata = os.path.join(CCFDIR,'fft_cc_data.txt')
fc_para     = eval(open(fc_metadata).read())
ncomp       = fc_para['ncomp']
samp_freq   = fc_para['samp_freq']
start_date  = fc_para['start_date']
end_date    = fc_para['end_date']
inc_hours   = fc_para['inc_hours']
cc_len      = fc_para['cc_len']
step        = fc_para['step']
maxlag      = fc_para['maxlag']
substack    = fc_para['substack']
substack_len= fc_para['substack_len']

# define new stacking para
keep_substack= True                                         # keep all sub-stacks in final ASDF file
flag         = False                                        # output intermediate args for debugging
stack_method = 'both'                                        # linear, pws or both

# cross component info
if ncomp==1:enz_system = ['ZZ']
else: enz_system = ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']
rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']

# new rotation para
rotation     = False                                        # rotation from E-N-Z to R-T-Z 
correction   = False                                        # angle correction due to mis-orientation
if rotation and correction:
    corrfile = os.path.join(rootpath,'meso_angles.dat')          # csv file containing angle info to be corrected
    locs     = pd.read_csv(corrfile)
else: locs = []

# maximum memory allowed per core in GB
MAX_MEM = 4.0

# make a dictionary to store all variables: also for later cc
stack_para={'samp_freq':samp_freq,'cc_len':cc_len,'step':step,'rootpath':rootpath,'STACKDIR':\
    STACKDIR,'start_date':start_date[0],'end_date':end_date[0],'inc_hours':inc_hours,'substack':substack,\
    'substack_len':substack_len,'maxlag':maxlag,'MAX_MEM':MAX_MEM,'keep_substack':keep_substack,\
    'stack_method':stack_method,'rotation':rotation,'correction':correction}
# save fft metadata for future reference
stack_metadata  = os.path.join(STACKDIR,'stack_data.txt') 

#######################################
###########PROCESSING SECTION##########
#######################################

#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if not os.path.isdir(STACKDIR):os.mkdir(STACKDIR)
    # save metadata 
    fout = open(stack_metadata,'w')
    fout.write(str(stack_para));fout.close()

    # cross-correlation files
    ccfiles   = sorted(glob.glob(os.path.join(CCFDIR,'*.h5')))

    # load station info
    tlocs = pd.read_csv(locations)
    sta = sorted(np.unique(tlocs['network']+'.'+tlocs['station']))
    for ii in range(len(sta)):
        tmp = os.path.join(STACKDIR,sta[ii])
        if not os.path.isdir(tmp):os.mkdir(tmp)

    # station-pairs
    pairs_all = []
    for ii in range(len(sta)-1):
        for jj in range(ii,len(sta)):
            pairs_all.append(sta[ii].replace('.','s')+'s'+sta[jj].replace('.','s'))

    splits  = len(pairs_all)
    if len(ccfiles)==0 or splits==0:
        raise IOError('Abort! no available CCF data for stacking')

else:
    splits,ccfiles,pairs_all = [None for _ in range(3)]

# broadcast the variables
splits    = comm.bcast(splits,root=0)
ccfiles   = comm.bcast(ccfiles,root=0)
pairs_all = comm.bcast(pairs_all,root=0)

# MPI loop: loop through each user-defined time chunck
for ipair in range (rank,splits,size):
    t0=time.time()

    if flag:print('%dth path for station-pair %s'%(ipair,pairs_all[ipair]))
    # source folder
    ttr   = pairs_all[ipair].split('s')
    snet  = ttr[0];ssta = ttr[1]
    rnet  = ttr[2];rsta = ttr[3]
    idir  = snet+'.'+ssta

    # continue when file is done
    toutfn = os.path.join(STACKDIR,idir+'/'+snet+'.'+ssta+'_'+rnet+'.'+rsta+'.tmp')   
    if os.path.isfile(toutfn):continue        

    # crude estimation on memory needs (assume float32)
    nccomp     = ncomp*ncomp
    num_chunck = len(ccfiles)*nccomp
    num_segmts = 1
    if substack:    # things are difference when do substack
        if substack_len==cc_len:
            num_segmts = int(np.floor((inc_hours*3600-cc_len)/step))
        else:
            num_segmts = int(inc_hours/(substack_len/3600))
    npts_segmt  = int(2*maxlag*samp_freq)+1
    memory_size = num_chunck*num_segmts*npts_segmt*4/1024**3

    if memory_size > MAX_MEM:
        raise ValueError('Require %s G memory (%s GB provided)! Cannot load cc data all once!' % (memory_size,MAX_MEM))
    if flag:
        print('Good on memory (need %5.2f G and %s G provided)!' % (memory_size,MAX_MEM))
        
    # allocate array to store fft data/info
    cc_array = np.zeros((num_chunck*num_segmts,npts_segmt),dtype=np.float32)
    cc_time  = np.zeros(num_chunck*num_segmts,dtype=np.float)
    cc_ngood = np.zeros(num_chunck*num_segmts,dtype=np.int16)
    cc_comp  = np.chararray(num_chunck*num_segmts,itemsize=2,unicode=True)

    # loop through all time-chuncks
    iseg = 0
    dtype = pairs_all[ipair] 
    for ifile in ccfiles:

        # load the data from daily compilation
        ds=pyasdf.ASDFDataSet(ifile,mpi=False,mode='r')
        try:
            path_list   = ds.auxiliary_data[dtype].list()
            tparameters = ds.auxiliary_data[dtype][path_list[0]].parameters 
        except Exception: 
            if flag:print('continue! no pair of %s in %s'%(dtype,ifile))
            continue
        
        if ncomp==3 and len(path_list)<9:
            if flag:print('continue! not enough cross components for %s in %s'%(dtype,ifile))
            continue
                   
        # load the 9-component data, which is in order in the ASDF
        for tpath in path_list:
            cmp1 = tpath.split('s')[0]
            cmp2 = tpath.split('s')[1]
            tcmp1 = cmp1[-1];tcmp2 = cmp2[-1]
            if cmp1[-1] == 'U':tcmp1 = 'Z'
            if cmp2[-1] == 'U':tcmp2 = 'Z'

            # read data and parameter matrix
            tdata = ds.auxiliary_data[dtype][tpath].data[:]
            ttime = ds.auxiliary_data[dtype][tpath].parameters['time']
            tgood = ds.auxiliary_data[dtype][tpath].parameters['ngood']
            if substack:
                for ii in range(tdata.shape[0]):
                    cc_array[iseg] = tdata[ii]
                    cc_time[iseg]  = ttime[ii]
                    cc_ngood[iseg] = tgood[ii]
                    cc_comp[iseg]  = tcmp1+tcmp2
                    iseg+=1
            else:
                cc_array[iseg] = tdata
                cc_time[iseg]  = ttime
                cc_ngood[iseg] = tgood
                cc_comp[iseg]  = tcmp1+tcmp2
                iseg+=1

    t1=time.time()
    if flag:print('loading CCF data takes %6.2fs'%(t1-t0))

    # continue when there is no data
    if iseg <= 1: continue
    outfn = snet+'.'+ssta+'_'+rnet+'.'+rsta+'.h5'         
    if flag:print('ready to output to %s'%(outfn))                     

    # matrix used for rotation
    if rotation:bigstack=np.zeros(shape=(9,npts_segmt),dtype=np.float32)
    if stack_method =='both':bigstack1=np.zeros(shape=(9,npts_segmt),dtype=np.float32)

    # loop through cross-component for stacking
    iflag=1
    for icomp in range(nccomp):
        comp = enz_system[icomp]
        indx = np.where(cc_comp==comp)[0]

        # jump if there are not enough data
        if len(indx)<2: 
            iflag=0;break

        t2=time.time()
        stack_h5 = os.path.join(STACKDIR,idir+'/'+outfn)
        # output stacked data
        if stack_method != 'both':
            cc_final,ngood_final,stamps_final,allstacks,nstacks = noise_module.stacking(cc_array[indx],cc_time[indx],cc_ngood[indx],stack_para)
            if not len(allstacks):continue
            if rotation:bigstack[icomp]=allstacks

            # write stacked data into ASDF file
            with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
                tparameters['time']  = stamps_final[0]
                tparameters['ngood'] = nstacks
                data_type = 'Allstack0'+stack_method
                ds.add_auxiliary_data(data=allstacks, data_type=data_type, path=comp, parameters=tparameters)
        else:
            cc_final,ngood_final,stamps_final,allstacks1,allstacks2,nstacks = noise_module.stacking(cc_array[indx],cc_time[indx],cc_ngood[indx],stack_para)
            if not len(allstacks1):continue
            if rotation:
                bigstack[icomp] =allstacks1
                bigstack1[icomp]=allstacks2

            # write stacked data into ASDF file
            with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
                tparameters['time']  = stamps_final[0]
                tparameters['ngood'] = nstacks
                ds.add_auxiliary_data(data=allstacks1, data_type='Allstack0linear', path=comp, parameters=tparameters)
                ds.add_auxiliary_data(data=allstacks2, data_type='Allstack0pws', path=comp, parameters=tparameters)

        # keep a track of all sub-stacked data from S1
        if keep_substack:
            for ii in range(cc_final.shape[0]):
                with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
                    tparameters['time']  = stamps_final[ii]
                    tparameters['ngood'] = ngood_final[ii]
                    data_type = 'T'+str(int(stamps_final[ii]))
                    ds.add_auxiliary_data(data=cc_final[ii], data_type=data_type, path=comp, parameters=tparameters)            
        
        t3 = time.time()
        if flag:print('takes %6.2fs to stack one component with %s stacking method' %(t3-t1,stack_method))

    # do rotation if needed
    if rotation and iflag:
        if np.all(bigstack==0):continue
        tparameters['station_source'] = ssta
        tparameters['station_receiver'] = rsta
        if stack_method!='both':
            bigstack_rotated = noise_module.rotation2(bigstack,tparameters,locs,flag)

            # write to file
            for icomp in range(nccomp):
                comp  = rtz_components[icomp]
                tparameters['time']  = stamps_final[0]
                tparameters['ngood'] = nstacks
                data_type = 'Allstack0'+stack_method
                with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds2:
                    ds2.add_auxiliary_data(data=bigstack_rotated[icomp], data_type=data_type, path=tpath, parameters=tparameters)
        else:
            bigstack_rotated  = noise_module.rotation2(bigstack,tparameters,locs,flag)
            bigstack_rotated1 = noise_module.rotation2(bigstack1,tparameters,locs,flag)

            # write to file
            for icomp in range(nccomp):
                comp=rtz_components[icomp]
                tparameters['time']  = stamps_final[0]
                tparameters['ngood'] = nstacks
                with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds2:
                    ds2.add_auxiliary_data(data=bigstack_rotated[icomp], data_type='Allstack0linear', path=comp, parameters=tparameters)    
                    ds2.add_auxiliary_data(data=bigstack_rotated1[icomp], data_type='Allstack0pws', path=comp, parameters=tparameters)

    t4 = time.time()
    if flag:print('takes %6.2fs to stack/rotate station pair %s' %(t4-t1,pairs_all[ipair]))

    # write file stamps 
    ftmp = open(toutfn,'w');ftmp.write('done');ftmp.close()

tt1 = time.time()
print('it takes %6.2fs to process step 2 in total' % (tt1-tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()
