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
'''

tt0=time.time()

########################################
#########PARAMETER SECTION##############
########################################

# absolute path parameters
rootpath  = '/Users/chengxin/Documents/NoisePy_example/JAKARTA'                # root path for this data processing
CCFDIR    = os.path.join(rootpath,'CCF')                    # dir where CC data is stored
STACKDIR  = os.path.join(rootpath,'STACK') 

# load fc_para from S1
fc_metadata = os.path.join(CCFDIR,'fft_cc_data.txt')
fc_para     = eval(open(fc_metadata).read())
samp_freq   = fc_para['samp_freq']
start_date  = fc_para['start_date']
end_date    = fc_para['end_date']
inc_hours   = fc_para['inc_hours']
cc_len      = fc_para['cc_len']
step        = fc_para['step']
maxlag      = fc_para['maxlag']
substack    = fc_para['substack']
substack_len= fc_para['substack_len']

# stacking para
f_substack = True                                           # whether to do sub-stacking (different from that in S1)
f_substack_len = substack_len                               # length for sub-stacking to output
out_format   = 'asdf'                                       # ASDF or SAC format for output
flag         = True                                         # output intermediate args for debugging
stack_method = 'linear'                                     # linear, pws

# rotation para
rotation     = 'False'                                      # rotation from E-N-Z to R-T-Z 
correction   = 'False'                                       # angle correction due to mis-orientation
if rotation and correction:
    corrfile = '/Users/chengxin/Documents/NoisePy_example/SCAL/angles.dat'     # csv file containing angle info to be corrected
    locs     = pd.read_csv(corrfile)

# maximum memory allowed per core in GB
MAX_MEM = 4.0

# make a dictionary to store all variables: also for later cc
stack_para={'samp_freq':samp_freq,'cc_len':cc_len,'step':step,'rootpath':rootpath,'STACKDIR':\
    STACKDIR,'start_date':start_date[0],'end_date':end_date[0],'inc_hours':inc_hours,'substack':substack,\
    'substack_len':substack_len,'maxlag':maxlag,'MAX_MEM':MAX_MEM,'f_substack':f_substack,'f_substack_len':\
    f_substack_len,'stack_method':stack_method,'rotation':rotation,'correction':correction}
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

    # load all station-pair info
    paths_all = noise_module.load_pfiles(ccfiles)
    splits  = len(paths_all)
    if len(ccfiles)==0 or splits==0:
        raise IOError('Abort! no available CCF data for stacking')

    # make directories for storing stacked data
    for ii in range(splits):
        tr   = paths_all[ii].split('s')
        tdir = os.path.join(STACKDIR,tr[0]+'.'+tr[1]+'.'+tr[3])
        if not os.path.isdir(tdir):os.mkdir(tdir)
else:
    splits,ccfiles,paths_all = [None for _ in range(3)]

# broadcast the variables
splits    = comm.bcast(splits,root=0)
ccfiles   = comm.bcast(ccfiles,root=0)
paths_all = comm.bcast(paths_all,root=0)

# MPI loop: loop through each user-defined time chunck
for ipath in range (rank,splits,size):
    t0=time.time()

    if flag:print('%dth path for station-pair %s'%(ipath,paths_all[ipath]))
    # source folder
    ttr   = paths_all[ipath].split('s')
    idir  = ttr[0]+'.'+ttr[1]+'.'+ttr[3]
    outfn = ttr[0]+'.'+ttr[1]+'.'+ttr[2]+'_'+ttr[4]+'.'+ttr[5]+'.'+ttr[6]+'.h5'
    if flag:print('source %s and output %s'%(idir,outfn))

    # crude estimation on memory needs (assume float32)
    num_chunck  = len(ccfiles)
    if substack:
        num_segmts = int(np.floor((inc_hours*3600-cc_len)/step))+1
    else: 
        num_segmts = 1
    npts_segmt  = int(2*maxlag*samp_freq)+1
    memory_size = num_chunck*num_segmts*npts_segmt*4/1024**3
    if memory_size > MAX_MEM:
        raise ValueError('Require %s G memory (%s GB provided)! Cannot load cc data all once!' % (memory_size,MAX_MEM))
    if flag:
        print('Good on memory (need %5.2f G and %s G provided)!' % (memory_size,MAX_MEM))
        
    # open array to store fft data/info in memory
    cc_array = np.zeros((num_chunck*num_segmts,npts_segmt),dtype=np.float32)
    cc_time  = np.zeros(num_chunck*num_segmts,dtype=np.float)
    cc_ngood = np.zeros(num_chunck*num_segmts,dtype=np.int16)

    # loop through all time-chuncks
    iseg = 0
    station_pair = paths_all[ipath]
    for ifile in range(len(ccfiles)):
        ds=pyasdf.ASDFDataSet(ccfiles[ifile],mpi=False,mode='r')
        if not ds.auxiliary_data.list(): continue
        path_list = ds.auxiliary_data['CCF'].list()            
        if station_pair not in path_list:
            if flag:print('continue! no data for %s in %s'%(station_pair,ccfiles[ifile]))
            continue

        # load the data by segments
        tdata = ds.auxiliary_data['CCF'][station_pair].data[:]
        ttime = ds.auxiliary_data['CCF'][station_pair].parameters['time']
        tgood = ds.auxiliary_data['CCF'][station_pair].parameters['ngood']
        tparameters = ds.auxiliary_data['CCF'][station_pair].parameters
        for ii in range(tdata.shape[0]):
            cc_array[iseg] = tdata[ii]
            cc_time[iseg]  = ttime[ii]
            cc_ngood[iseg] = tgood[ii]
            iseg+=1
    t1=time.time()
    if flag:print('loading CCF data takes %6.2fs'%(t1-t0))

    # continue when there is no data
    if iseg <= 1: continue

    # do substacking if needed
    if f_substack:
        substacks,stime,num_stacks = noise_module.do_stacking(cc_array[:iseg],cc_time[:iseg],cc_ngood[:iseg],f_substack_len,stack_para)
        t2=time.time()
        if flag:print('finished substacking, which takes %6.2fs'%(t2-t1))
        
        if not len(substacks):print('continue! no substacks done!');continue

        if out_format=='asdf':
            stack_h5 = os.path.join(STACKDIR,idir+'/'+stack_method+'_'+outfn)
            with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
                for iii in range(substacks.shape[0]):
                    tparameters['time']  = stime[iii]
                    tparameters['ngood'] = num_stacks[iii]
                    tparameters['stack_method'] = stack_method
                    tpath     = ttr[2][-1]+ttr[6][-1]
                    data_type = 'T'+str(int(stime[iii]))
                    ds.add_auxiliary_data(data=substacks[iii], data_type=data_type, path=tpath, parameters=tparameters)
    
    # do all stacking
    t3=time.time()
    allstacks,alltime,num_stacks = noise_module.do_stacking(cc_array[:iseg],cc_time[:iseg],cc_ngood[:iseg],0,stack_para)
    t4=time.time()

    if out_format=='asdf':
        stack_h5 = os.path.join(STACKDIR,idir+'/'+stack_method+'_'+outfn)
        with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
            tparameters['time']  = alltime
            tparameters['ngood'] = num_stacks
            tparameters['stack_method'] = stack_method
            tpath     = ttr[2][-1]+ttr[6][-1]
            data_type = 'Allstack'
            ds.add_auxiliary_data(data=allstacks, data_type=data_type, path=tpath, parameters=tparameters)

    t5 = time.time()
    if flag:print('takes %6.2fs to process one chunck data, %6.2fs for all stacking' %(t5-t0,t4-t3))
        
comm.barrier()

if rotation:
    # do rotation now
    if rank == 0:
        sfiles = glob.glob(os.path.join(STACKDIR,'*/*.h5'))
        splits = len(sfiles)
    else:
        sfiles,splits = [None for _ in range(2)]

    # broadcast new variables
    splits    = comm.bcast(splits,root=0)
    sfiles    = comm.bcast(sfiles,root=0)

    # MPI loop through each user-defined time chunck
    for istack in range (rank,splits,size):
        noise_module.do_rotation(sfiles[istack],stack_para,locs,flag)

tt1 = time.time()
print('it takes %6.2fs to process step 2 in total' % (tt1-tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()