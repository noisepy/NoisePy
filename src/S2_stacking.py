import sys
import time
import obspy
import pyasdf
import os, glob
import datetime
import numpy as np
import noise_module
from mpi4py import MPI

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
Stacking script of NoisePy:
    1) read the saved cross-correlation data to do sub-stacks (if needed) and all-time average;
    2) save the outputs in ASDF or SAC format based on user's choice.

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@fas.harvard.edu)
        
Note:
'''

tt0=time.time()

########################################
#########PARAMETER SECTION##############
########################################

# absolute path parameters
rootpath  = '/Users/chengxin/Documents/Harvard/NoisePy/v4.0_July'                       # root path for this data processing
CCFDIR    = os.path.join(rootpath,'CCF')                # dir where CC data is stored
STACKDIR  = os.path.join(rootpath,'STACK') 

# assemble path information used to read CC data (stored in ASDF files)
pfiles    = glob.glob(CCFDIR,'paths_*.lst')

# load fc_para from S1
fc_metadata = os.path.join(rootpath,'fft_cc_data.txt')
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
f_substack = True                                       # whether to do sub-stacking (different from that in S1)
f_substack_len = 10*cc_len                              # length for sub-stacking to output
out_format = 'ASDF'                                     # ASDF or SAC format for output
flag       = True                                       # output intermediate args for debugging

# maximum memory allowed per core in GB
MAX_MEM = 4.0

# make a dictionary to store all variables: also for later cc
stack_para={'samp_freq':samp_freq,'dt':dt,'cc_len':cc_len,'step':step,'rootpath':rootpath,\
    'STACKDIR':STACKDIR,'start_date':start_date[0],'end_date':end_date[0],'inc_hours':inc_hours,\
    'substack':substack,'substack_len':substack_len,'maxlag':maxlag,'MAX_MEM':MAX_MEM,\
    'f_substack':f_substack,'f_substack_len':f_substack_len}
# save fft metadata for future reference
stack_metadata  = os.path.join(rootpath,'stack_data.txt') 

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
    # all station-pair info saved in ASDF
    paths_all = noise_module.load_pfiles(pfiles)
    splits  = len(paths_all)
    if len(ccfiles)==0 or splits:
        raise IOError('Abort! no available CCF data for stacking')

    # make directories for storing stacked data
    for ii in range(paths_all):
        tr   = paths_all.split('s')
        tdir = tr[0]+'.'+tr[1]+'.'+tr[2]+'.'+tr[3]
        if not os.path.isdir(tdir):os.mkdir(tdir)
else:
    splits,ccfiles,paths_all = [None for _ in range(3)]

# broadcast the variables
splits    = comm.bcast(splits,root=0)
ccfiles   = comm.bcast(ccfiles,root=0)
paths_all = comm.bcast(paths_all,root=0)
extra = splits % size

# MPI loop: loop through each user-defined time chunck
for ipath in range (rank,splits+size-extra,size):
    if ipath<splits:
        t10=time.time()

        # source folder
        ttr = paths_all[ipath].split('s')
        idir = ttr[0]+'.'+ttr[1]+'.'+ttr[2]+'.'+ttr[3]

        # crude estimation on memory needs (assume float32)
        num_chunck  = len(ccfiles)
        num_segmts  = int(np.round(inc_hours/substack_len/3600))
        npts_segmt  = int(substack_len*samp_freq)
        memory_size = num_chunck*num_segmts*npts_segmt*4/1024**3
        if memory_size > MAX_MEM:
            raise ValueError('Require %s G memory (%s GB provided)! Reduce inc_hours as it cannot load %s h all once!' % (memory_size,MAX_MEM,inc_hours))
            
        # open array to store fft data/info in memory
        cc_array = np.zeros((num_chunck*num_segmts,npts_segmt),dtype=np.float32)
        cc_time  = np.zeros(num_chunck*num_segmts,dtype=np.int16)

        # loop through all time-chuncks
        iseg = 0
        station_pair = paths_all[ipath]
        for ifile in range(len(ccfiles)):
            ds=pyasdf.ASDFDataSet(ccfiles[ifile],mode='r')
            path_list = ds.auxiliary_data['CCF'].list()            
            if station_pair not in path_list:
                if flag:print('continue! no data in %s'%ccfiles[ifile]);continue
    
            # load the data by segments
            tdata = ds.auxiliary_data['CCF'][station_pair].data
            ttime = ds.auxiliary_data['CCF'][station_pair].parameters['time']
            for ii in range(tdata.shape[0]):
                cc_array[iseg] = tdata[ii,:]
                cc_time[iseg]  = ttime[ii]
                iseg+=1

        # do substacking if needed
        if f_substack:
            substacks,stime = noise_module.do_stacking(cc_array,cc_time,f_substack_len)
            if rotation:
                noise_module.rotation(substacks)
        
        # do all stacking
        allstacks = noise_module.do_stacking(cc_array,cc_time,0)
        if rotation:
            noise_module.rotation(allstacks)
        
        # output the files
        if out_format=='ASDF':
            xxx
        elif out_format == 'SAC':
            xxx 

tt1 = time.time()
print('it takes %6.4fs to process step 2 of stacking' % (tt1-tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()
            