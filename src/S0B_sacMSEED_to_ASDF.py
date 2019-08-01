import sys
import glob
import os,gc
import obspy
import time
import pyasdf
import numpy as np
import noise_module
import pandas as pd
from mpi4py import MPI

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
os.system('export HDF5_USE_FILE=FALSE')

'''
this script helps clean the sac/mseed files on your local machine for NoisePy package by using the function of preprocess_raw
in the noise module. so it is similar to the script of S0. 

by Chengxin Jiang, Marine Denolle (Jul.30.2019)

NOTE: In this script, the station of the same name but of different channels are takes as different stations
'''

#######################################################
################PARAMETER SECTION######################
#######################################################
tt0=time.time()

# paths and filenames
rootpath  = '/Users/chengxin/Documents/NoisePy_example/JAKARTA'         # absolute path for your project
RAWDATA   = os.path.join(rootpath,'MSEED')                              # dir where mseed files are stored
if not os.path.isdir(RAWDATA):
    raise ValueError('Abort! no path of %s exists'%RAWDATA)
DATADIR   = os.path.join(rootpath,'CLEANED_DATA')                       # dir where cleaned data in ASDF format are outputted
locations = os.path.join(rootpath,'station.lst')                        # station info including network,station,channel,latitude,longitude,elevation
if not os.path.isfile(locations): 
    raise ValueError('Abort! station info is needed for this script')
locs = pd.read_csv(locations)

# useful parameters for cleaning the data
input_fmt = 'mseed'                                                     # input file format between 'sac' and 'mseed' 
samp_freq = 20                                                          # targeted sampling rate
rm_resp   = False                                                       # False to not remove, True to remove, and select 'inv' to remove with inventory
respdir   = 'none'                                                      # output response directory (required if rm_resp is not 'inv')
freqmin   = 0.05                                                        # pre filtering frequency bandwidth
freqmax   = 5                                                           # note this cannot exceed Nquist freq
outform   = 'asdf'                                                      # output file formats
flag      = True                                                        # print intermediate variables and computing time

# station information
start_date = ['2013_10_10_0_0_0']                                       # start date of local data
end_date   = ['2013_10_30_0_0_0']                                       # end date of local data
inc_hours  = 24                                                         # sac/mseed file length for a continous recording

# parameters for later cross-correlations: to estimate memory needs
cc_len    = 3600                                                        # basic unit of data length for fft (s)
step      = 900                                                         # overlapping between each cc_len (s)
MAX_MEM   = 3.0                                                         # maximum memory allowed per core in GB

# assemble parameters for data pre-processing
prepro_para = {'rm_resp':rm_resp,'respdir':respdir,'freqmin':freqmin,'freqmax':freqmax,'samp_freq':samp_freq,'inc_hours':inc_hours,\
    'start_date':start_date,'end_date':end_date}
metadata = os.path.join(DATADIR,'download_info.txt') 

##########################################################
#################PROCESSING SECTION#######################
##########################################################

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
    # output parameter info
    fout = open(metadata,'w')
    fout.write(str(prepro_para));fout.close()

    # make directory
    if not os.path.isdir(DATADIR):os.mkdir(DATADIR)

    # assemble timestamp info
    allfiles = glob.glob(os.path.join(RAWDATA,'*/*'))             # make sure sac/mseed files can be found through this path
    nfiles   = len(allfiles)
    if not nfiles: raise ValueError('Abort! no data found in subdirectory of %s'%RAWDATA)
    all_stimes = np.zeros(shape=(nfiles,2),dtype=np.float)
    for ii in range(nfiles):
        tr = obspy.read(allfiles[ii])
        all_stimes[ii][0] = tr[0].stats.starttime-obspy.UTCDateTime(1970,1,1)
        all_stimes[ii][1] = tr[0].stats.endtime-obspy.UTCDateTime(1970,1,1)

    # all time chunck for output: loop for MPI
    all_chunck = noise_module.get_event_list(start_date[0],end_date[0],inc_hours)   
    splits     = len(all_chunck)-1
    if splits<1:raise ValueError('Abort! no chunck found between %s-%s with inc %s'%(start_date[0],end_date[0],inc_hours))
else:
    splits,all_chunck,all_stimes,allfiles = [None for _ in range(4)]

# broadcast the variables
splits     = comm.bcast(splits,root=0)
all_chunck = comm.bcast(all_chunck,root=0)
all_stimes = comm.bcast(all_stimes,root=0)
allfiles   = comm.bcast(allfiles,root=0)

# MPI: loop through each time-chunck
for ick in range(rank,splits,size):
    t0=time.time()

    # time window defining the time-chunck
    s1=obspy.UTCDateTime(all_chunck[ick])
    s2=obspy.UTCDateTime(all_chunck[ick+1]) 
    date_info = {'starttime':s1,'endtime':s2}
    time1=s1-obspy.UTCDateTime(1970,1,1)
    time2=s2-obspy.UTCDateTime(1970,1,1) 

    # find all data pieces within the time-chunck
    indx1 = np.where((all_stimes[:,0]>=time1) & (all_stimes[:,0]<time2))[0]
    indx2 = np.where((all_stimes[:,1]>=time1) & (all_stimes[:,1]<time2))[0]
    indx3 = np.concatenate((indx1,indx2))
    indx  = np.unique(indx3)
    if not len(indx): print('continue! no data found between %s-%s'%(s1,s2));continue

    # trim down the sac/mseed file list with time in time-chunck
    tfiles = []
    for ii in indx:
        tfiles.append(allfiles[ii])

    # loop through station
    nsta = len(locs)
    for ista in range(nsta):

        # the station info:
        station = locs.iloc[ista]['station']
        network = locs.iloc[ista]['network']
        comp    = locs.iloc[ista]['channel']
        if flag: print("working on station %s channel %s" % (station,comp)) 

        # find files for the station from the short list of tfiles
        ttfiles  = [ifile for ifile in tfiles if station in ifile]  
        tttfiles = [ifile for ifile in ttfiles if comp in ifile]

        source = obspy.Stream()
        for ifile in tttfiles:
            try:
                tr = obspy.read(ifile)
                for ttr in tr:
                    source.append(ttr)
            except Exception as inst:
                print(inst);continue
        
        # jump if no good data left
        if not len(source):continue

        # make inventory to save into ASDF file
        t1=time.time()
        inv1   = noise_module.stats2inv(source[0].stats,locs=locs)
        tr = noise_module.preprocess_raw(source,inv1,prepro_para,date_info)
        t2=time.time()
        if flag:print('pre-processing takes %6.2fs'%(t2-t1))

        # jump if no good data left
        if not len(tr):continue

        # ready for output
        ff=os.path.join(DATADIR,all_chunck[ick]+'T'+all_chunck[ick+1]+'.h5')
        with pyasdf.ASDFDataSet(ff,mpi=False,compression='gzip-3') as ds:
            # add the inventory for all components + all time of this tation         
            try:ds.add_stationxml(inv1) 
            except Exception: pass 

            tlocation = str('00')        
            new_tags = '{0:s}_{1:s}'.format(comp.lower(),tlocation.lower())
            ds.add_waveforms(tr,tag=new_tags)     
    
    t3=time.time()
    print('it takes '+str(t3-t0)+' s to process one station in step 0B')

tt1=time.time()
print('step1 takes '+str(tt1-tt0)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
