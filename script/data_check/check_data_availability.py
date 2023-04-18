import glob
import os

import pandas as pd

"""
check the availability of data throughout the year
"""

###########################
####### STEP ONE ##########
###########################

# absolute path of the data
DATADIR = "/n/holylfs/EXTERNAL_REPOS/DENOLLE/KANTO/DATA/2012"
location = "/n/scratchssdlfs/denolle_lab/CCFs_Kanto/station.lst"
if not os.path.isfile(location):
    raise ValueError("Abort! station list not found")
locs = pd.read_csv(location)
sta = locs["station"]
comp = locs["channel"]
nsta = len(sta)

# number of events
events = glob.glob(os.path.join(DATADIR, "Event_*"))
neve = len(events)
print("%d stations and %d events in total" % (nsta, neve))

# loop through each station
for ii in range(nsta):
    tmp = "*" + sta[ii] + "*" + comp[ii] + "*.sac"
    allfiles = glob.glob(os.path.join(DATADIR + "/Event_*", tmp))

    print("%s %s %6.2f" % (sta[ii], comp[ii], len(allfiles) / neve))
