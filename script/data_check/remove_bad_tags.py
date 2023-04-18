import glob
import os

import pyasdf

"""
this script finds the station pairs with stacked cross-correaltion missing one or several components
"""

STACKDIR = "/Users/chengxin/Documents/Harvard/Kanto_basin/Mesonet_BW/STACK1"
ALLFILES = glob.glob(os.path.join(STACKDIR, "*/*.h5"))

nfiles = len(ALLFILES)
ncomp = 17  # expected component number including both E-N-Z and R-T-Z systems

for ii in range(nfiles):
    # -------reading mode to find the bad traces--------
    with pyasdf.ASDFDataSet(nfiles[ii], mode="r") as ds:
        slist = ds.auxiliary_data.list()

        badlist = []
        for data_type in slist:
            rlist = ds.auxiliary_data[data_type].list()

            if len(rlist) != ncomp:
                print("missing data @ %s" % data_type)
                badlist.append(data_type)

    with pyasdf.ASDFDataSet(nfiles[ii]) as ds:
        if badlist:
            for jj in range(len(badlist)):
                print("removing trace %s" % badlist[jj])
                del ds.auxiliary_data[badlist[jj]]
