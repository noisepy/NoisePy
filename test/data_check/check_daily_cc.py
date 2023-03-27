import matplotlib.pyplot as plt
import numpy as np
import pyasdf

CCFile = "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_opt"
maxlag = 800
dt = 0.1
tt = np.arange(-maxlag,maxlag+dt,dt)

ds = pyasdf.ASDFDataSet(CCFile+'/2010_01_01.h5',mode='r')
if ds:
    listS = ds.auxiliary_data.list()
    for isource in listS:
        listR = ds.auxiliary_data[isource].list()

        for ireceiver in listR:
            plt.plot(tt,ds.auxiliary_data[isource][ireceiver].data[:],'r',linewidth=1)
            plt.show()
del ds
