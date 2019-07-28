import numpy as np 
import pyasdf 
import obspy

'''
check the way how tags are generated for the ASDF waveform attribute
Chengxin @Harvard (Jul/02/2019)
'''

def write_waveform(outfn):

    for ii in range(5):
        tr = obspy.read()

        with pyasdf.ASDFDataSet(outfn) as ds:
            tags = 'test_'+str(ii)
            ds.add_waveforms(tr,tag=tags)

