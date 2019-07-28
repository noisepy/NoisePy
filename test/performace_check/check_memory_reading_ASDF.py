import pyasdf
import np 

def read_asdf(tfile):

    with pyasdf.ASDFDataSet(tfile,MPI=False,mode='r') as ds:
        data_types = ds.auxiliary_data.list()
        path = ds.auxiliary_data[data_types[0]].list()
        Nfft = ds.auxiliary_data[data_types[0]][path[0]].parameters['nfft']
        Nseg = ds.auxiliary_data[data_types[0]][path[0]].parameters['nseg']
    ncomp = len(data_types)
    nsta  = len(path)

    memory_size = nsta*ncomp*Nfft//2*Nseg*8/1024/1024/1024

    cc_array = np.zeros((nsta*ncomp,Nseg*Nfft//2),dtype=np.complex64)
    cc_std   = np.zeros((nsta*ncomp))
