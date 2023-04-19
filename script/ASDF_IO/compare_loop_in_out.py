import time

import numpy as np
import pyasdf

"""
check the efficiency of writing ASDF files in a for loop
Chengxin@Harvard (Jul/2/2019)
"""


def out_loop(outfn, nloop, shape):
    """
    open the asdf file outside of the for loop
    """
    data_type = "RAND"
    paramters = {"none": 1}
    with pyasdf.ASDFDataSet(outfn) as ds:
        for ii in range(nloop):
            a = np.random.rand(shape[0], shape[1])
            path = "test{0:d}".format(ii)
            ds.add_auxiliary_data(data=a, data_type=data_type, path=path, parameters=paramters)


def in_loop(outfn, nloop, shape):
    """
    open asdf file to write every time
    """
    data_type = "RAND"
    paramters = {"none": 1}
    for ii in range(nloop):
        a = np.random.rand(shape[0], shape[1])

        with pyasdf.ASDFDataSet(outfn) as ds:
            path = "test{0:d}".format(ii)
            ds.add_auxiliary_data(data=a, data_type=data_type, path=path, parameters=paramters)


def main():
    outfn1 = "test_out.h5"
    outfn2 = "test_in.h5"
    nloop = 10
    shape = [40, 72000]

    t0 = time.time()
    out_loop(outfn1, nloop, shape)
    t1 = time.time()
    in_loop(outfn2, nloop, shape)
    t2 = time.time()
    print("it takes %5.1f and %5.1f s" % ((t1 - t0), (t2 - t1)))


if __name__ == "__main__":
    main()
