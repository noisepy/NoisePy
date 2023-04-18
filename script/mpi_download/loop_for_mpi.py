import numpy as np
from mpi4py import MPI

"""
a script to test configuration of MPI loop
"""


def test1():
    data = np.random.rand(10)

    # --------MPI---------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        splits = len(data)
        print(data)
    else:
        splits = [None for _ in range(1)]

    splits = comm.bcast(splits, root=0)

    for ii in range(rank, splits, size):
        tdata = data[ii]
        print("index %d rank %d and data %f" % (ii, rank, tdata))


def test2():
    # --------MPI---------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data = np.random.rand(11)
        splits = len(data)
        print(data)
    else:
        splits, data = [None for _ in range(2)]

    splits = comm.bcast(splits, root=0)
    data = comm.bcast(data, root=0)

    for ii in range(rank, splits, size):
        tdata = data[ii]
        print("index %d rank %d and data %f" % (ii, rank, tdata))

    comm.barrier()


def test3():
    # --------MPI---------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data = np.random.rand(11)
        splits = len(data)
        print(data)
    else:
        splits, data = [None for _ in range(2)]

    splits = comm.bcast(splits, root=0)
    data = comm.bcast(data, root=0)
    extra = splits % size

    for ii in range(rank, splits + size - extra, size):
        if ii < splits:
            tdata = data[ii]
            print("index %d rank %d and data %f" % (ii, rank, tdata))


def main():
    print("start test1")
    test1()
    print("now for test2")
    test2()
    print("good for test3")
    test3()


if __name__ == "__main__":
    main()
