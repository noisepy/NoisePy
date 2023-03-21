import sys
import numpy as np

'''
this script generates a large matrix to compare its
memory size with that estimated from mprof module
'''

# define matrix dimension
n1 = 10000
n2 = 25000

# get the random matrix
data = np.random.rand(n1,n2)
es = n1*n2*8/1024**3
ss = sys.getsizeof(data)/1024**3

# delay the time for accurate memory monitoring
tdata0 = np.zeros(shape=data.shape,dtype=data.dtype)
for ii in range(data.shape[0]):
    for jj in range(data.shape[1]):
        tdata0[ii,jj] = data[ii,jj]+0.1*data[ii,jj]

print('memory estimates are %5.3f %5.3f'%(es,ss))


# allocate a porportion of the data matrix
data = np.random.rand(n1,n2)
tdata1 = np.zeros(shape=data.shape,dtype=data.dtype)
for ii in range(0,int(0.5*n1)):
    for jj in range(0,int(0.5*n2)):
        tdata1[ii,jj] = data[ii,jj]+0.1*data[ii,jj]

ss = sys.getsizeof(tdata1)/1024**3
print('new memory estimates are %5.3f %5.3f'%(es,ss))
