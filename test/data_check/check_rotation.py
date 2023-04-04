import numpy as np
from numpy import cos, sin

# script to check that the rotation is done correctly
az_test = 45 * np.pi / 180
baz_test = 225 * np.pi / 180
npts = 1
# rotation in seisnoise
# convert [EE,EN,NN,NE] into [TT,RR,TR,RT]
big_rot = np.asarray(
    [
        [
            -cos(az_test) * cos(baz_test),
            cos(az_test) * sin(baz_test),
            -sin(az_test) * sin(baz_test),
            sin(az_test) * cos(baz_test),
        ],
        [
            -sin(az_test) * sin(baz_test),
            -sin(az_test) * cos(baz_test),
            -cos(az_test) * cos(baz_test),
            -cos(az_test) * sin(baz_test),
        ],
        [
            -cos(az_test) * sin(baz_test),
            -cos(az_test) * cos(baz_test),
            sin(az_test) * cos(baz_test),
            sin(az_test) * sin(baz_test),
        ],
        [
            -sin(az_test) * cos(baz_test),
            sin(az_test) * sin(baz_test),
            cos(az_test) * sin(baz_test),
            -cos(az_test) * cos(baz_test),
        ],
    ]
)
az_rot = np.asarray([[cos(az_test), -sin(az_test)], [sin(az_test), cos(az_test)]])
baz_rot = np.asarray([[-cos(baz_test), -sin(baz_test)], [sin(baz_test), -cos(baz_test)]])
#
# crap=np.zeros(shape=(9,npts))
crap = np.asarray([1, 0, 0, 0])  # just an EE signal
A = np.matmul(big_rot, crap)
print(A)
# ZN,ZE =>ZR, ZT test: ZE=1, ZN=0
crap = np.asarray([1, 0])
A = np.matmul(crap, baz_rot)
print(A)
# A = np.matmul(crap,az_rot)
# print(A)
# NZ,EZ =>RZ, TZ test: EZ=1, NZ=0
crap = np.asarray([1, 0])
A = np.matmul(crap, baz_rot)
print(A)

# # the rotation in noisepy
cosa = np.cos(az_test)
sina = np.sin(az_test)
cosb = np.cos(baz_test)
sinb = np.sin(baz_test)
# ENZ2RTZ
# rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
crap = np.zeros(shape=(9, npts))
crap[0] = 1  # EE=1
tcorr = np.zeros(shape=(9, npts), dtype=np.float32)
tcorr[0] = -cosb * crap[7] - sinb * crap[6]
tcorr[1] = sinb * crap[7] - cosb * crap[6]
tcorr[2] = crap[8]
tcorr[3] = (
    -cosa * cosb * crap[4] - cosa * sinb * crap[3] - sina * cosb * crap[1] - sina * sinb * crap[0]
)
tcorr[4] = (
    cosa * sinb * crap[4] - cosa * cosb * crap[3] + sina * sinb * crap[1] - sina * cosb * crap[0]
)
tcorr[5] = cosa * crap[5] + sina * crap[2]
tcorr[6] = (
    sina * cosb * crap[4] + sina * sinb * crap[3] - cosa * cosb * crap[1] - cosa * sinb * crap[0]
)
tcorr[7] = (
    -sina * sinb * crap[4] + sina * cosb * crap[3] + cosa * sinb * crap[1] - cosa * cosb * crap[0]
)
tcorr[8] = -sina * crap[5] + cosa * crap[2]
print(tcorr)
print("YOU SHOULD DEBUG NOISEPY")
