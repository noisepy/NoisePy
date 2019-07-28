import obspy
import glob
import os
import matplotlib.pyplot as plt 

event = '/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/noise_data/Event_2010_???'
cc_len = 3600
step   = 1800
day_dir = glob.glob(event)

#----loop through each day----
for iday in range(len(day_dir)):
    sacfiles = glob.glob(os.path.join(day_dir[iday],'*ATDH*.sac'))

    for ista in range(len(sacfiles)):
        sacfile = sacfiles[ista]
        comp = sacfile.split('.')[3]
        sta  = sacfile.split('.')[1]

        #-----start to read files-----
        source1 = obspy.read(sacfile)
        dt=1/source1[0].stats.sampling_rate

        #------------Pre-Processing-----------
        source = obspy.Stream()
        source = source1.merge(method=1,fill_value=0.)[0]

        for ii,win in enumerate(source.slide(window_length=cc_len, step=step)):
            win.detrend(type="constant")
            win.detrend(type="linear")
            win.taper(max_percentage=0.05,max_length=20)
            plt.plot(win)
            plt.title([sta,comp,ii])
            plt.show()