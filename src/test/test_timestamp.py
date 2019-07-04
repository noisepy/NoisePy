from datetime import datetime
import numpy as np 
import obspy

# 
dataS_t = []
dataS_t.append(obspy.UTCDateTime(2010,10,1,0,0)-obspy.UTCDateTime(1970,1,1,0,0))
dataS_t.append(obspy.UTCDateTime(2010,10,2,0,0)-obspy.UTCDateTime(1970,1,1,0,0))

# Timestamps should be exactly the same with dataS_t
Timestamps = []
Timestamps.append(datetime.fromtimestamp(dataS_t[0]))
Timestamps.append(datetime.fromtimestamp(dataS_t[1]))
