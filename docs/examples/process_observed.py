import obspy
from obspy.core.util.geodetics import gps2DistAzimuth
import numpy as np

from pyasdf import ASDFDataSet

ds = ASDFDataSet("./observed.h5")

event = ds.events[0]

origin = event.preferred_origin() or event.origins[0]
event_latitude = origin.latitude
event_longitude = origin.longitude

# Figure out these parameters somehonw!
starttime = obspy.UTCDateTime("2010-03-11T06:22:19.021324Z")
npts = 5708
sampling_rate = 1.0


# Loop over both period sets. This will result in two files. It could also be
# saved to the same file.
for min_period, max_period in [(27.0, 60.0)]:
    f2 = 1.0 / max_period
    f3 = 1.0 / min_period
    f1 = 0.8 * f2
    f4 = 1.2 * f3
    pre_filt = (f1, f2, f3, f4)

    def process_function(st, inv):
        st.detrend("linear")
        st.detrend("demean")
        st.taper(max_percentage=0.05, type="hann")

        st.attach_response(inv)
        st.remove_response(
            output="DISP", pre_filt=pre_filt, zero_mean=False, taper=False
        )

        st.detrend("linear")
        st.detrend("demean")
        st.taper(max_percentage=0.05, type="hann")

        st.interpolate(
            sampling_rate=sampling_rate, starttime=starttime, npts=npts
        )

        station_latitude = inv[0][0].latitude
        station_longitude = inv[0][0].longitude
        _, baz, _ = gps2DistAzimuth(
            station_latitude,
            station_longitude,
            event_latitude,
            event_longitude,
        )

        components = [tr.stats.channel[-1] for tr in st]
        if "N" in components and "E" in components:
            st.rotate(method="NE->RT", back_azimuth=baz)

        # Convert to single precision to save space.
        for tr in st:
            tr.data = np.require(tr.data, dtype="float32")

        return st

    tag_name = "preprocessed_%is_to_%is" % (int(min_period), int(max_period))

    tag_map = {"raw_recording": tag_name}

    ds.process(process_function, tag_name + ".h5", tag_map=tag_map)

# Important when running with MPI as it might otherwise not be able to finish.
del ds
