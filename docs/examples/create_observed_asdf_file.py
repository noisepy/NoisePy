import glob
import os

from pyasdf import ASDFDataSet

filename = "observed.h5"

if os.path.exists(filename):
    raise Exception("File '%s' exists." % filename)

ds = ASDFDataSet(filename)

# Add event
ds.add_quakeml(
    "./GCMT_event_SOUTH_SANDWICH_ISLANDS_REGION_Mag_5.6_2010-3-11-6.xml"
)
event = ds.events[0]

# Add waveforms.
filenames = glob.glob("./SAC/*.SAC")
for _i, filename in enumerate(filenames):
    print("Adding SAC file %i of %i..." % (_i + 1, len(filenames)))
    # We associate the waveform with the previous event. This is optional
    # but recommended if the association is meaningful.
    ds.add_waveforms(filename, tag="raw_recording", event_id=event)

# Add StationXML files.
filenames = glob.glob("./StationXML/*.xml")
for _i, filename in enumerate(filenames):
    print("Adding StationXML file %i of %i..." % (_i + 1, len(filenames)))
    ds.add_stationxml(filename)
