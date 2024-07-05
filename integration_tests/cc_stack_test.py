import os
from datetime import datetime, timezone

import pytest
from datetimerange import DateTimeRange

from noisepy.seis import (  # noisepy core functions
    cross_correlate,
    stack_cross_correlations,
)
from noisepy.seis.io.channel_filter_store import channel_filter
from noisepy.seis.io.channelcatalog import (
    XMLStationChannelCatalog,  # Required stationXML handling object
)
from noisepy.seis.io.datatypes import (  # Main configuration object
    CCMethod,
    ConfigParameters,
    StackMethod,
)
from noisepy.seis.io.numpystore import NumpyCCStore, NumpyStackStore
from noisepy.seis.io.s3store import (  # Object to query SCEDC data from on S3
    SCEDCS3DataStore,
)

S3_STORAGE_OPTIONS = {"s3": {"anon": True}}
S3_DATA = "s3://scedc-pds/continuous_waveforms/"
S3_STATION_XML = "s3://scedc-pds/FDSNstationXML/CI/"  # S3 storage of stationXML


@pytest.mark.parametrize(
    "stack_method, substack, cc_method",
    [
        (StackMethod.ALL, False, CCMethod.XCORR),
        (StackMethod.LINEAR, True, CCMethod.DECONV),
        (StackMethod.LINEAR, True, CCMethod.COHERENCY),
        (StackMethod.LINEAR, True, CCMethod.XCORR),
    ],
)
def test_cc_stack(tmp_path, stack_method, substack, cc_method):
    path = str(tmp_path)

    cc_data_path = os.path.join(path, "CCF")
    stack_data_path = os.path.join(path, "STACK")

    config = ConfigParameters()  # default config parameters which can be customized
    config.start_date = datetime(2002, 1, 1, tzinfo=timezone.utc)
    config.end_date = datetime(2002, 1, 3, tzinfo=timezone.utc)
    config.stack_method = stack_method
    config.substack = substack
    config.keep_substack = substack
    config.cc_method = cc_method
    # timeframe for analysis
    timerange = DateTimeRange(config.start_date, config.end_date)

    networks = ["CI"]
    stations = "RPV,SVD".split(",")
    catalog = XMLStationChannelCatalog(S3_STATION_XML, storage_options=S3_STORAGE_OPTIONS)  # Station catalog
    raw_store = SCEDCS3DataStore(
        S3_DATA,
        catalog,
        channel_filter(networks, stations, ["BHE", "BHN", "BHZ"]),
        timerange,
        storage_options=S3_STORAGE_OPTIONS,
    )  # Store for reading raw data from S3 bucket
    cc_store = NumpyCCStore(cc_data_path)  # Store for writing CC data

    cross_correlate(raw_store, config, cc_store)

    cc_store = NumpyCCStore(cc_data_path, mode="r")
    stack_store = NumpyStackStore(stack_data_path)
    stack_cross_correlations(cc_store, stack_store, config)
