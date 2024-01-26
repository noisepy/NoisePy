import os
from datetime import datetime, timezone

import pytest
from datetimerange import DateTimeRange

from noisepy.seis.scedc_s3store import NCEDCS3DataStore

timespan1 = DateTimeRange(datetime(2022, 1, 2, tzinfo=timezone.utc), datetime(2022, 1, 3, tzinfo=timezone.utc))
timespan2 = DateTimeRange(datetime(2021, 2, 3, tzinfo=timezone.utc), datetime(2021, 2, 4, tzinfo=timezone.utc))
timespan3 = DateTimeRange(datetime(2023, 6, 1, tzinfo=timezone.utc), datetime(2023, 6, 2, tzinfo=timezone.utc))
files_dates = [
    ("AFD.NC.HHZ..D.2022.002", timespan1),
    ("KCPB.NC.HHN..D.2021.034", timespan2),
    ("LMC.NC.HHN..D.2023.152", timespan3),
]


@pytest.mark.parametrize("file,expected", files_dates)
def test_parsefilename2(file: str, expected: DateTimeRange):
    assert expected == NCEDCS3DataStore._parse_timespan(None, file)


data_paths = [
    (os.path.join(os.path.dirname(__file__), "./data/s3ncedc"), None),
    ("s3://ncedc-pds/continuous_waveforms/NC/2022/2022.002/", None),
    ("s3://ncedc-pds/continuous_waveforms/", timespan1),
]


read_channels = [
    (NCEDCS3DataStore._parse_channel(None, "AFD.NC.HHZ..D.2022.002")),
    (NCEDCS3DataStore._parse_channel(None, "NSP.NC.EHZ..D.2022.002")),
    (NCEDCS3DataStore._parse_channel(None, "PSN.NC.EHZ..D.2022.002")),
]
