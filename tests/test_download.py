import os
import pathlib
import shutil
from unittest.mock import patch

import numpy as np
import pytest
from dateutil.parser import isoparse
from obspy import Stream, Trace
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from noisepy.seis.channelcatalog import CSVChannelCatalog
from noisepy.seis.datatypes import ConfigParameters
from noisepy.seis.fdsn_download import download, download_stream


@patch("noisepy.seis.fdsn_download.download_stream")
@patch.object(Client, "get_stations_bulk")
def test_download(get_stations_bulk_mock, download_stream_mock, tmp_path: pathlib.Path):
    cfg = ConfigParameters()
    cfg.net_list = ["CI"]
    cfg.stations = ["BAK"]
    cfg.start_date = isoparse("2021-01-01T00:00:00Z")
    cfg.end_date = isoparse("2021-01-01T00:01:00Z")

    # mock the download_stream function
    st = Stream(Trace(np.zeros(1000), header={"starttime": cfg.start_date, "sampling_rate": 10}))
    download_stream_mock.return_value = (0, st)
    csv_file = os.path.join(os.path.dirname(__file__), "./data/station.csv")
    catalog = CSVChannelCatalog(csv_file)
    # mock the get_stations_bulk function by returning and inventory from csv
    get_stations_bulk_mock.return_value = catalog.get_inventory(None, None)
    download(str(tmp_path), cfg)
    tmp_path.joinpath("station.csv").unlink()

    # Throws because of missing station.csv
    cfg.down_list = True
    with pytest.raises(IOError):
        download(str(tmp_path), cfg)

    # copy station.csv to tmp_path and try again
    shutil.copy(csv_file, str(tmp_path))
    download(str(tmp_path), cfg)

    files = list(tmp_path.glob("*.h5"))
    assert len(files) == 1


def test_download_erros():
    cfg = ConfigParameters()

    with patch.object(Client, "get_waveforms") as get_waveforms_mock:
        get_waveforms_mock.side_effect = FDSNNoDataException("no data")
        ista, st = download_stream(cfg, None, "", "", "", "", None, None, 0)
        assert ista == -1
        assert st is None
        get_waveforms_mock.side_effect = Exception("retriable")
        ista, st = download_stream(cfg, None, "", "", "", "", None, None, 0)
        assert ista == -1
        assert st is None
        assert get_waveforms_mock.call_count == 6  # 5 retries + 1
