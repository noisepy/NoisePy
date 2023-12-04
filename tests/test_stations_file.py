import os

from noisepy.seis.datatypes import ConfigParameters


def test_stations_file_behavior():
    # Test loading from stations_file
    config = ConfigParameters(stations_file=os.path.join(os.path.dirname(__file__), "./data/stations.txt"))
    assert config.stations == ["*"]

    config.load_stations()

    assert config.stations == ["RPV", "SVD", "BBR"]

    new_stations = ["new_station1", "new_station2"]
    config.stations_file = os.path.join(os.path.dirname(__file__), "./data/stations1.txt")
    ConfigParameters.save_stations(config, new_stations)
