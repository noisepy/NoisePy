from noisepy.seis.datatypes import ConfigParameters

def test_stations_file_behavior():
    # Test loading from stations_file
    config = ConfigParameters(stations_file="../src/noisepy/seis/stations.txt")
    assert config.stations == []

    ConfigParameters.load_stations(config)

    assert config.stations  == ["RPV", "SVD","BBR"]

    new_stations = ['new_station1','new_station2']
    config.stations_file = "../src/noisepy/seis/stations1.txt"
    ConfigParameters.save_stations(config,new_stations)

    assert config.stations == []

