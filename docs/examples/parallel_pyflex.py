import pyflex
from pyasdf import ASDFDataSet

ds = ASDFDataSet("./preprocessed_27s_to_60s.h5")
other_ds = ASDFDataSet("./preprocessed_synthetic_27s_to_60s.h5")

event = ds.events[0]


def weight_function(win):
    return win.max_cc_value


config = pyflex.Config(
    min_period=27.0,
    max_period=60.0,
    stalta_waterlevel=0.11,
    tshift_acceptance_level=15.0,
    dlna_acceptance_level=2.5,
    cc_acceptance_level=0.6,
    c_0=0.7,
    c_1=2.0,
    c_2=0.0,
    c_3a=1.0,
    c_3b=2.0,
    c_4a=3.0,
    c_4b=10.0,
    s2n_limit=0.5,
    max_time_before_first_arrival=-50.0,
    min_surface_wave_velocity=3.0,
    window_signal_to_noise_type="energy",
    window_weight_fct=weight_function,
)


def process(this_station_group, other_station_group):
    # Make sure everything thats required is there.
    if (
        not hasattr(this_station_group, "StationXML")
        or not hasattr(this_station_group, "preprocessed_27s_to_60s")
        or not hasattr(
            other_station_group, "preprocessed_synthetic_27s_to_60s"
        )
    ):
        return

    stationxml = this_station_group.StationXML
    observed = this_station_group.preprocessed_27s_to_60s
    synthetic = other_station_group.preprocessed_synthetic_27s_to_60s

    all_windows = []

    for component in ["Z", "R", "T"]:
        obs = observed.select(component=component)
        syn = synthetic.select(component=component)
        if not obs or not syn:
            continue

        windows = pyflex.select_windows(
            obs, syn, config, event=event, station=stationxml
        )
        print(
            "Station %s.%s component %s picked %i windows"
            % (
                stationxml[0].code,
                stationxml[0][0].code,
                component,
                len(windows),
            )
        )
        if not windows:
            continue
        all_windows.append(windows)
    return all_windows


import time

a = time.time()
results = ds.process_two_files_without_parallel_output(other_ds, process)
b = time.time()

if ds.mpi.rank == 0:
    print(results)
    print(len(results))

print("Time taken:", b - a)

# Important when running with MPI as it might otherwise not be able to finish.
del ds
del other_ds
