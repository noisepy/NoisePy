import pyasdf

with pyasdf.ASDFDataSet("./asdf_example.h5", mode="r") as ds:
    # Get dictionary of resource_id -> Lat/Lng pairs
    events = {
        str(e.resource_id): [
            (e.preferred_origin() or e.origins[0]).get(i)
            for i in ["latitude", "longitude"]
        ]
        for e in ds.events
    }

    # Loop over all stations.
    for s in ds.waveforms:
        try:
            coords = s.coordinates
        except pyasdf.ASDFException:
            continue

        # Get set of all event ids.
        #
        # Get set for all event ids - the `get_waveform_attributes()`
        # method is fairly new. If you version of pyasdf does not yet
        # have it please update or use:
        # group = s._WaveformAccessor__hdf5_group
        # event_ids = list({group[i].attrs.get("event_id", None)
        #                   for i in s.list()})
        # event_ids = [i.decode() for i in event_ids if i]

        # Note that this assumes only one event id per waveform.
        event_ids = set(
            _i["event_ids"][0]
            for _i in s.get_waveform_attributes().values()
            if "event_ids" in _i
        )

        for e_id in event_ids:
            if e_id not in events:
                continue
            # Do what you want - this will be called once per src/rec pair.
            print(
                "%.2f %.2f %.2f %.2f"
                % (
                    events[e_id][0],
                    events[e_id][1],
                    coords["latitude"],
                    coords["longitude"],
                )
            )
