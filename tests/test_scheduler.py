from noisepy.seis.scheduler import SingleNodeScheduler


def test_single_node_scheduler_get_indices():
    single_node_scheduler = SingleNodeScheduler()
    items = ["a", "b", "c", "d"]
    expected_indices = [0, 1, 2, 3]
    indices = single_node_scheduler.get_indices(items)

    assert indices == expected_indices


def test_synchronize():
    scheduler = SingleNodeScheduler()
    result = scheduler.synchronize()

    assert result is None
