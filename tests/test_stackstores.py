from pathlib import Path

import numpy as np
import pytest

from noisepy.seis.asdfstore import ASDFStackStore
from noisepy.seis.datatypes import Station
from noisepy.seis.stores import StackStore
from noisepy.seis.zarrstore import ZarrStackStore


# Use the built in tmp_path fixture: https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html
# to create CC stores
@pytest.fixture
def asdfstore(tmp_path: Path) -> ASDFStackStore:
    return ASDFStackStore(str(tmp_path))


@pytest.fixture
def zarrstore(tmp_path: Path) -> ZarrStackStore:
    return ZarrStackStore(str(tmp_path))


def _stackstore_test_helper(store: StackStore):
    data = np.random.random(10)
    src = Station("nw", "sta1")
    rec = Station("nw", "sta2")
    comp = "BH"
    name = "Allstack_linear"
    params = {"key": "value"}

    # assert empty state
    assert not store.is_done(src, rec)

    # add CC (src->rec) for ts1
    store.append(src, rec, comp, "Allstack_linear", params, data)
    assert not store.is_done(src, rec)
    # now mark ts1 done and assert it
    store.mark_done(src, rec)
    assert store.is_done(src, rec)

    sta_pairs = store.get_station_pairs()
    assert sta_pairs == [(src, rec)]
    stacks = store.get_stack_names(src, rec)
    assert stacks == [name]
    components = store.get_components(src, rec, name)
    assert components == [comp]
    read_params, read_data = store.read(src, rec, comp, name)
    assert params == read_params
    assert np.all(data == read_data)


def test_asdfstore(asdfstore: ASDFStackStore):
    _stackstore_test_helper(asdfstore)


def test_zarrstore(zarrstore: ZarrStackStore):
    _stackstore_test_helper(zarrstore)
