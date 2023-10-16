from pathlib import Path

import numpy as np
import pytest
from utils import date_range

from noisepy.seis.asdfstore import ASDFStackStore
from noisepy.seis.datatypes import Stack, Station
from noisepy.seis.numpystore import NumpyStackStore
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


@pytest.fixture
def numpystore(tmp_path: Path) -> NumpyStackStore:
    return NumpyStackStore(str(tmp_path))


def _stackstore_test_helper(store: StackStore):
    src = Station("nw", "sta1")
    rec = Station("nw", "sta2")
    ts = date_range(4, 1, 2)

    stack1 = Stack("EE", "Allstack_linear", {"key1": "value1"}, np.random.random(10))
    stack2 = Stack("NZ", "Allstack_robust", {"key2": "value2"}, np.random.random(7))
    stacks = [stack1, stack2]
    store.append(ts, src, rec, stacks)

    sta_pairs = store.get_station_pairs()
    assert sta_pairs == [(src, rec)]
    read_stacks = store.read(ts, src, rec)
    assert len(read_stacks) == len(stacks)
    for s1, s2 in zip(read_stacks, stacks):
        assert s1.name == s2.name
        assert s1.component == s2.component
        assert s1.parameters == s2.parameters
        assert s1.data.shape == s2.data.shape
        assert np.all(s1.data == s2.data)

    bad_read = store.read(ts, Station("nw", "sta3"), rec)
    assert len(bad_read) == 0

    sta_stacks = store.read_bulk(ts, [(src, rec)])
    assert len(sta_stacks) == 1
    assert sta_stacks[0][0] == (src, rec)
    assert len(sta_stacks[0][1]) == len(stacks)


def test_asdfstore(asdfstore: ASDFStackStore):
    _stackstore_test_helper(asdfstore)


def test_zarrstore(zarrstore: ZarrStackStore):
    _stackstore_test_helper(zarrstore)


def test_numpystore(numpystore: NumpyStackStore):
    _stackstore_test_helper(numpystore)
