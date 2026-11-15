import pytest

from radarwow.pipeline import ConfigError, Pipeline

BASE = {"run": {"name": "t", "start": "2022-05-05 07:00", "end": "2022-05-05 11:00"}, "radar": [{"site": "KTLX", "source": "nexrad"}]}


def test_examples_parse():
    import glob
    import os

    files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "examples", "*.toml"))
    assert len(files) >= 3
    for f in files:
        assert Pipeline.from_toml(f).describe()


@pytest.mark.parametrize("bad, msg", [
    ({"radar": [{"site": "KTLX", "source": "nexrd"}]}, "source"),
    ({"radar": [{"site": "KTLX", "source": "nexrad", "n_tilt": 3}]}, "unknown key"),
    ({"radar": [{"site": "KTLX", "source": "nexrad", "n_tilts": "3"}]}, "expected int"),
    ({"merge": {"method": "KED"}}, "needs a [gauges]"),
    ({"gauges": {"source": "hads"}}, "networks"),
    ({"gauges": {"source": "fmi"}, "merge": {"method": "kriging"}}, "method"),
    ({"run": {"name": "t", "start": "2022-05-05 11:00", "end": "2022-05-05 07:00"}}, "before"),
    ({"mosiac": {}}, "unknown section"),
])
def test_bad_configs_fail_early(bad, msg):
    with pytest.raises(ConfigError, match=msg.replace("[", r"\[")):
        Pipeline.from_dict({**BASE, **bad})
