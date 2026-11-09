import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def global_points():
    """Points uniform on the sphere above about -60 degrees latitude, as (lon, lat) in degrees."""

    def make(n=4000, seed=0):
        rng = np.random.default_rng(seed)
        lon = rng.uniform(-180, 180, n)
        lat = np.degrees(np.arcsin(rng.uniform(-0.87, 1.0, n)))
        return np.column_stack([lon, lat])

    return make
