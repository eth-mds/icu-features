from pathlib import Path

import numpy as np
import pytest

from icu_features.load import features, load

DATA_DIR = Path(__file__).parents[1] / "tests" / "testdata"


@pytest.mark.parametrize(
    "outcome",
    ["mortality_at_24h", "circulatory_failure_at_8h", "respiratory_failure_at_24h"],
)
def test_eicu_demo(outcome):
    df, y, _ = load(["eicu_demo"], outcome=outcome, data_dir=DATA_DIR)
    assert np.isnan(y).sum() == 0
    assert len(df) == len(y) > 0
    assert sorted(df.columns) == sorted(features())
