import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from icu_benchmarks.scripts.feature_engineering import (
    continuous_features,
    discrete_features,
    treatment_continuous_features,
    treatment_indicator_features,
)


@pytest.mark.parametrize(
    "feature, input, expected",
    [
        (
            "ffilled",
            [None, None, 1.0, 2.0, 3.0, None, None, None, None, None, None, 0.0, None],
            [None, None, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0],
        ),
        (
            "mean_h8",
            [None, None, 1.0, 2.0, 3.0, None, None, None, None, None, None, 0.0, 0.0],
            [None, None, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 1.5, 0.0],
        ),
        (
            "std_h8",
            [None, None, 1.0, 2.0, 3.0, None, None, None, None, None, None, 0.0, 0.0],
            [None, None, None, 0.5] + [np.sqrt(2 / 3)] * 6 + [0.5, 1.5, 0.0],
        ),
        (
            "slope_h8",
            [None, None, 1.0, 2.0, 3.0, None, None, None, None, None, None, 3.0, 0.0],
            [None, None, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -3.0],
        ),
        (
            "fraction_nonnull_h8",
            [None, None, 1.0, 2.0, 3.0, None, None, None, None, None, None, 3.0, 0.0],
            [
                0.0,
                0.0,
                1 / 3,
                1 / 2,
                3 / 5,
                1 / 2,
                3 / 7,
                3 / 8,
                3 / 8,
                3 / 8,
                2 / 8,
                2 / 8,
                2 / 8,
            ],
        ),
        (
            "all_missing_h8",
            [None, None, 1.0, 2.0, None, None, None, None, None, None, None, None, 0.0],
            [True, True] + [False] * 9 + [True, False],
        ),
        (
            "min_h8",
            [None, 1.0, 0.5, 2.0, 1.0, None, None, None, None, None, 3.0, 3.0, 0.0],
            [None, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0],
        ),
        (
            "max_h8",
            [None, 1.0, 0.5, 2.0, 1.0, None, None, None, None, None, 3.0, 3.0, 0.0],
            [None, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        ),
    ],
)
def test_continuous_features(feature, input, expected):
    features = continuous_features("feature", "time")
    name = f"feature_{feature}"
    expr = [e for e in features if e.meta.output_name() == name][0]

    df = pl.DataFrame({"feature": input, "time": range(len(input))})
    result = df.select(expr).to_series()

    assert_series_equal(result, pl.Series(expected), check_names=False)


@pytest.mark.parametrize(
    "feature, input, expected",
    [
        (
            "mode_h8",
            [
                None,
                None,
                "a",
                "b",
                "b",
                "A",
                None,
                "A",
                "b",
                None,
                None,
                None,
                "a",
                "a",
            ],
            pl.Series(
                [
                    "(MISSING)",
                    "(MISSING)",
                    "a",
                    "a",
                    "b",
                    "b",
                    "b",
                    "A",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "a",
                ],
                dtype=pl.Enum(["a", "A", "b", "(MISSING)"]),
            ),
        ),
        (
            "num_nonmissing_h8",
            [
                "(MISSING)",
                "(MISSING)",
                "a",
                "b",
                "b",
                "A",
                "(MISSING)",
                "A",
                "b",
                "(MISSING)",
                "(MISSING)",
                "(MISSING)",
                "a",
                "a",
            ],
            pl.Series([0, 0, 1, 2, 3, 4, 4, 5, 6, 6, 5, 4, 4, 4], dtype=pl.Float64),
        ),
    ],
)
def test_discrete_features(feature, input, expected):
    features = discrete_features("feature", "time")
    name = f"feature_{feature}"
    expr = [e for e in features if e.meta.output_name() == name][0]
    enum = pl.Enum(["a", "A", "b", "(MISSING)"])

    df = pl.DataFrame({"feature": input, "time": range(len(input))})
    df = df.with_columns(pl.col("feature").cast(enum))
    result = df.select(expr).to_series()

    assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize(
    "feature, input, expected",
    [
        (
            "num_nonmissing_h8",
            [
                None,
                None,
                True,
                None,
                True,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 0],
        ),
        (
            "any_nonmissing_h8",
            [
                None,
                None,
                True,
                None,
                True,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [False, False] + [True] * 10 + [False],
        ),
    ],
)
def test_treatment_indicator_features(feature, input, expected):
    features = treatment_indicator_features("feature", "time")
    name = f"feature_{feature}"
    expr = [e for e in features if e.meta.output_name() == name][0]

    df = pl.DataFrame({"feature": input, "time": range(len(input))})

    result = df.select(expr).to_series()

    assert_series_equal(
        result, pl.Series(expected), check_names=False, check_dtypes=False
    )


@pytest.mark.parametrize(
    "feature, input, log_eps, expected",
    [
        (
            "rate_h8",
            [1.0, 1.0, None, None, None, None, 1.0, 1.0],
            0.0,
            np.log([1.0, 1.0, 2 / 3, 0.5, 2 / 5, 2 / 6, 3 / 7, 0.5]),
        ),
        (
            "rate_h8",
            [1.0, 1.0, None, None, None, None, 1.0, 1.0],
            1.0,
            np.log([2.0, 2.0, 5 / 3, 1.5, 7 / 5, 8 / 6, 10 / 7, 1.5]),
        ),
    ],
)
def test_treatment_continuous_features(feature, input, log_eps, expected):
    features = treatment_continuous_features("feature", "time", log_eps=log_eps)
    name = f"log_feature_{feature}"
    expr = [e for e in features if e.meta.output_name() == name][0]

    df = pl.DataFrame({"feature": input, "time": range(len(input))})

    result = df.select(expr).to_series()

    assert_series_equal(
        result, pl.Series(expected), check_names=False, check_dtypes=False
    )
