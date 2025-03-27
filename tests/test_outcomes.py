import re

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from icu_features.feature_engineering import eep_label, outcomes, polars_nan_or


def to_bool(x):
    return pl.Series(x.split(" ")).replace("-", None).cast(pl.Int32).cast(pl.Boolean)


@pytest.mark.parametrize(
    "events, expected, horizon, switches_only",
    [
        (
            "- 1 - - 0 0 - - - - - 0 0 0 - - 1 - 1 - 0 0 0 - 1 0 1",
            "1 - 0 0 0 - - 0 0 0 0 0 1 1 1 1 - - - 0 1 1 1 1 - 1 -",
            4,
            True,
        ),
        (
            "- 1 - - 0 0 - - - - - 0 0 0 - - 1 - 1 - 0 0 0 - 1 0 1",
            "1 - 0 0 0 - - 0 0 0 0 0 1 1 1 1 - 1 - 0 1 1 1 1 - 1 -",
            4,
            False,
        ),
    ],
)
def test_eep_labels(events, expected, horizon, switches_only):
    df = pl.DataFrame(
        {
            "events": to_bool(events),
            "expected": to_bool(expected),
        }
    ).with_columns(eep_label(pl.col("events"), horizon, switches_only).alias("labels"))
    assert_series_equal(df["labels"], df["expected"], check_names=False)


@pytest.mark.parametrize(
    "args, expected", [(("0 0 0 - - 1", "0 0 0 - - -", "0 1 - 1 0 0"), "0 1 - 1 - 1")]
)
def test_polars_nan_or(args, expected):
    df = pl.DataFrame({str(i): to_bool(x) for i, x in enumerate(args)}).with_columns(
        polars_nan_or(*[pl.col(str(i)) for i in range(len(args))]).alias("result")
    )
    assert_series_equal(df["result"], to_bool(expected), check_names=False)


@pytest.mark.parametrize(
    "outcome_name, input, expected",
    [
        (
            "mortality_at_24h",
            pl.DataFrame(
                {
                    "death_icu": True,
                    "time_hours": np.arange(0, 36),
                }
            ),
            pl.Series("mortality_at_24h", [None] * 24 + [True] + [None] * 11),
        ),
        (
            "mortality_at_24h",
            pl.DataFrame(
                {
                    "death_icu": False,
                    "time_hours": np.arange(0, 36),
                }
            ),
            pl.Series([None] * 24 + [False] + [None] * 11),
        ),
        (
            "decompensation_at_24h",
            pl.DataFrame(
                {
                    "death_icu": True,
                    "time_hours": np.arange(0, 36),
                }
            ),
            pl.Series([False] * 12 + [True] * 24),
        ),
        (
            "decompensation_at_24h",
            pl.DataFrame(
                {
                    "death_icu": False,
                    "time_hours": np.arange(0, 36),
                }
            ),
            pl.Series([False] * 36),
        ),
        (
            "remaining_los",
            pl.DataFrame(
                {
                    "los_icu": 4.0,
                    "time_hours": np.arange(0, 4 * 24),
                }
            ),
            pl.Series(np.arange(4 * 24, 0, -1) / 24),
        ),
        (
            "log_lactate_in_4h",
            pl.DataFrame(
                {
                    "lact": 4 * [None]
                    + [
                        np.exp(-1.0) - 0.1,
                        np.exp(2.1) - 0.1,
                        None,
                        None,
                        np.exp(1.2) - 0.1,
                        np.exp(0.0) - 0.1,
                        None,
                    ],
                    "time_hours": np.arange(0, 11),
                }
            ),
            pl.Series([-1.0, 2.1, None, None, 1.2, 0.0] + 5 * [None]),
        ),
    ],
)
def test_outcomes(outcome_name, input, expected):
    expr = [e for e in outcomes() if e.meta.output_name() == outcome_name][0]
    assert_series_equal(
        input.with_columns(expr).select(outcome_name).to_series(),
        expected.rename(outcome_name),
    )


@pytest.mark.parametrize(
    "outcome_name, input, expected_events, expected_labels",
    [
        (
            "respiratory_failure_at_24h",
            pl.DataFrame(
                {
                    "pf_ratio": [None] * 36 + [100] * 12 + [None] * 24 + [500] * 12,
                }
            ),
            pl.Series([None] * 36 + [True] * 12 + [None] * 24 + [False] * 12),
            pl.Series([None] * 12 + [True] * 24 + [None] * 12 + [False] * 35 + [None]),
        ),
        (
            "circulatory_failure_at_8h",
            pl.DataFrame(
                {
                    "dobu_ind": None,  # these are either None or True
                    "levo_ind": None,
                    "norepi_ind": None,
                    "epi_ind": None,
                    "milrin_ind": None,
                    "teophyllin_ind": None,
                    "dopa_ind": None,
                    "adh_ind": None,
                    "map": [70] * 4 + [None] * 4 + [70] * 4 + [60] * 4 + [60] * 4,
                    "lact": [1] * 4 + [1] * 4 + [1] * 4 + [1] * 4 + [3] * 4,
                }
            ),
            # events (in blocks of 4): - - 0 - 1
            pl.Series([None] * 8 + [False] * 4 + [None] * 4 + [True] * 4),
            pl.Series([False] * 8 + [True] * 8 + [None] * 4),
        ),
        (
            "circulatory_failure_at_8h",
            pl.DataFrame(
                {
                    "dobu_ind": None,
                    "levo_ind": None,
                    "norepi_ind": None,
                    "epi_ind": None,
                    "milrin_ind": None,
                    "teophyllin_ind": None,
                    "dopa_ind": [None] * 14 + [True] * 6,
                    "adh_ind": [None] * 12 + [True] * 4 + [None] * 4,
                    # events (in blocks of 4): 0 - 0 - 1
                    "map": [70] * 4 + [60] * 4 + [70] * 12,
                    "lact": [1] * 4 + [1] * 4 + [1] * 4 + [1] * 4 + [3] * 4,
                }
            ),
            pl.Series([False] * 4 + [None] * 4 + [False] * 4 + [None] * 4 + [True] * 4),
            pl.Series([False] * 8 + [True] * 8 + [None] * 4),
        ),
        (
            "kidney_failure_at_48h",
            pl.DataFrame(
                {
                    "crea": [1.0] * 24 + [None] * 72 + [2.0] * 72 + [4.0] * 48,
                    "rel_urine_rate": 1,
                    "ufilt_ind": [False] * 216,
                }
            ),
            pl.Series(
                [False] * 24 + [None] * 72 + [False] * 72 + [True] * 24 + [False] * 24
            ),
            pl.Series(
                [False] * 23
                + [None] * 25
                + [False] * 72
                + [True] * 48
                + [None] * 24
                + [False] * 23
                + [None]
            ),
        ),
    ],
)
def test_eep_outcomes(outcome_name, input, expected_events, expected_labels):
    expr = [e for e in outcomes() if e.meta.output_name() == outcome_name][0]

    assert_series_equal(
        input.with_columns(expr).select(outcome_name).to_series(),
        expected_labels.rename(outcome_name),
    )

    horizon = int(re.search(r"\d+", outcome_name).group(0))
    input = input.insert_column(0, expected_events.rename("expected_events"))
    assert_series_equal(
        input.select(eep_label(pl.col("expected_events"), horizon)).to_series(),
        input.with_columns(expr).select(outcome_name).to_series(),
        check_names=False,
    )
