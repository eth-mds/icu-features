import logging
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

VARIABLE_REFERENCE_PATH = Path(__file__).parents[1] / "resources" / "variables.tsv"
HORIZONS = [8, 24, 72]
CAT_MISSING_NAME = "(MISSING)"

variables = pl.read_csv(VARIABLE_REFERENCE_PATH, separator="\t", null_values=["None"])


def switch(col, bounds, values):
    """
    Map intervals to values.

    Returns values[idx] if bounds[idx] <= col < bounds[idx + 1].
    """
    if len(bounds) != len(values) + 1:
        raise ValueError(f"Length mismatch between bounds and values for {col}.")

    if isinstance(col, str):
        col = pl.col(col)

    return pl.coalesce(
        pl.when(col.gt(lw1) & col.le(lw2)).then(val)
        for lw1, lw2, val in zip(bounds[:-1], bounds[1:], values)
    ).fill_null(0)


def additional_variables():
    """Compute variables that are not exported from ricu."""
    # Fio2 is often missing if a patient is not ventilated. If the patient is not
    # ventilated and was not ventilated in the last hour, we fill fio2 with 21% (ambient
    # air).
    vent_ind = pl.col("vent_ind") | pl.col("vent_ind").shift(1, fill_value=False)
    fio2 = (pl.col("fio2") / 100.0).fill_null(pl.when(~vent_ind).then(0.21))
    pf_ratio = pl.col("po2") / fio2

    # APACHE II from
    # https://icucycle.com/wp-content/uploads/2018/06/PROSPECT-APACHE-II-Calculation
    # -Worksheet.pdf
    # and
    # https://icucycle.com/wp-content/uploads/2018/06/PROSPECT-RCT-APACHE-SOP-July-21-
    # 2016.pdf
    # We don't use hco3 and chronic health points.
    # The apache score is (at least in theory) only valid for time point 24:
    # > All APACHE II data collected must be from the first 24 hours following ICU
    # > admission.
    # We fill with 0 if no data is available:
    # > When recording variables for the Acute Physiology Score, if a physiologic
    # > measurement is not obtained during the 24 hour time frame, assign a zero (“0”)
    # > point score.
    # We take a max over the last 24 hours for physiological measurements:
    # > For all acute physiologic measurements: choose the worst, most abnormal value
    # > recorded during the full 24 hour assessment period.
    # The maximal value apache_ii can take is 10 * 4 + (15 - 3) + 6 = 58.
    # The minimal value is 0.
    fio2_ffilled = pl.col("fio2").forward_fill(4).fill_null(21)
    pco2_ffilled = pl.col("pco2").forward_fill(4)
    po2_ffilled = pl.col("po2").forward_fill(4)
    # https://en.wikipedia.org/wiki/Alveolar%E2%80%93arterial_gradient
    # Here, 7.13 = (P_atm - P_H2O) / 100% and 1.25 = 1 / 0.8.
    aado2 = fio2_ffilled * 7.13 - pco2_ffilled * 1.25 - po2_ffilled

    apache_ii_scores = [
        switch(
            "temp", [0, 30, 32, 34, 36, 38.5, 39, 41, 100], [4, 3, 2, 1, 0, 1, 3, 4]
        ),
        switch("map", [0, 50, 70, 110, 130, 160, 1e4], [4, 2, 0, 2, 3, 4]),
        switch("hr", [0, 40, 55, 70, 110, 140, 180, 500], [4, 3, 2, 0, 2, 3, 4]),
        switch("resp", [0, 6, 10, 12, 25, 35, 50, 100], [4, 2, 1, 0, 1, 3, 4]),
        pl.when(fio2_ffilled.lt(50))
        .then(switch("po2", [np.inf, 70, 60, 55, 0], [0, 1, 3, 4]))
        .otherwise(switch(aado2, [np.inf, 500, 350, 200, 0], [4, 3, 2, 0])),
        switch(
            "na",
            [np.inf, 180, 160, 155, 150, 130, 120, 110, 0],
            [4, 3, 2, 1, 0, 2, 3, 4],
        ),
        switch("k", [np.inf, 7, 6, 5.5, 3.5, 3, 2.5, 0], [4, 3, 1, 0, 1, 2, 4]),
        # the thresholds for creatinine are in umol/l, crea is in mg/dl
        switch(pl.col("crea") * 88.42, [np.inf, 305, 170, 130, 53, 0], [4, 3, 2, 0, 2]),
        switch("hct", [np.inf, 60, 50, 46, 30, 20, 0], [4, 2, 1, 0, 2, 4]),
        # wbc is in "Giga/l", the same as "total / mm3".
        switch("wbc", [np.inf, 40, 20, 15, 3, 1, 0], [4, 2, 1, 0, 2, 4]),
        15 - pl.col("tgcs").fill_null(3),
    ]
    apache_ii = pl.sum_horizontal(
        x.rolling_max(24, min_samples=1, center=False) for x in apache_ii_scores
    )
    apache_ii += switch("age", [0, 45, 55, 65, 75, np.inf], [0, 2, 3, 5, 6])

    return [
        pf_ratio.alias("pf_ratio"),
        apache_ii.alias("apache_ii"),
    ]


def continuous_features(
    column_name: str, time_col: str, horizons: list[int] | None = None
):
    """
    Compute continuous features for a column.

    These are:
    - ffilled: The column with missing values filled forward.
    - missing: True if the value is missing.
    - mean: The mean of the column within the last `horizon` hours. This is nan if there
      are only missing values within the last `horizon` hours.
    - std: The standard deviation of the column within the last `horizon` hours. This is
      nan if there are less than 2 non-missing values within the last `horizon` hours.
    - slope: The slope of the column within the last `horizon` hours. Uses hours as the
      x-axis. This is nan if there are less than 2 non-missing values within the last
      `horizon` hours.
    - fraction_nonnull: The fraction of non-missing values within the last `horizon`.
    - all_missing: True if all values within the last `horizon` are missing.
    - min: The minimum of the column within the last `horizon` hours.
    - max: The maximum of the column within the last `horizon` hours.

    Parameters
    ----------
    column_name : str
        Name of column for which to compute features. E.g., `hr`.
    time_col : str
        Name of the time column. E.g., `time_hours`.
    horizons : list[int] or None
        Horizons for which to compute the features. If None, uses constants.HORIZONS.
    """
    if horizons is None:
        horizons = HORIZONS

    column = pl.col(column_name)
    # Most features can be computed using cumulative sums. To "ignore" missing values,
    # we fill them with 0.
    col_filled = column.fill_null(0)
    # time is never null. But we want to ignore entries corresponding to missing values
    # in `column_name`.
    time = pl.when(column.is_null()).then(0).otherwise(pl.col(time_col))
    nonnulls = column.is_not_null().cast(pl.Int32).cum_sum()

    time_cs = time.cum_sum()
    time_sq_cs = (time * time).cum_sum()

    col_cs = col_filled.cum_sum()
    col_sq_cs = (col_filled * col_filled).cum_sum()
    timexcol_cs = (time * col_filled).cum_sum()

    expressions = [
        column.forward_fill().alias(f"{column_name}_ffilled"),
        column.pow(2).forward_fill().alias(f"{column_name}_sq_ffilled"),
        column.is_null().alias(f"{column_name}_missing"),
    ]

    for horizon in horizons:
        window_size = (pl.col(time_col) + 1).clip(None, horizon)
        col_sum = col_cs - col_cs.shift(horizon, fill_value=0)
        col_sq_sum = col_sq_cs - col_sq_cs.shift(horizon, fill_value=0)
        time_sum = time_cs - time_cs.shift(horizon, fill_value=0)
        time_sq_sum = time_sq_cs - time_sq_cs.shift(horizon, fill_value=0)
        timexcol_sum = timexcol_cs - timexcol_cs.shift(horizon, fill_value=0)
        nonnulls_sum = nonnulls - nonnulls.shift(horizon, fill_value=0)

        # Use when to avoid division by zero & s.t. column contains Nones, not nans.
        col_mean = pl.when(nonnulls_sum > 0).then(col_sum / nonnulls_sum)
        col_sq_mean = pl.when(nonnulls_sum > 0).then(col_sq_sum / nonnulls_sum)

        # std = sum_i (x_i - mean)^2 / n = sum_i x_i^2 / n - mean^2
        col_std = (col_sq_sum / nonnulls_sum - col_mean * col_mean).clip(0, None).sqrt()
        col_std = pl.when(nonnulls_sum >= 2).then(col_std)

        # See https://en.wikipedia.org/wiki/Simple_linear_regression#Expanded_formulas
        numerator = nonnulls_sum * timexcol_sum - time_sum * col_sum
        denom = nonnulls_sum * time_sq_sum - time_sum * time_sum
        slope = pl.when(nonnulls_sum >= 2).then(numerator / denom)

        fraction_nonnull = nonnulls_sum / window_size

        all_missing = nonnulls_sum == 0

        rolling_min = column.rolling_min(window_size=horizon, min_samples=1)
        rolling_max = column.rolling_max(window_size=horizon, min_samples=1)

        expressions += [
            col_mean.alias(f"{column_name}_mean_h{horizon}"),
            col_sq_mean.alias(f"{column_name}_sq_mean_h{horizon}"),
            col_std.alias(f"{column_name}_std_h{horizon}"),
            slope.alias(f"{column_name}_slope_h{horizon}"),
            fraction_nonnull.alias(f"{column_name}_fraction_nonnull_h{horizon}"),
            all_missing.alias(f"{column_name}_all_missing_h{horizon}"),
            rolling_min.alias(f"{column_name}_min_h{horizon}"),
            rolling_max.alias(f"{column_name}_max_h{horizon}"),
        ]

    return expressions


def discrete_features(
    column_name: str, time_col: str, horizons: list[int] | None = None
):
    """
    Compute discrete features for a column.

    These are:
    - The last value (no suffix).
    - mode: The mode of the column within the last `horizon` hours. Ignores missing
      values.
    - num_nonmissing: The number of non-missing values within the last `horizon` hours.

    Parameters
    ----------
    column_name : str
        Name of column for which to compute features. E.g., `hr`.
    time_col : str
        Name of the time column. E.g., `time_hours`.
    horizons : list[int] or None
        Horizons for which to compute the features. If None, uses constants.HORIZONS.
    """
    if horizons is None:
        horizons = HORIZONS

    def get_rolling_mode(series, horizon):
        """Compute rolling mode for a series of (value, time) tuples."""
        # If the column contains only missings, `mode` returns an empty list. Else, it
        # returns a list with the mode(s) of the column. This list has multiple entries
        # if there's ties. We take the first entry (after sorting) as the mode.
        return (
            pl.select(col=series[0], _time=series[1])
            # Use period="xi" (index-based), as polars does not support time-based with
            # the period column a `Duration`. It only supports this for "absolute" time.
            .rolling(index_column="_time", period=f"{horizon}i")
            .agg(pl.col("col").drop_nulls().mode())
            .select(
                pl.when(pl.col("col").list.len() >= 1)
                .then(pl.col("col").list.sort().list.first())
                .fill_null(CAT_MISSING_NAME)
            )
            .to_series()
        )

    column = pl.col(column_name)
    nonnulls = (column.ne(CAT_MISSING_NAME)).cum_sum()

    expressions = [column]

    for horizon in horizons:
        col_mode = pl.map_groups(
            exprs=(column, pl.col(time_col)),
            function=lambda series: get_rolling_mode(series, horizon),
            return_dtype=pl.List(pl.String),
        )
        nonnulls_sum = nonnulls - nonnulls.shift(horizon, fill_value=0).cast(pl.Float64)

        expressions += [
            col_mode.alias(f"{column_name}_mode_h{horizon}"),
            nonnulls_sum.alias(f"{column_name}_num_nonmissing_h{horizon}"),
        ]

    return expressions


def treatment_indicator_features(
    column_name: str, time_col: str, horizons: list[int] | None = None
):
    """
    Compute features for a treatment indicator column.

    These are:
    - The last value (no suffix).
    - num: The number of "True" values within the last `horizon` hours.
    - any: True if there is a "True" value within the last `horizon`
    hours.

    Parameters
    ----------
    column_name : str
        Name of column for which to compute features. E.g., `hep_ind`.
    time_col : str
        Not used.
    horizons : list[int] or None
        Horizons for which to compute the features. If None, uses constants.HORIZONS.
    """
    if horizons is None:
        horizons = HORIZONS

    col = pl.col(column_name).fill_null(False)
    col_cs = col.cast(pl.Int32).cum_sum()

    expressions = [col]

    for horizon in horizons:
        col_sum = (col_cs - col_cs.shift(horizon, fill_value=0)).cast(pl.Float64)
        expressions += [
            col_sum.alias(f"{column_name}_num_h{horizon}"),
            (col_sum > 0).alias(f"{column_name}_any_h{horizon}"),
        ]
    return expressions


def treatment_continuous_features(
    column_name: str,
    time_col: str,
    horizons: list[int] | None = None,
    log_transform: bool = True,
    log_eps: float = 0.0,
):
    """
    Compute features for a variable with continuous treatment values.

    These are:
    - rate: The log of the mean of the column within the last `horizon` hours after
      imputing with zero.

    Parameters
    ----------
    column_name : str
        Name of column for which to compute features. E.g., `hr`.
    time_col : str
        Not used.
    horizons : list[int] or None
        Horizons for which to compute the features. If None, uses constants.HORIZONS.
    log_transform : bool, optional, default = True
        Whether to log transform the mean (rate).
    log_eps : float, optional, default = 0.0
        Epsilon to add before taking the log. This is to avoid taking the log of 0.
    """
    if horizons is None:
        horizons = HORIZONS

    col_cs = pl.col(column_name).fill_null(0.0).cum_sum()
    expressions = list()

    for horizon in horizons:
        col_sum = col_cs - col_cs.shift(horizon, fill_value=0)
        col_mean = col_sum / (pl.col(time_col) + 1).clip(None, horizon)
        # This ensures that zeros get mapped to zeros.
        if log_transform:
            expressions += [
                (col_mean + log_eps).log().alias(f"log_{column_name}_rate_h{horizon}"),
            ]
        else:
            expressions += [col_mean.alias(f"{column_name}_rate_h{horizon}")]

    return expressions


def eep_label(events: pl.Expr, horizon: int):
    """
    From an event series, create a label for the early event prediction (eep) task.

     - If there is a positive event at the current time step, the label is missing.
     - If the last event in the history was positive, and there is a positive event
       within the next `horizon` hours, the label is missing.
     - If there was no event in the history or the last event in the history was
       negative, and there is a positive event within the next `horizon` hours, the
       label is true. This holds even if there is a negative event at the current time
       step.
     - Else, if there is a negative event within the next `horizon` hours (and no
       positive event within the next `horizon` hours or at the current time step), the
       label is false.

    The "next `horizon` hours" exclude the current time step.

    E.g., if `-` are missings, for a 4 hour horizon:

    events:           - 1 - - 0 0 - - - - - 0 0 0 - - 1 - 1 - 0 0 0 - 1 0 1
    events_ffilled:   0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1
    positive_labels:  1 1 - - - - - - - - - - 1 1 1 1 1 1 1 - 1 1 1 1 1 1 1
    negative_labels:  - 0 0 0 0 0 - 0 0 0 0 0 0 0 - - 0 0 0 0 0 0 0 0 0 0 -
    coalesced_labels: 1 1 0 0 0 0 - 0 0 0 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1
    label:            1 - 0 0 0 - - 0 0 0 0 0 1 1 1 1 - - - 0 1 1 1 1 - 1 -

    Note that at the time step of a positive event, the label is always missing. At the
    time step of a negative event, the label could be true, false, or missing.

    This definition ensures that only "switches" from stable to unstable are predicted.

    Parameters
    ----------
    events : pl.Expr
        An expression for an event series. Boolean with possibly missing values.
    horizon : int
        The horizon for the early event prediction task.
    """
    events_ffilled = events.forward_fill().replace(None, False)

    positive_labels = events.replace(False, None).backward_fill(horizon)
    # shift(-1) and backward_fill(horizon - 1) excludes the last zero.
    negative_labels = events.replace(True, None).shift(-1).backward_fill(horizon - 1)

    coalesced_label = pl.coalesce(positive_labels, negative_labels)
    return pl.when(coalesced_label.eq(False) | events_ffilled.eq(False)).then(
        coalesced_label
    )


def polars_nan_or(*args: pl.Expr):
    """
    Nan or operation for polars Series.

    If any of the arguments is nonmissing and positive, return True. If all arguments
    are nonmissing and negative, return False. Else, return None.

    Examples
    --------
    >>> import polars as pl
    >>> a = pl.Series("a", [1, 2, None])
    >>> b = pl.Series("b", [0, None, 2])
    >>> polars_nan_or(a < 0, b == 2)
    [ False, None, True ]
    """
    return (
        pl.when(pl.max_horizontal(*args))  # This ignores nans
        .then(True)
        .when(pl.all_horizontal([a.is_not_null() for a in args]))
        .then(False)
    )


def outcomes():
    """
    Compute outcomes.

    These are:
    - mortality_at_24h: A single label at time 24h after entry to the ICU whether the
      patient dies in the ICU. THis is a "once per patient" prediction task.
    - decompensation_at_24h: Whether the patient decompensates within the next 24 hours.
      This has label is true if the patient dies within the next 24 hours. Else, this is
      false. This does not have missing values.
    - respiratory_failure_at_24h: Whether the patient has a respiratory failure within
      the next 24 hours. If the PaO2/FiO2 ratio is below 200, the patient is considered
      to have a respiratory failure (event).
    - remaining_los: The remaining length of stay in the ICU. Missing for the last time-
      step (instead of zero).
    - circulatory_failure_at_8h: Whether the patient has a circulatory failure within
      the next 8 hours. Circulatory failure is defined via blood pressure and lactate.
      Blood pressure is considered low if the mean arterial pressure is below 65 mmHg or if
      the patient is on any blood pressure increasing drug. Lactate is high if it is
      above 2 mmol/l. If the two criteria (map and lactate) don't agree, or if one of them is
      missing, the event label is missing.
    - kidney_failure_at_48h: Whether the patient has a kidney failure within the next 48
      hours. The patient has a kidney failure if they are in stage 3 according to the
      KDIGO guidelines:
      https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf
    - los_at_24h: The length of stay in the ICU at 24 hours after entry.
    - log_creatine_in_1h: The log of the creatinine value in 1 hour.
    - log_lactate_in_1h: The log of the lactate value in 1 hour.
    - log_po2: The logarithm of the PaO2 value.
    """
    # mortality_at_24h
    # This is a "once per patient" prediction task. At time step 24h, a label is
    # assigned. All other timesteps are missing.
    mortality_at_24h = (
        pl.when(pl.col("time_hours").eq(24))
        .then(pl.col("death_icu").first().fill_null(False))
        .alias("mortality_at_24h")
    )

    # decompensation_at_24h
    # This has label is true if the patient dies within the next 24 hours. Else, this
    # is false.
    decompensation_at_24h = (
        # Need <24 (or <=23) s.t. we get 24x the label `death_icu`.
        pl.when(pl.col("time_hours").max() - pl.col("time_hours") < 24)
        .then(pl.col("death_icu"))
        .otherwise(False)
    ).alias("decompensation_at_24h")

    # respiratory_failure_at_24h
    # If the PaO2/FiO2 ratio is below 200, the patient is considered to have a
    # respiratory failure (event). This used pf_ratio from other_variables().
    RESP_PF_DEF_TSH = 200.0
    events = pl.col("pf_ratio") < RESP_PF_DEF_TSH
    resp_failure_at_24h = eep_label(events, 24).alias("respiratory_failure_at_24h")

    # remaining_los
    remaining_los = pl.col("los_icu") - pl.col("time_hours") / 24.0
    remaining_los = pl.when(remaining_los > 0).then(remaining_los).otherwise(None)
    remaining_los = remaining_los.alias("remaining_los")

    # circulatory_failure_at_8h
    # A patient is considered to have a circulatory failure if the mean arterial
    # is low (below 65, or being raised by a drug) and the lactate is high (above 2).
    # If the two criteria (map and lactate) don't agree, or if one of them is missing,
    # the event label is missing.
    circulatory_drug_indicators = [  # These raise the map
        "dobu_ind",  # Dobutamine
        "levo_ind",  # Levosimendan
        "norepi_ind",  # Norepinephrine
        "epi_ind",  # Epinephrine
        "milrin_ind",  # Milrinone
        "teophyllin_ind",  # Teophylline
        "dopa_ind",  # Dopamine
        "adh_ind",  # Vasopressin
    ]
    drugs_indicator = (
        pl.sum_horizontal(
            [pl.col(c).fill_null(False) for c in circulatory_drug_indicators]
        )
        > 0
    )
    # Bad map is True if the mean arterial pressure is below 65 or if the patient is on
    # any map increasing drug. If the mean arterial pressure is above 65 and the patient
    # is not on any map increasing drug, bad_map is False. If the patient does not have
    # a map value and is not on any map increasing drug, bad_map is None.
    LOW_MAP_TSH = 65.0
    low_map = pl.col("map") <= LOW_MAP_TSH
    bad_map = pl.coalesce(drugs_indicator.replace(False, None), low_map)

    HIGH_LACT_TSH = 2.0
    bad_lact = pl.col("lact") >= HIGH_LACT_TSH
    # An event occurs if the map is "bad" (low or on drugs) and the lactate is high.
    # If the map is "good" (not low and not on drugs) and the lactate is not high, the
    # event label is negative. If the map and lact don't agree, or if one of them is
    # missing, the event label is missing.
    event = (
        pl.when(bad_map & bad_lact).then(True).when(~bad_map & ~bad_lact).then(False)
    )
    circulatory_failure_at_8h = eep_label(event, 8).alias("circulatory_failure_at_8h")

    # kidney_failure_at_48h
    # The patient has a kidney failure if they are in stage 3 according to
    # https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf
    relative_creatine = pl.col("crea") / pl.col("crea").shift(1).rolling_min(
        window_size=7 * 24, min_samples=1
    )

    # AKI 1 is
    # - max absolute creatinine increase of 0.3 within 48h or
    # - a relative creatinine increase of 1.5.
    creatine_min_48 = pl.col("crea").rolling_min(window_size=48, min_samples=1)
    creatine_max_48 = pl.col("crea").rolling_max(window_size=48, min_samples=1)
    creatine_change_48 = creatine_max_48 - creatine_min_48
    aki_1 = polars_nan_or(creatine_change_48 >= 0.3, relative_creatine >= 1.5)

    # AKI 3 is any of
    # - a relative creatine increase of 3.0 x baseline
    # - AKI 1 and creatinine >= 4.0
    # - not more than 0.3ml/kg/h urine rate for 24h
    # - no urine for 12h
    low_urine_rate = ((pl.col("urine_rate") / pl.col("weight")) < 0.3).cast(pl.Int32)
    low_urine_rate_24 = low_urine_rate.rolling_sum(window_size=24, min_samples=1).eq(24)

    # True if aki_1 is True and creatine >= 4 (neither missing). False if either aki_1
    # is False or creatine < 4. Else, missing.
    high_creatine = ~polars_nan_or(~aki_1, ~pl.col("crea").gt(4))
    # True if the urine_rate is consistently equal to 0 for 12 hours. False if it is
    # ever above 0. If all values are missing, the result is missing.
    anuria = (
        pl.col("urine_rate")
        .eq(0)
        .cast(pl.Int32)
        .rolling_sum(window_size=12, min_samples=1)
        .eq(12)
    )

    aki_3 = polars_nan_or(
        relative_creatine >= 3.0,
        high_creatine,
        low_urine_rate_24,
        anuria,
    )
    # If the weight is missing, the patient could only ever have a positive label, as
    # urine related conditions are always missing. We thus set the label to missing.
    aki_3 = pl.when(pl.col("weight").is_null()).then(None).otherwise(aki_3)
    kidney_failure_at_48h = eep_label(aki_3, 48).alias("kidney_failure_at_48h")

    # total length of stay, predicted at 24h after entry to the ICU.
    los_at_24h = pl.when(pl.col("time_hours").eq(24)).then(pl.col("los_icu"))
    los_at_24h = pl.when(los_at_24h > 0.1).then(los_at_24h).alias("los_at_24h")
    log_los_at_24h = los_at_24h.log().alias("log_los_at_24h")

    # log(lactate) in 4 hours. This is 1/2 the forecast horizon of circ. failure eep.
    log_lactate_in_4h = (
        (pl.col("lact") + 0.1).log().shift(-4).alias("log_lactate_in_4h")
    )

    log_pf_ratio_in_12h = (
        pl.col("pf_ratio").log().shift(-12).alias("log_pf_ratio_in_12h")
    )

    # The "raw" ICU data contains urine measurements from a bag at specific times (ml).
    # In `ricu`, we divide these measurement values by the time distance to the last
    # measurement. This gives us a urine rate (ml/h). This divided by the patient's
    # weight is the relative urine rate (ml/h/kg). These "relative rate" measurements
    # are only non-missing at the timepoint of the measurement.
    # We assign a label if there is a (positive) measurement in 2 hours.
    log_rel_urine_rate_in_2h = (
        pl.when(
            pl.col("rel_urine_rate").is_not_null() & pl.col("rel_urine_rate").ge(0.01)
        )
        .then(pl.col("rel_urine_rate").log())
        .shift(-2)
        .alias("log_rel_urine_rate_in_2h")
    )

    return [
        mortality_at_24h,
        decompensation_at_24h,
        resp_failure_at_24h,
        remaining_los,
        circulatory_failure_at_8h,
        kidney_failure_at_48h,
        los_at_24h,
        log_los_at_24h,
        log_lactate_in_4h,
        log_rel_urine_rate_in_2h,
        log_pf_ratio_in_12h,
    ]


@click.command()
@click.option("--dataset", type=str, required=True)
@click.option("--data_dir", type=str, default=None)
def main(dataset: str, data_dir: str | Path):  # noqa D
    logger.info(f"dataset: {dataset}")
    data_dir = Path(data_dir)

    dyn = pl.scan_parquet(data_dir / dataset / "dyn.parquet")
    sta = pl.scan_parquet(data_dir / dataset / "sta.parquet")

    if "patient_id" not in sta.collect_schema().names():
        sta = sta.with_columns(pl.col("stay_id").alias("patient_id"))

    dyn = dyn.join(sta, on="stay_id", how="full", coalesce=True, validate="m:1")
    dyn = dyn.with_columns(
        (pl.col("time").dt.total_hours()).cast(pl.Int32).alias("time_hours")
    )

    # The code below assumes that all missing values are encoded as nulls, not nans.
    # nans behave differently in comparisons (1.0 == nan is False, not None)
    tic = perf_counter()
    nan_columns = dyn.select(pl.col([pl.Float32, pl.Float64]).is_nan().any()).collect()
    nan_columns = [col.name for col in nan_columns if col.any()]
    if nan_columns:
        raise ValueError(f"The following columns contain nans: {nan_columns}")
    toc = perf_counter()
    logger.info(f"Time to check for nans: {toc - tic:.2f}s")

    # Lazy polars.DataFrame.upsample for time_column with dtype int.
    # The result is a DataFrame with a row for each hour between 0 and the maximum
    # time in the dataset for each stay_id.
    time_ranges = (
        dyn.group_by("stay_id")
        .agg(pl.col("time_hours").max().alias("max_time"))
        .with_columns(
            pl.col("max_time")
            .map_elements(
                # +1 to include max_time
                lambda x: pl.int_range(0, x + 1, eager=True, dtype=pl.Int32),
                return_dtype=pl.List(pl.Int32),
            )
            .alias("time_hours")
        )
        .drop("max_time")
        .explode("time_hours")
    )

    # equivalent to a pandas outer join
    dyn = dyn.join(time_ranges, on=["stay_id", "time_hours"], how="full", coalesce=True)
    dyn = dyn.sort(["stay_id", "time_hours"])

    expressions = ["time_hours", "anchoryear", "carevue", "metavision", "patient_id"]

    for row in variables.rows(named=True):
        tag = row["VariableTag"]
        col = pl.col(tag).clip(row["LowerBound"], row["UpperBound"])

        # Log transform some of the continuous variables. We add a small epsilon to
        # avoid taking the log of 0.
        if row["LogTransform"] and row["DataType"] != "treatment_cont":
            tag = f"log_{row['VariableTag']}"
            eps = row["LogTransformEps"] or 0.0
            col = (col + eps).log().replace([np.nan, -np.inf], None).alias(tag)

        if row["VariableType"] == "static":
            expressions.append(col)

        # Cast categorical variables to Enum. `samp` is binary. For simplicity, we cast
        # it to string first.
        elif row["DataType"] == "categorical":
            col = col.cast(pl.String).fill_null(CAT_MISSING_NAME)
            expressions += discrete_features(tag, "time_hours")

        elif row["DataType"] == "continuous":
            expressions += continuous_features(tag, "time_hours")

        elif row["DataType"] == "treatment_ind":
            expressions += treatment_indicator_features(tag, "time_hours")

        elif row["DataType"] == "treatment_cont":
            expressions += treatment_continuous_features(
                tag,
                "time_hours",
                log_eps=row["LogTransformEps"],
                log_transform=row["LogTransform"],
            )
        else:
            raise ValueError(f"Unknown DataType: {row['DataType']}")

        dyn = dyn.with_columns(col)

    expressions += outcomes()

    q = dyn.group_by("stay_id").agg(expressions).explode(pl.exclude("stay_id"))

    # These hashes are useful for subsetting. .hash() returns a u64-int. By normalizing
    # with 2**64, we get a pseudo-random float between 0 and 1.
    q = q.with_columns(
        (pl.col("stay_id").hash() / 2.0**64).alias("stay_id_hash"),
        (pl.col("patient_id").hash() / 2.0**64).alias("patient_id_hash"),
    ).with_columns(
        pl.when(pl.col("patient_id_hash") < 0.7)
        .then(pl.lit("train"))
        .when(pl.col("patient_id_hash") < 0.85)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
        .cast(pl.Enum(["train", "val", "test"])),
        pl.lit(dataset).alias("dataset"),
        pl.col("time_hours").log1p().alias("log_time_hours"),
    )

    tic = perf_counter()
    out = q.collect()
    toc = perf_counter()
    logger.info(f"Time to compute features: {toc - tic:.2f}s")
    logger.info(f"out.shape: {out.shape}")

    tic = perf_counter()
    out.write_parquet(data_dir / dataset / "features.parquet")
    toc = perf_counter()
    logger.info(f"Time to write features: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
