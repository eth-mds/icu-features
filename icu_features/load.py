from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.dataset as ds
from pyarrow.parquet import ParquetDataset

from icu_features.constants import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    HORIZONS,
    TREATMENT_CONTINUOUS_FEATURES,
    TREATMENT_INDICATOR_FEATURES,
    VARIABLE_REFERENCE_PATH,
)


def load(
    sources: list[str],
    outcome: str,
    data_dir: str | Path,
    split: str | None = None,
    variables: list[str] | None = None,
    categorical_features: list[str] | None = None,
    continuous_features: list[str] | None = None,
    treatment_indicator_features: list[str] | None = None,
    treatment_continuous_features: list[str] | None = None,
    horizons=None,
    other_columns: list[str] | None = None,
):
    """
    Load data as a polars DataFrame and a numpy array.

    This automatically subsets `X` and `y` based on the split and the `outcome` variable
    being non-null. The numpy arrays are contiguous.

    Parameters
    ----------
    sources : list of str
        The sources to load data from. E.g., ['eicu', 'mimic', 'sic'].
    outcome : str
        The outcome variable. E.g., `"mortality_at_24h"`.
    data_dir : str or pathlib.Path
        The directory containing the data.
    split : str, optional, default = None
        Either `"train"`, `"val"`, `"train_val"`, or `"test"`. If `None`, all data is
        loaded.
    variables : list of str, optional, default = None
        For which variables (e.g., `hr`) to load features. If `None`, all variables from
        the variable reference are loaded.
    categorical_features : list of str, optional, default = None
        Which categorical features to load. E.g., `"mode"`. If `None`, all categorical
        features `icu_features.constants.CATEGORICAL_FEATURES` are loaded.
    continuous_features : list of str, optional, default = None
        Which continuous features to load. E.g., `"mean"`. If `None`, all continuous
        features `icu_features.constants.CONTINUOUS_FEATURES` are loaded.
    treatment_indicator_features : list of str, optional, default = None
        Which treatment indicator features to load. E.g., `"num"`. If `None`, all
        treatment indicator features
        `icu_features.constants.TREATMENT_INDICATOR_FEATURES` are loaded.
    treatment_continuous_features : list of str, optional, default = None
        Which features of continuous treatments to load. E.g., `"rate"`. If `None`, all
        continuous treatment features
        `icu_features.constants.TREATMENT_CONTINUOUS_FEATURES` are loaded.
    horizons : list of int, optional, default = None
        The horizons for which to load features. If `None`, all horizons
        `icu_benchmarks.constants.HORIZONS` are loaded.
    other_columns : list of str, optional, default = None
        Other columns to load. E.g., `["stay_id_hash"]`.

    Returns
    -------
    df : polars.DataFrame
        The features.
    y : numpy.ndarray
        The outcome.
    * other_columns : * numpy.ndarray
    """
    if horizons is None:
        horizons = HORIZONS

    filters = ~ds.field(outcome).is_null()

    if split == "train":
        filters &= ds.field("patient_id_hash") < 0.7
    elif split == "val":
        filters &= (ds.field("patient_id_hash") >= 0.7) & (
            ds.field("patient_id_hash") < 0.85
        )
    elif split == "test":
        filters &= ds.field("patient_id_hash") >= 0.85
    elif split == "train_val":
        filters &= ds.field("patient_id_hash") < 0.85
    elif split is not None:
        raise ValueError(f"Invalid split: {split}")

    columns = features(
        variables=variables,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        treatment_indicator_features=treatment_indicator_features,
        treatment_continuous_features=treatment_continuous_features,
        horizons=horizons,
    )

    if other_columns is None:
        other_columns = []

    columns_to_load = columns + [outcome, "dataset", "stay_id_hash"]
    if "time_hours" not in columns:
        columns_to_load += ["time_hours"]

    columns_to_load += [c for c in other_columns if c not in columns_to_load]

    df = pl.from_arrow(
        ParquetDataset(
            [Path(data_dir) / source / "features.parquet" for source in sources],
            filters=filters,
        ).read(columns=columns_to_load)
    ).sort(["dataset", "stay_id_hash", "time_hours"])

    y = df[outcome].to_numpy()
    assert np.isnan(y).sum() == 0

    return (df.select(columns), y) + tuple(
        df.select(c).to_series() for c in other_columns
    )


def features(
    variables=None,
    categorical_features=None,
    continuous_features=None,
    treatment_indicator_features=None,
    treatment_continuous_features=None,
    horizons=None,
):
    """
    Get variable-feature names.

    Parameters
    ----------
    variables : list of str, optional, default = None
        For which variables (e.g., `hr`) to load features. If `None`, all variables from
        the variable reference are loaded.
    categorical_features : list of str, optional, default = None
        Which categorical features to load. E.g., `"mode"`. If `None`, all categorical
        features `icu_features.constants.CATEGORICAL_FEATURES` are loaded.
    continuous_features : list of str, optional, default = None
        Which continuous features to load. E.g., `"mean"`. If `None`, all continuous
        features `icu_features.constants.CONTINUOUS_FEATURES` are loaded.
    treatment_indicator_features : list of str, optional, default = None
        Which treatment indicator features to load. E.g., `"num"`. If `None`, all
        treatment indicator features
        `icu_features.constants.TREATMENT_INDICATOR_FEATURES` are loaded.
    treatment_continuous_features : list of str, optional, default = None
        Which features of continuous treatments to load. E.g., `"rate"`. If `None`, all
        continuous treatment features
        `icu_features.constants.TREATMENT_CONTINUOUS_FEATURES` are loaded.
    horizons : list of int, optional, default = None
        The horizons for which to load features. If `None`, all horizons
        `icu_benchmarks.constants.HORIZONS` are loaded.

    Returns
    -------
    features : list of str
        The feature names.
    """
    if continuous_features is None:
        continuous_features = CONTINUOUS_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if treatment_indicator_features is None:
        treatment_indicator_features = TREATMENT_INDICATOR_FEATURES
    if treatment_continuous_features is None:
        treatment_continuous_features = TREATMENT_CONTINUOUS_FEATURES
    if horizons is None:
        horizons = HORIZONS

    variable_reference = pl.read_csv(
        VARIABLE_REFERENCE_PATH, separator="\t", null_values=["None"]
    )

    features = []

    if variables is not None:
        variables = variables.copy()

    for row in variable_reference.rows(named=True):
        variable = row["VariableTag"]

        if variables is not None and variable not in variables:
            continue

        if variables is not None:
            variables.remove(row["VariableTag"])

        if row["LogTransform"] is True:
            variable = f"log_{variable}"

        if row["VariableType"] == "static":
            features += [variable]
        elif row["DataType"] == "continuous":
            features += [  # These variables are not based on any horizon.
                f"{variable}_ffilled",
                f"{variable}_missing",
                f"{variable}_sq_ffilled",
            ]
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in continuous_features
                for horizon in horizons
            ]
        elif row["DataType"] == "categorical":
            features += [variable]
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in categorical_features
                for horizon in horizons
            ]
        elif row["DataType"] == "treatment_ind":
            features += [variable]
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in treatment_indicator_features
                for horizon in horizons
            ]
        elif row["DataType"] == "treatment_cont":
            features += [
                f"{variable}_{feature}_h{horizon}"
                for feature in treatment_continuous_features
                for horizon in horizons
            ]
        else:
            raise ValueError(f"Unknown DataType: {row['DataType']}")

    if variables is None or "time_hours" in variables:
        features += ["time_hours"]
        if variables is not None:
            variables.remove("time_hours")

    if variables is not None and len(variables) > 0:
        raise ValueError(f"Unknown variables: {variables}")

    return features
