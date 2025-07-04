import polars as pl
from icdmappings import Mapper

datasets = [
    "miiv",
    "aumc",
    "mimic",
    "eicu",
    "zigong",
    "picdb",
    "hirid",
    "sic",
    "nwicu",
    "mimic-carevue",
]

mapper = Mapper()


def icd10_blocks(x):
    """Map from icd 10 strings to ~130 Blocks of ICD-10 CM codes."""
    return mapper.map(x, source="icd10", target="block") or ""


def icd9_blocks(x):
    """Map from icd 9 strings to ~130 Blocks of ICD-10 CM codes."""
    icd10code = mapper.map(x, source="icd9", target="icd10")
    if x is not None and icd10code is None:
        icd10code = mapper.map(x.replace(".", ""), source="icd9", target="icd10")
    if icd10code is None:
        return ""
    return mapper.map(icd10code, source="icd10", target="block") or ""


for dataset in datasets:
    print(f"Processing {dataset}")
    sta = pl.scan_parquet(f"/cluster/work/math/lmalte/data/{dataset}/sta.parquet")
    sta = sta.collect()

    if dataset == "hirid":
        sta = sta.with_columns(
            pl.when(pl.col("apache_group").is_null())
            .then(pl.lit([]))
            .otherwise(
                pl.col("apache_group")
                .cast(pl.String)
                .map_elements(lambda x: [x], return_dtype=pl.List(pl.String))
                .alias("icd_blocks")
            )
        )

    if dataset == "eicu":
        # For eicu, we don't extract as list, but as a string of comma-separated values.
        sta = sta.with_columns(
            pl.col("icd9_diagnosis").map_elements(
                lambda s: s.map_elements(
                    lambda ls: ls.split(", ") if ls is not None else [],
                    skip_nulls=False,
                    return_dtype=pl.List(pl.String),
                ),
            ),
            return_dtype=pl.List(pl.List(pl.String)),
        ).map_elements(lambda s: s.explode(), return_dtype=pl.List(pl.String))

    sta = sta.with_columns(
        pl.col("icd10_diagnosis").cast(pl.List(pl.String)).fill_null([]),
        pl.col("icd9_diagnosis").cast(pl.List(pl.String)).fill_null([]),
    )
    sta = sta.with_columns(
        pl.col("icd10_diagnosis")
        .map_elements(
            lambda s: sorted(z for z in {icd10_blocks(x) for x in s} if z != ""),
            return_dtype=pl.List(pl.String),
        )
        .alias("icd10_blocks"),
        pl.col("icd9_diagnosis")
        .map_elements(
            lambda s: sorted(z for z in {icd9_blocks(x) for x in s} if z != ""),
            return_dtype=pl.List(pl.String),
        )
        .alias("icd9_blocks"),
    )

    sta = sta.with_columns(
        pl.concat_list("icd10_blocks", "icd9_blocks").alias("icd10_blocks"),
    )
    sta.write_parquet(f"/cluster/work/math/lmalte/data/{dataset}/sta.parquet")
