from pathlib import Path

import click
import polars as pl


@click.command()
@click.option("--data_dir", type=click.Path(exists=True))
def main(data_dir):
    """
    Split mimic, miiv, and aumc datasets into subsets.

    These are:
     - mimic -> mimic-carevue and mimic-metavision.
     - miiv -> miiv-late.
     - aumc -> aumc-early and aumc-late.

    The mimic hospital used the Philips Carevue EHR system until 2008 and the iMDsoft
    Metavision ICU system from 2008 onwards. The data format differs between these two
    EHR systems. We split the mimic dataset into mimic-carevue and mimic-metavision
    subsets to allow for separate analysis of these two EHR systems. The miiv (mimic-iv)
    dataset contains data from the mimic hospital after the switch to the Metavision EHR
    system. There is an overlap between miiv and mimic-metavision. This cannot be
    identified by stay_id due to anonymization. We thus filter out all "early" stays
    from the miiv dataset to create the miiv-late dataset with no overlap with
    mimic-metavision.

    The aumc dataset contains data from the Amsterdam UMC hospital from 2003 - 2016.
    The year has been anonymized and grouped into two periods: around 2006 and around
    2013. We split the aumc dataset into aumc-early and aumc-late subsets to allow for
    separate analysis of these two periods.

    Parameters
    ----------
    data_dir : str
        Path to the data directory. We expect input data at, e.g.,
        `data_dir/mimic/sta.parquet` and `data_dir/mimic/dyn.parquet`. We write the
        output data to `data_dir/mimic-carevue/sta.parquet`, etc.
    """
    for source, target, filter_ in [
        ("mimic", "mimic-carevue", pl.col("carevue") & pl.col("metavision").is_null()),
        (
            "mimic",
            "mimic-metavision",
            pl.col("metavision") & pl.col("carevue").is_null(),
        ),
        ("miiv", "miiv-late", pl.col("year") > 2012),
        ("aumc", "aumc-early", pl.col("year") == 2006),
        ("aumc", "aumc-late", pl.col("year") == 2013),
    ]:
        sta = pl.scan_parquet(Path(data_dir) / source / "sta.parquet").filter(filter_)
        sta = sta.collect()

        dyn = pl.scan_parquet(Path(data_dir) / source / "dyn.parquet")
        dyn = dyn.filter(pl.col("stay_id").is_in(sta["stay_id"])).collect()

        (Path(data_dir) / target).mkdir(parents=True, exist_ok=True)
        sta.write_parquet(Path(data_dir) / target / "sta.parquet")
        dyn.write_parquet(Path(data_dir) / target / "dyn.parquet")


if __name__ == "__main__":
    main()
