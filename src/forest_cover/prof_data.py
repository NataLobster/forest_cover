import pandas as pd
from pandas_profiling import ProfileReport
import click
from pathlib import Path


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-report-path",
    default="data/prof.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def prof(
    dataset_path: Path,
    save_report_path: Path,
) -> None:
    df = pd.read_csv(dataset_path)
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file(save_report_path)
