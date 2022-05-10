from pathlib import Path
from typing import Any, Dict

from joblib import dump

import pandas as pd
import click
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from .data import get_dataset, get_dataset_svd, get_dataset_pca
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--random-state", default=42, type=int)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
@click.option(
    "--classifier",
    default="KNeighborsClassifier",
    type=str,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--feature-eng",
    default=None,
    type=str,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    feature_eng: str,
    classifier: str,
    use_scaler: bool,
) -> None:
    hyper_param: Dict[Any, Any]
    hyper_param = dict()  # "hyper_param: Dict[<type>, <type>]
    # get data
    if feature_eng is None:
        features, target, name_fe = get_dataset(
            dataset_path, random_state, test_split_ratio
        )
    elif feature_eng == "SVD":
        features, target, name_fe = get_dataset_svd(
            dataset_path, random_state, test_split_ratio
        )
    elif feature_eng == "PCA":
        features, target, name_fe = get_dataset_pca(
            dataset_path, random_state, test_split_ratio
        )
    pipeline, hyper_param = create_pipeline(
        use_scaler,
        classifier,
        random_state,
    )
    # tune hyperparameters
    cv_inner = KFold(n_splits=5, shuffle=True)
    search = GridSearchCV(
        pipeline,
        hyper_param,
        scoring="accuracy",
        n_jobs=1,
        cv=cv_inner,
        refit=True,
    )
    search.fit(features, target)
    dump(classifier, save_model_path)
    click.echo(f"best_params {search.best_params_}.")
    click.echo(f"best_score {search.best_score_}.")
    # get prediction for test.csv
    dataset = pd.read_csv("data/test.csv")
    predictions = search.predict(dataset)

    submission = pd.DataFrame({"Id": dataset["Id"], "Cover_Type": predictions})

    submission.to_csv("data/forest-cover-submission.csv", index=False)
