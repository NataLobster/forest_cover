from pathlib import Path
from typing import Any, Dict

from joblib import dump

import numpy as np
import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate
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
    "--n-neighbors",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--weights",
    default="uniform",
    type=str,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
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
    random_state: int,
    save_model_path: Path,
    classifier: str,
    test_split_ratio: float,
    use_scaler: bool,
    n_neighbors: int,
    weights: str,
    n_estimators: int,
    max_depth: int,
    feature_eng: str,
) -> None:
    scoring = ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
    hyper_param: Dict[Any, Any]
    hyper_param = dict()  # "hyper_param: Dict[<type>, <type>]

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
    with mlflow.start_run():
        pipeline, hyper_param = create_pipeline(
            use_scaler,
            classifier,
            random_state,
        )

        cv_inner = KFold(n_splits=3, shuffle=True)
        search = GridSearchCV(
            pipeline,
            hyper_param,
            scoring="accuracy",
            n_jobs=1,
            cv=cv_inner,
            refit=True,
        )
        cv_outer = KFold(n_splits=10, shuffle=True)
        score = cross_validate(
            search,
            features,
            target,
            cv=cv_outer,
            scoring=scoring,
            return_train_score=True,
            return_estimator=False,
        )
        mlflow.log_param(" classifier", classifier)
        mlflow.log_params(hyper_param)

        mlflow.log_param(" feature_engineering", name_fe)
        mlflow.log_metric("train_accuracy", np.mean(score["train_accuracy"]))
        mlflow.log_metric("test_accuracy", np.mean(score["test_accuracy"]))
        mlflow.log_metric("train_f1", np.mean(score["train_f1_weighted"]))
        mlflow.log_metric("test_f1", np.mean(score["test_f1_weighted"]))
        mlflow.log_metric(
            "train_roc_auc", np.mean(score["train_roc_auc_ovr_weighted"])
        )
        mlflow.log_metric(
            "test_roc_auc", np.mean(score["test_roc_auc_ovr_weighted"])
        )
        mlflow.sklearn.log_model(pipeline, "model")
        click.echo(
            f"Type of feature selecting: {name_fe} \n"
            f"Train accuracy, F1, ROC_AUC:"
            f" {np.mean(score['train_accuracy'])}"
            f",{np.mean(score['train_f1_weighted'])},"
            f"{np.mean(score['train_roc_auc_ovr_weighted'])}."
        )
        click.echo(
            f"Test accuracy, F1, ROC_AUC: "
            f"{np.mean(score['test_accuracy'])}"
            f",{np.mean(score['test_f1_weighted'])},"
            f"{np.mean(score['test_roc_auc_ovr_weighted'])}."
        )
        dump(classifier, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
