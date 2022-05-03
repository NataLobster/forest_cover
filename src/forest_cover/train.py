from pathlib import Path
from joblib import dump

import numpy as np
import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate

from forest_cover.data import get_dataset
from forest_cover.pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
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
    default='KNeighborsClassifier',
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
def train(
    dataset_path: Path, random_state: int, save_model_path: Path,
    classifier: str, test_split_ratio: float, use_scaler:bool, n_neighbors:int,
    weights:str, n_estimators:int, max_depth:int
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run() as run:
        pipeline = create_pipeline(use_scaler, classifier, n_neighbors,
                                   random_state, weights, n_estimators,
                                   max_depth)
        pipeline.fit(features_train, target_train)
        scoring = ['accuracy','f1_weighted','roc_auc_ovr_weighted']
        score = cross_validate(pipeline,features_train,target_train, cv=5,
                           scoring=scoring,return_train_score=True)
        mlflow.log_param("classifier", classifier)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("weights", weights)
        mlflow.log_param(" n_estimators",  n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("train_accuracy", np.mean(score['train_accuracy']))
        mlflow.log_metric("test_accuracy", np.mean(score['test_accuracy']))
        mlflow.sklearn.log_model(pipeline, "model")
        click.echo(
            f"Train accuracy, F1, ROC_AUC: {np.mean(score['train_accuracy'])}"
            f",{np.mean(score['train_f1_weighted'])},"
            f"{np.mean(score['train_roc_auc_ovr_weighted'])}.")
        click.echo(
            f"Test accuracy, F1, ROC_AUC: {np.mean(score['test_accuracy'])}"
            f",{np.mean(score['test_f1_weighted'])},"
            f"{np.mean(score['test_roc_auc_ovr_weighted'])}.")
        dump(classifier, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

