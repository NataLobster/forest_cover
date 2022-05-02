from pathlib import Path
from joblib import dump

import click


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .data import get_dataset


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
    default="data/model_lr.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--random-state", default=42, type=int)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
def train(
    dataset_path: Path,random_state: int, save_model_path: Path,
    test_split_ratio: float
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    classifier = LogisticRegression(random_state=42,  max_iter=5000).fit(features_train, target_train)
    accuracy = accuracy_score(target_val, classifier.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
    dump(classifier, save_model_path)

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
    default="data/model_rf.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--random-state", default=42, type=int)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)

def train_rf(
    dataset_path: Path, random_state: int, save_model_path: Path, 
    test_split_ratio: float,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    classifier = RandomForestClassifier(random_state=42).fit(features_train, target_train)
    accuracy = accuracy_score(target_val, classifier.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
    dump(classifier, save_model_path)
