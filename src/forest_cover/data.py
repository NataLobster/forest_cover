from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )

    return features_train, features_val, target_train, target_val


def get_dataset_svd(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    svd = TruncatedSVD(n_components=20, random_state=42)
    features_new = svd.fit_transform(features)
    features_train, features_val, target_train, target_val = train_test_split(
        features_new, target, test_size=test_split_ratio, random_state=random_state
    )

    return features_train, features_val, target_train, target_val


def get_dataset_pca(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    pca = PCA(n_components=20, random_state=42)
    features_new = pca.fit_transform(features)
    features_train, features_val, target_train, target_val = train_test_split(
        features_new, target, test_size=test_split_ratio, random_state=random_state
    )

    return features_train, features_val, target_train, target_val

