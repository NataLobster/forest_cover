from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


# get clear data
def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.Series, str]:
    dataset = pd.read_csv(csv_path)
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    click.echo(f"Dataset shape: {features.shape}.")
    # features_train, features_val, target_train,
    # target_val = train_test_split(
    #    features, target, test_size=test_split_ratio,
    #    random_state=random_state
    # )

    # return features_train, features_val, target_train, target_val, 'None'
    return features, target, "None"


# get data with svd transform
def get_dataset_svd(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.Series, str]:
    dataset = pd.read_csv(csv_path)
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    svd = TruncatedSVD(n_components=30, random_state=42)
    features_new = svd.fit_transform(features)
    click.echo(f"Dataset shape: {features_new.shape}.")
    # features_train, features_val, target_train,
    # target_val = train_test_split(
    #    features_new, target, test_size=test_split_ratio,
    #    random_state=random_state
    # )

    # return features_train, features_val, target_train, target_val, 'SVD'
    return features_new, target, "SVD"


# get data with pca transform
def get_dataset_pca(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.Series, str]:
    dataset = pd.read_csv(csv_path)
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    pca = PCA(n_components=30, random_state=42)
    features_new = pca.fit_transform(features)
    click.echo(f"Dataset shape: {features_new.shape}.")
    # features_train, features_val, target_train,
    # target_val = train_test_split(
    #    features_new, target, test_size=test_split_ratio,
    #    random_state=random_state
    # )

    # return features_train, features_val, target_train, target_val, 'PCA'
    return features_new, target, "PCA"
