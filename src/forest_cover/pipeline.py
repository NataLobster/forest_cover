import configparser
from typing import Tuple, Dict, Any, List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(
        use_scaler: bool,
        classifier: str,
        random_state: int,
) -> tuple[Pipeline, dict[Any, list[Any]]]:

    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if classifier == "KNeighborsClassifier":
        pipeline_steps.append(
            (
                "classifier",
                KNeighborsClassifier(),
            )
        )
    elif classifier == "RandomForestClassifier":
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(),
            )
        )

    # большая печалька, но не хочет работать через ini из train
    config = configparser.ConfigParser()  # создаём объекта парсера
    config.read("train.ini")  # читаем конфиг

    hyper_param = dict(config[classifier])

    for key in hyper_param.keys():
        hyper_param[key] = list(hyper_param[key].split(','))
        for i in range(len(hyper_param[key])):
            if hyper_param[key][i]=='None':
                hyper_param[key][i] = None
            else:
                try:
                    hyper_param[key][i] = int(hyper_param[key][i])
                except Exception:
                    pass
    """if classifier == "KNeighborsClassifier":
        hyper_param = dict()
        hyper_param['classifier__n_neighbors'] = [3, 5, 8]
        hyper_param['classifier__weights'] = ['uniform', 'distance']
    if classifier == "RandomForestClassifier":
        hyper_param = dict()
        hyper_param['classifier__n_estimators'] = [100, 200, 500]
        hyper_param['classifier__max_depth'] = [10, 20, None]"""
    print(hyper_param)

    return Pipeline(steps=pipeline_steps), hyper_param

