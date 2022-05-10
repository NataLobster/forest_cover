import configparser
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


def create_pipeline(
    use_scaler: bool,
    classifier: str,
    random_state: int,
) -> tuple[Any, Dict[str, str]]:
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
                RandomForestClassifier(random_state=random_state),
            )
        )
    elif classifier == "ExtraTreesClassifier":
        pipeline_steps.append(
            (
                "classifier",
                ExtraTreesClassifier(random_state=random_state),
            )
        )

    # получаем список гиперпараметров из ini
    config = configparser.ConfigParser()  # создаём объекта парсера
    config.read("train.ini")  # читаем конфиг
    hyper_param: Dict[Any, Any]
    hyper_param = dict(config[classifier])

    for key in hyper_param.keys():
        hyper_param[key] = list(hyper_param[key].split(","))
        for i in range(len(hyper_param[key])):
            if hyper_param[key][i] == "None":
                hyper_param[key][i] = None
            else:
                try:
                    hyper_param[key][i] = int(hyper_param[key][i])
                except Exception:
                    pass

    return Pipeline(steps=pipeline_steps), hyper_param
