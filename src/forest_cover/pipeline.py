from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(
    use_scaler: bool, classfier: str,  random_state: int, n_neighbors:int,
    weights:str, n_estimators:int, max_depth:int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if classfier == 'KNeighborsClassifier':
        pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
        )
        )
    elif classfier == 'RandomForestClassifier':
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state=random_state),
            )
        )

    return Pipeline(steps=pipeline_steps)

