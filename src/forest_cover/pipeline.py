from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(
    use_scaler: bool, classfier: str, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if classfier == 'KNeighborsClassifier':
        pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(),
        )
        )
    elif classfier == 'RandomForestClassifier':
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(random_state=random_state),
            )
        )

    return Pipeline(steps=pipeline_steps)

