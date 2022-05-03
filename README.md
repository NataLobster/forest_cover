Homework for RS School Machine Learning course.

## Usage
This package allows you to predict the forest cover type.
1. Clone this repository to your machine.
2. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. You can generate a EDA report with the following code
```sh
poetry run prof_data -d <path to csv with data> -s <path to save report>
```
6. Run train with the following command (logistic regression):
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
Run train with the following command (random forest):
```sh
poetry run train_rf -d <path to csv with data> -s <path to save trained model>
```
You can use two models (KNeighborsClassifier and RandomForestClassifier) with corresponding hyperparameters
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
![изображение](https://user-images.githubusercontent.com/70448060/166432179-ba8ffdbe-7f69-47ec-990a-0f19bc45b728.png)

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
