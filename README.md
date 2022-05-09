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
6. Run train with the following command :
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can use two models (KNeighborsClassifier, RandomForestClassifier, ExtraTreesClassifier) with corresponding hyperparameters
for example, with
```sh
poetry run train --classifier RandomForestClassifier --n-estimators 200 --max-depth 10
```
(this was relevant before the implementation of task 9)  
Now hyperparameters are set in the train.ini file. You should specify just classifier, path or type of feature engineering (None, PCA,SVD)
for example
```sh
poetry run train --classifier RandomForestClassifier --feature-eng PCA
```

output for task7:
![изображение](https://user-images.githubusercontent.com/70448060/167253224-f9578bd3-c263-4e12-b4a5-43f1c4f1173c.png)

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
task8
![изображение](https://user-images.githubusercontent.com/70448060/167253279-360f1b76-73cb-4188-ac8f-6b2864df89f4.png)
![изображение](https://user-images.githubusercontent.com/70448060/167253343-95d29090-355c-4d12-bd60-ddbc31600940.png)

or after implementing task9
![изображение](https://user-images.githubusercontent.com/70448060/167372667-f245c406-6434-46d0-b710-658e30878132.png)

### Pay attention that nested cv takes a lot of time for some models

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Then you can run 
```
poetry run black src
```
```
poetry run flake8
```
task12
![изображение](https://user-images.githubusercontent.com/70448060/166528660-0e44c37c-9390-444b-969e-7ba206e21cd3.png)
![изображение](https://user-images.githubusercontent.com/70448060/167245738-b24ceb83-da62-4faa-92cb-530bba85b608.png)

 ```
poetry run mypy src
```
task13
![изображение](https://user-images.githubusercontent.com/70448060/167246245-b7989243-c51c-41bf-8548-ccac3f8bb1a7.png)

Or combine it into a single command
```
poetry run nox
```
task14
![изображение](https://user-images.githubusercontent.com/70448060/167295998-61b844ea-744e-432a-bbde-37496fb44726.png)


