# Titanic Training, and Validation Package

## Introduction

This Python package provides a comprehensive solution for performing training and validation on the Titanic dataset. It leverages DVC (Data Version Control) and follows MLOps best practices to ensure reproducibility and efficient management of data and model versions. The package includes a CLI (Command Line Interface) built using Typer, allowing users to easily interact with the various functionalities.

## Package Structure (reelevant files only)

<pre>
titanic_effort/
├── data/
│ ├── train.csv.dvc
│ └── validation.csv.dvc
├── models/
│ └── best_model.pkl.dvc
├── notebooks/
│ └── preprocessing.ipynb
├── src/
│ ├── ml_core/
│ │ ├── train.py
│ │ └── validation.py
│ ├── ml_pipelines/
│ │ ├── feature_selection.py
│ │ ├── model_training.py
│ │ └── pipeline_connection.py
│ ├── utils/
│ │ ├── data_functions.py
│ │ ├── keys_extraction.py
│ │ └── utils.py
│ ├── CLI.py
│ └── configs.py
├── tests/
│ ├── test_ml_core/
| | └── ...
│ ├── test_pipelines/
| | └── ...
│ └── test_utils/
|   └── ...
├── .gitignore
├── README.md
└── ...
</pre>
## Package Overview

### Data and Model Versioning

I utilize DVC (Data Version Control) for managing dataset and model versions. This approach ensures reproducibility, traceability, and easy collaboration. The data is stored in the `data/` directory, while model artifacts are saved in the `models/` directory. DVC is integrated with AWS S3 to store and manage these versions remotely.

### Training Process

I provide a Jupyter Notebook for the training process. You can access it [here](https://nbviewer.org/github/santiagoprado12/Titanic-effort/blob/dev/notebooks/preprocessing.ipynb).

Files and their use:

- `src/analysis.py`: Contains functions for exploratory data analysis and preprocessing.
- `src/training.py`: Implements the model training pipeline.
- `src/validation.py`: Handles model validation and evaluation.

## CLI Overview

### Train Command

Train the model.

```bash
python3 -m src.CLI train --model <model_type>... [--register-model] [--acc-threshold <threshold>]
```
- model <model_type>...: Specify one or more model types to train.
- register-model (-rm): Register the trained model.
- acc-threshold <threshold> (-th): Accuracy threshold for model registration (between 0 and 1).


### Validate Command

Validate the model.

```bash
python3 -m src.CLI validation [--acc-threshold <threshold>]
```
- acc-threshold <threshold> (-th): Accuracy threshold for the model to be retraining (between 0 and 1) if none only will show the validation score.

### Data Update Command

Update the data in the registry.

```bash
python3 -m src.CLI dataup [--only-train] [--only-validation]
```
- only-train (-ot): Update only the training data.
- only-validation (-ov): Update only the validation data.

### Test Command

Run the unit tests.

```bash
python3 -m src.CLI test [--coverage]
```

- coverage (-c): Run the tests with coverage.

### Help Command
Display help message.

```bash
python3 -m src.CLI --help
```



##  **Quick Start Guide:**

### Installation

1. clone the repository:

   ```bash
    git clone santiagoprado12/titanic_effort
    ```

2. go to the project directory:

    ```bash
    cd titanic_effort
    ```

3. create a virtual environment and install the package:

    ```bash
    make install && source .venv/bin/activate
    ```

4. set the environment variable for your secret key (optional):

    ```bash
    export SECRET_KEY=<your_secret_key>
    ```

### Train and Validate a Models


1. Training a Model:

    ```bash
    python3 -m src.CLI train -rm --model=knn --model=random_forest --model=gradient_boosting --model=svm -th=0.7
    ```
    Feel free of adding more models to the list, change the grid parameters and search tecnique in the file `src/configs.py`.

2. Model Validation:

    ```bash
    python3 -m src.CLI validation
    ```

### Update Data in the Registry

1. Save the new data in the `data/` directory, as `train.csv` and `validation.csv`.

    ```bash
    cp <path_to_new_data>/train.csv data/train.csv
    cp <path_to_new_data>/validation.csv data/validation.csv
    ```

    or you can simply copy and paste the data into the `data/` directory.

2. Update the data in the registry:

    ```bash
    python3 -m src.CLI dataup
    ```

Now you can train and validate the model with the new data using the commands described above.

### Run the Unit Tests

1. Run the tests with the coverage report:

    ```bash
    python3 -m src.CLI test -c
    ```

Made with ❤️ and **titanic-effort** by Santiago Prado



