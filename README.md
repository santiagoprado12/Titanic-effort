# Titanic Training, and Validation Package

## Introduction

This comprehensive package is tailored for model training, validation and deployment on the iconic Titanic dataset. By harnessing the power of Data Version Control (DVC) and embracing MLOps best practices, ensure absolute reproducibility and streamlined management of data and model versions. 

## Key Features

- **MLOps Excellence**: Follow best practices in MLOps, ensuring reliable and reproducible model development.
- **CLI Convenience**: Interact effortlessly with the package using the intuitive Command Line Interface (CLI) built with Typer.
- **Data Validation with Great Expectations**: Verify and validate incoming data using the power of Great Expectations before integration.

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

DVC (Data Version Control) is used for managing dataset and model versions. This approach ensures reproducibility, traceability, and easy collaboration. The data is stored in the `data/` directory, while model is saved in the `models/` directory. DVC is integrated with AWS S3 to store and manage these versions remotely.

### Design Training and pipelines Process

I provide a Jupyter Notebook with the training process. You can access it [here](https://nbviewer.org/github/santiagoprado12/Titanic-effort/blob/dev/notebooks/preprocessing.ipynb).


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

### Requirements

1. python > 3.8

2. make installed 

### Installation

1. clone the repository:

   ```bash
    git clone https://github.com/santiagoprado12/titanic_effort.git
    ```

2. go to the project directory:

    ```bash
    cd titanic_effort
    ```

3. create a virtual environment and install the package:

    ```bash
    make install && source venv/bin/activate
    ```

4. set the environment variable for your secret key (optional):

    ```bash
    export SECRET_KEY=<your_secret_key>
    ```

### Train and Validate the Models


1. Training a Model:

    ```bash
    python3 -m src.CLI train -rm --model=knn --model=random_forest --model=gradient_boosting -th=0.7
    ```
    Feel free of adding more models to the list, change the grid parameters and search tecnique in the file [`src/configs.py`](src/configs.py).

2. Model Validation:

    ```bash
    python3 -m src.CLI validation
    ```

### Update Data in the Registry

1. Save the new data in the [data/](data/) directory, as [train.csv](train.csv)  and [validation.csv](validation.csv).

    ```bash
    cp <path_to_new_data>/train.csv data/train.csv
    cp <path_to_new_data>/validation.csv data/validation.csv
    ```

    or you can simply copy and paste the data into the [data/](data/) directory.

2. Update the data in the registry:

    ```bash
    python3 -m src.CLI dataup
    ```
Your data will be tested using **great expectations**, to guarantee the data quality, if it passes the test it will be updated in the registry.

**Now you can train and validate the model over the new data using the commands described above.**

### Run the Unit Tests

1. Run the tests with the coverage report:

    ```bash
    python3 -m src.CLI test -c
    ```

Then you can see the coverage report in the terminal, this is the current coverage report:

```    
Name                                               Stmts   Miss  Cover
----------------------------------------------------------------------
api/app/models.py                                     23      0   100%
api/app/utils.py                                      20      0   100%
api/app/views.py                                       9      0   100%
api/main.py                                            7      0   100%
src/__init__.py                                        0      0   100%
src/configs.py                                        10      0   100%
src/ml_pipelines/__init__.py                           3      0   100%
src/ml_pipelines/feature_selection.py                 65      0   100%
src/ml_pipelines/model_training.py                    29      7    76%
src/ml_pipelines/pipeline_connection.py               54     10    81%
src/utils/__init__.py                                  3      0   100%
src/utils/data_functions.py                           49      3    94%
src/utils/keys_extraction.py                          24      0   100%
src/utils/utils.py                                     8      0   100%
tests/__init__.py                                      0      0   100%
tests/test_api/__init__.py                             0      0   100%
tests/test_api/test_api.py                             7      0   100%
tests/test_pipelines/__init__.py                       0      0   100%
tests/test_pipelines/test_feature_selection.py        20      0   100%
tests/test_pipelines/test_model_training.py           56      2    96%
tests/test_pipelines/test_pipeline_connection.py      68      6    91%
tests/test_utils/__init__.py                           0      0   100%
tests/test_utils/test_data_functions.py               30      0   100%
tests/test_utils/test_keys_extraction.py              39      1    97%
tests/test_utils/test_utils.py                         8      0   100%
----------------------------------------------------------------------
TOTAL                                                532     29    95%
```
## Putting in Production the Model
### Model Inference Endpoint
see the code [here](api/)

The model inference endpoint is deployed on AWS App Runner. This means that the trained machine learning model is accessible via an API hosted on the provided URL: https://vp5ye3hgsu.us-east-2.awsapprunner.com/. Users can send requests to this endpoint to get predictions from the trained model.
**Try it out!**

But other tecnologies can be used to deploy the model, like AWS Lambda, AWS ECS, AWS EKS, AWS EC2, AWS SageMaker, etc. It depends on the use case.


### CI/CD with GitHub Actions
see the code [here](.github/workflows/CICD.yaml)

This project has a CI/CD setup using GitHub Actions. Here's a high-level overview of the process:

1. **Testing and Deployment in Docker:** When changes are pushed to the GitHub repository, GitHub Actions kicks in. It runs tests to ensure that the code is functioning correctly. If the tests pass, it then builds a Docker container containing the application and dependencies. This ensures consistency across environments.

2. **ECR (Elastic Container Registry):** The Docker image is pushed to ECR, which is a managed Docker container registry provided by AWS. This serves as a centralized location to store and manage the Docker images.

3. **App Runner Deployment:** The Docker image is pulled from ECR and deployed on AWS App Runner. This sets up the API endpoint, making the machine learning model accessible to users.

### Data Validation and Model Retraining
see the code [here](.github/workflows/monitoring.yaml)

This package has implemented in github actions a data validation and model retraining pipeline, in order to maintain the model accuracy:

1. **Data quality check:** Before being uploaded to trhe registry, the data is validated using Great Expectations. This ensures that the incoming data adheres to the expected format and quality.

2. **Model Accuracy Check:**  When a new data version is uploaded to the registry, a GitHub Action is triggered. After data validation, the model's accuracy is evaluated using the new data. If the accuracy falls below a defined threshold, it triggers a retraining process.

3. **Model Retraining and Versioning:** In case of lower accuracy, a new model training process is initiated. The updated model is then saved as a new version using Data Version Control (DVC). This ensures to have a traceable history of model versions and their performance.

### ETL Pipeline

This package hasn´t implemented an ETL pipeline, but it can be done using Airflow or AWS Step Functions. In order to extract the data from the source, transform it and load it into the registry. The importance of this is to have a reliable data source, and maintain updated the training and validation data.



Made with ❤️ and a **titanic-effort** by Santiago Prado



