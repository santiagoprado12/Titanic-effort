from enum import Enum
from typing_extensions import Annotated
from typing import List, Optional
import typer
import src.ml_core.train as train_model
from src.utils.keys_extraction import KeysExtraction
from src.utils.utils import run_makefile
import src.ml_core.validation as validation_model
import os
from src.configs import Configs

app = typer.Typer()  # Create a new typer.Typer() application.


def create_enum(enum_name, values):
    return Enum(enum_name, {value: value for value in values})

model_type_list = [model["name"] for model in Configs().models]
ModelType = create_enum("ModelType", model_type_list)


@app.command()
def train(model: Annotated[Optional[List[ModelType]], typer.Option(..., "-m", "--model", help="model to train")],
          register: bool = typer.Option(False, "--register-model", "-rm", help="register the trained model"),
          threshold: float = typer.Option(None, "--acc-threshold", "-th", help="accuracy threshold for the model to be registered (between 0 and 1))")):
    """Train the model"""

    if threshold is None:
        KeysExtraction().set_env_variables()
        run_makefile("dvc-pull-data")
        train_model.train([mod.value for mod in model])
    elif 0 <= threshold <= 1:
        KeysExtraction().set_env_variables()
        run_makefile("dvc-pull-data")
        train_model.train([mod.value for mod in model], threshold)
    else:
        typer.echo("Invalid input. Please enter a float number between 0 and 1.")
        raise typer.Abort()

    if register:
        run_makefile("register_model")


@app.command()
def validation(threshold: float = typer.Option(None, "--acc-threshold", "-th", help="accuracy threshold for retrain the model (between 0 and 1))")):
    """Validate the model"""

    KeysExtraction().set_env_variables()
    run_makefile("dvc-pull-data")
    run_makefile("dvc-pull-model")
    score = validation_model.validate()

    if threshold is not None:

        if 0 <= threshold <= 1:
            if score < threshold:
                typer.echo("The model is not good enough. training a new model.")
                run_makefile("train-model")
        else:
            typer.echo("Invalid input. Please enter a float number between 0 and 1.")
            raise typer.Abort()


@app.command()
def dataup(only_train: bool = typer.Option(False, "--only-train", "-t", help="only register the train data"),
           only_validation: bool = typer.Option(False, "--only-validation", "-v", help="only register the validation data")):
    """Update the data in the registry""" 

    if only_train:
        if os.path.exists("data/train.csv"):
            KeysExtraction().set_env_variables()
            run_makefile("dvc-push-train-data")
        else:
            typer.echo("The train data does not exist. Please save the train (train.csv) data in the data folder and try again.")
            raise typer.Abort()

    elif only_validation:
        if os.path.exists("data/validation.csv"):
            KeysExtraction().set_env_variables()
            run_makefile("dvc-push-validation-data")
        else:
            typer.echo("The validation data does not exist. Please save the validation (validation.csv) data in the data folder and try again.")
            raise typer.Abort()

    else:
        if os.path.exists("data/train.csv") and os.path.exists("data/validation.csv"):
            KeysExtraction().set_env_variables()
            run_makefile("dvc-push-data")
        else:
            typer.echo("The train or validation data does not exist. Please save the train (train.csv) and validation (validation.csv) data in the data folder and try again.")
            raise typer.Abort()


@app.command()
def test(coverage: bool = typer.Option(False, "--coverage", "-c", help="run the tests with coverage", show_default=False)):
    """Run the tests"""

    if coverage:
        run_makefile("test-coverage")
    else:
        run_makefile("test")


if __name__ == '__main__':
    app()
