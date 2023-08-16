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

from ruamel.yaml import YAML
import great_expectations as gx
from pprint import pprint

yaml = YAML()
context = gx.get_context()

app = typer.Typer()  # Create a new typer.Typer() application.


def create_enum(enum_name, values):
    return Enum(enum_name, {value: value for value in values})

model_type_list = [model["name"] for model in Configs().models]
ModelType = create_enum("ModelType", model_type_list)


@app.command()
def train(model: Annotated[Optional[List[ModelType]], typer.Option(..., "-m", "--model", help="model to train")],
          register: bool = typer.Option(False, "--register-model", "-rm", help="register the trained model"),
          threshold: float = typer.Option(None, "--acc-threshold", "-th", help="accuracy threshold for the model to be registered (between 0 and 1))"),
          git_actions: bool = typer.Option(False, "--git-actions", "-ga", help="is running from git actions?")):
    """Train the model"""

    if threshold is None:
        KeysExtraction().set_env_variables()
        if not git_actions: run_makefile("dvc-pull-data")
        train_model.train([mod.value for mod in model])
    elif 0 <= threshold <= 1:
        KeysExtraction().set_env_variables()
        if not git_actions: run_makefile("dvc-pull-data")
        train_model.train([mod.value for mod in model], threshold)
    else:
        typer.echo("Invalid input. Please enter a float number between 0 and 1.")
        raise typer.Abort()

    if register:
        if not git_actions: run_makefile("register_model")


@app.command()
def validation(threshold: float = typer.Option(None, "--acc-threshold", "-th", help="accuracy threshold for retrain the model (between 0 and 1))"),
               git_actions: bool = typer.Option(False, "--git-actions", "-ga", help="is running from git actions?")):
    """Validate the model"""

    KeysExtraction().set_env_variables()
    if not git_actions: run_makefile("dvc-pull-data")
    if not git_actions: run_makefile("dvc-pull-model")
    score = validation_model.validate()

    if threshold is not None:

        if 0 <= threshold <= 1:
            if score < threshold:
                typer.echo("The model is not good enough. training a new model.")
                if not git_actions: run_makefile("train-git-actions") 
                if git_actions: run_makefile("train") 
        else:
            typer.echo("Invalid input. Please enter a float number between 0 and 1.")
            raise typer.Abort()


@app.command()
def dataup(only_train: bool = typer.Option(False, "--only-train", "-t", help="only register the train data"),
           only_validation: bool = typer.Option(False, "--only-validation", "-v", help="only register the validation data")):
    """Update the data in the registry""" 
    context = gx.get_context()
    
    data = ["data/train.csv", "data/validation.csv"]
    data = ["data/train.csv"] if only_train else data
    data = ["data/validation.csv"] if only_validation else data

    gx_checkouts = {
        "data/train.csv": "myckp_train",
        "data/validation.csv": "myckp_validation"
    }

    commands = {
        "data/train.csv": "dvc-push-train-data",
        "data/validation.csv": "dvc-push-validation-data"
    }

    for data_path in data:
        if os.path.exists(data_path):
            res = context.run_checkpoint(checkpoint_name=gx_checkouts[data_path])

            if res.success:
                KeysExtraction().set_env_variables()
                run_makefile(commands[data_path])
            else:
                val_id = list(res["run_results"].keys())[0]
                details_url = res["run_results"][val_id]["actions_results"]["update_data_docs"]["local_site"]

                typer.echo("The {} data does not pass the validation. Please check the data docs for more details: {}".format(data_path, details_url))
                raise typer.Abort()
        else:
            typer.echo("The {} data does not exist. Please save the {} data in the data folder and try again.".format(data_path, data_path.split("/")[1]))
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
