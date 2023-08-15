from enum import Enum
from typing_extensions import Annotated
from typing import List, Optional
import typer
import src.ml_core.train as train_model
from src.utils.keys_extraction import KeysExtraction
from src.utils.utils import run_makefile
import src.ml_core.validation as validation_model

app = typer.Typer() # Create a new typer.Typer() application.


class ModelType(str, Enum):
    random_forest = "random_forest"
    gradient_boosting = "gradient_boosting"
    knn = "knn"

@app.command()
def train(model: Annotated[Optional[List[ModelType]], typer.Option(..., "-m", "--model", help="model to train")],
          register: bool = typer.Option(False, "--register-model", "-rm", help="register the trained model"),
          threshold: float = typer.Option(None, "--acc-threshold", "-th", help="accuracy threshold for the model to be registered (between 0 and 1))")):
    """Train the model"""

    if threshold is None:
        KeysExtraction().set_env_variables()
        run_makefile("dvc-pull-data")
        train_model.main([mod.value for mod in model])
    elif 0 <= threshold <= 1:
        KeysExtraction().set_env_variables()
        run_makefile("dvc-pull-data")
        train_model.main([mod.value for mod in model], threshold)
    else:
        typer.echo("Invalid input. Please enter a float number between 0 and 1.")
        raise typer.Abort()

    if register:
        run_makefile("register_model")


@app.command()
def validation():
    """Validate the model"""

    KeysExtraction().set_env_variables()
    run_makefile("dvc-pull-data")
    run_makefile("dvc-pull-model")
    validation_model.main()


@app.command()
def dataup(only_train: bool = typer.Option(False, "--only-train", "-t", help="only register the train data"),
            only_validation: bool = typer.Option(False, "--only-validation", "-v", help="only register the validation data")):
    """Update the data in the registry"""

    if only_train:
        KeysExtraction().set_env_variables()
        run_makefile("dvc-push-train-data")
    
    elif only_validation:
        KeysExtraction().set_env_variables()
        run_makefile("dvc-push-validation-data")
    
    else:
        KeysExtraction().set_env_variables()
        run_makefile("dvc-push-data")


@app.command()
def test(coverage: bool = typer.Option(False, "--coverage", "-c", help="run the tests with coverage", show_default=False)):
    """Run the tests"""

    if coverage:
        run_makefile("test-coverage")
    else:
        run_makefile("test")


if __name__ == '__main__':
    app()