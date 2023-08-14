from enum import Enum
from typing_extensions import Annotated
from typing import List, Optional
import typer
import src.ml_core.train as train_model
from src.utils.keys_extraction import KeysExtraction
from src.utils.utils import run_makefile

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

    if register:
        run_makefile("register_model")

@app.command()
def main(user: Annotated[Optional[List[str]], typer.Option()] = None):
    if not user:
        print("No provided users")
        raise typer.Abort()
    for u in user:
        print(f"Processing user: {u}")

@app.command()
def say_bye(name: str):
    """Say bye to users"""

    print(f'Good bye {name}')


if __name__ == '__main__':
    app()