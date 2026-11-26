import argparse
from pathlib import Path

from simple_model import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate diabetes classifiers.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "pima-indians-diabetes-labeled.csv",
        help="Path to the CSV with named columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "simple_model",
        help="Directory for the model and cross-validation metrics.",
    )
    arguments = parser.parse_args()
    train(arguments.data, arguments.output)
