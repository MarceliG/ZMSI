# https://docs.python.org/3/library/argparse.html

import argparse


def parser() -> argparse.Namespace:
    """
    Create and configures an argument parser for the application.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - processing (bool): If provided, triggers data preprocessing.
            - fine_tune (str): Path to the dataset used for fine-tuning the model.
            - predict (str): Path to the input data for making predictions.
            - help (str): Displays information on how to use the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Application for data preprocessing, model fine-tuning, and prediction."
    )

    parser.add_argument(
        "--download",
        "-d",
        action="store_true",
        help="Download dataset.",
    )

    parser.add_argument(
        "--preprocessing",
        "-p",
        action="store_true",
        help="Perform data preprocessing.",
    )

    # TODO: prepare your code for such situations or similar
    # parser.add_argument(
    #     "--fine-tune",
    #     "-ft",
    #     type=str,
    #     help="Path to the data for model fine-tuning.",
    # )

    # parser.add_argument(
    #     "--predict",
    #     "-pr",
    #     type=str,
    #     help="Path to the input data for prediction.",
    # )

    return parser.parse_args()
