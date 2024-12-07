import os

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification

from src.config import FilePath
from src.logs import logger


def download_dataset(name: str) -> Dataset:
    """
    Download a dataset by name.

    Args:
        name (str): The name of the dataset to download.

    Returns:
        Dataset: Object containing the downloaded data.
    """
    logger.info(f"Start downloading the dataset: {name}...")
    dataset = load_dataset(name)
    logger.info("Dataset downloaded")
    return dataset


def save_dataset(dataset: Dataset, path: str) -> None:
    """
    Save a dataset to a specified path.

    Args:
        dataset (Dataset): The dataset object to save.
        path (str): The directory path where the dataset will be saved.
    """
    logger.info(f"Start saving the dataset on the path: {path}...")
    dataset.save_to_disk(path)
    logger.info("Dataset saved")


def load_dataset_from_path(path: str) -> Dataset:
    """
    Load a dataset from a path.

    Args:
        path (str): The path to the directory where the dataset is stored.

    Returns:
        Dataset: A dataset object loaded from the specified path.
    """
    logger.info(f"Start loading the dataset from the path: {path}...")
    dataset = load_from_disk(path)
    logger.info("Dataset successfully loaded")
    return dataset


def combine_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, validation_df: pd.DataFrame) -> DatasetDict:
    """
    Concatenates training, testing, and validation DataFrames into a single DatasetDict.

    Args:
        train_df (pd.DataFrame): The training DataFrame.
        test_df (pd.DataFrame): The testing DataFrame.
        validation_df (pd.DataFrame): The validation DataFrame.

    Returns:
        DatasetDict: A DatasetDict containing all the data from the three DataFrames.
    """
    logger.info("Start combination dataset (train, test, validation) into DatasetDict...")
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    datasetdict = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})
    logger.info("combined DatasetDict")
    return datasetdict


def save_dataframe_as_markdown(statistics_df: pd.DataFrame, filename: str) -> None:
    """
    Save a DataFrame as a Markdown table in a specified file.

    Args:
        statistics_df (pd.DataFrame): The DataFrame containing the statistics to be saved.
        filename (str): The name of the file (including extension).
    """
    statistics_df.to_markdown(os.path.join(FilePath.statistics, filename))


def load_model_from_disc(path: str) -> AutoModelForSequenceClassification:
    """
    Load a pre-trained BERT model for sequence classification from the specified directory.

    Args:
        path (str): The path to the directory where the model is saved.

    Returns:
        BertForSequenceClassification: The loaded BERT model ready for sequence classification.
    """
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(path)
    logger.info("Model loaded")
    return model
