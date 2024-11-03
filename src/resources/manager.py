import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk


def download_dataset(name: str) -> Dataset:
    """
    Download a dataset by name.

    Args:
        name (str): The name of the dataset to download.

    Returns:
        Dataset: Object containing the downloaded data.
    """
    return load_dataset(name)


def save_dataset(dataset: Dataset, path: str) -> None:
    """
    Save a dataset to a specified path.

    Args:
        dataset (Dataset): The dataset object to save.
        path (str): The directory path where the dataset will be saved.
    """
    dataset.save_to_disk(path)


def load_dataset_from_path(path: str) -> Dataset:
    """
    Load a dataset from a path.

    Args:
        path (str): The path to the directory where the dataset is stored.

    Returns:
        Dataset: A dataset object loaded from the specified path.
    """
    return load_from_disk(path)


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
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    return DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})
