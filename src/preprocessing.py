import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset

from src.logs import logger
from src.resources import FilePath


def download_dataset(name: str) -> Dataset:
    """
    Download a dataset by name.

    Args:
        name (str): The name of the dataset to download.

    Returns:
        Dataset: Object containing the downloaded data.
    """
    return load_dataset(name)


def split_datasets(dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into training, test, and validation sets.

    Args:
        dataset (Dataset): A HuggingFace Dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - The first DataFrame contains the 'train' split.
            - The second DataFrame contains the 'test' split.
            - The third DataFrame contains the 'validation' split.
    """
    return pd.DataFrame(dataset["train"]), pd.DataFrame(dataset["test"]), pd.DataFrame(dataset["validation"])


def analyses():
    dataset = download_dataset(name="legacy-datasets/allegro_reviews")
    train, test, validation = split_datasets(dataset)

    train["text_length"] = train["text"].apply(len)
    test["text_length"] = test["text"].apply(len)
    validation["text_length"] = validation["text"].apply(len)

    avg_length_train = train["text_length"].mean()
    avg_length_test = test["text_length"].mean()
    avg_length_validation = validation["text_length"].mean()

    logger.info(f"Średnia długość recenzji w zbiorze treningowym: {avg_length_train:.2f}")
    logger.info(f"Średnia długość recenzji w zbiorze testowym: {avg_length_test:.2f}")
    logger.info(f"Średnia długość recenzji w zbiorze walidacyjnym: {avg_length_validation:.2f}")

    # Tworzenie wykresu z trzema histogramami
    plt.figure(figsize=(10, 6))

    # Wykres dla danych treningowych
    plt.subplot(3, 1, 1)  # 3 wiersze, 1 kolumna, pierwszy wykres
    plt.hist(train["text_length"], bins=50, alpha=0.7, label="Train")
    plt.title("Histogram długości recenzji - Train")
    plt.xlabel("Długość recenzji")
    plt.ylabel("Liczba recenzji")

    # Wykres dla danych testowych
    plt.subplot(3, 1, 2)  # Drugi wykres
    plt.hist(test["text_length"], bins=50, alpha=0.7, label="Test")
    plt.title("Histogram długości recenzji - Test")
    plt.xlabel("Długość recenzji")
    plt.ylabel("Liczba recenzji")

    # Wykres dla danych walidacyjnych
    plt.subplot(3, 1, 3)  # Trzeci wykres
    plt.hist(validation["text_length"], bins=50, alpha=0.7, label="Validation")
    plt.title("Histogram długości recenzji - Validation")
    plt.xlabel("Długość recenzji")
    plt.ylabel("Liczba recenzji")

    plt.tight_layout()

    plt.savefig(FilePath.hitogram_path)
