import re

import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.logs import logger
from src.resources import FilePath, Plotter, combine_datasets, download_dataset


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


def calculate_mean(df: pd.DataFrame) -> float:
    """
    Calculate the average length of text in a given dataset.

    Args:
        df (pd.DataFrame): A DataFrame containing column named 'text'.

    Returns:
        float: The average length of the text entries in the dataset.
    """
    return df["text_length"].mean()


def calculate_std(df: pd.DataFrame) -> float:
    """
    Calculate the average length of text in a given dataset.

    Args:
        df (pd.DataFrame): A DataFrame containing column named 'text'.

    Returns:
        float: The average length of the text entries in the dataset.
    """
    return df["text_length"].std()


def remove_similar_rows(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Find and remove rows that are duplicates or very similar based on text similarity.

    Args:
        df (pd.DataFrame): A DataFrame containing a column named 'text'.
        threshold (float): Similarity threshold for considering two texts as similar.

    Returns:
        pd.DataFrame: A DataFrame containing rows that are duplicates or similar.
    """
    vectorizer = TfidfVectorizer().fit_transform(df["text"])
    cosine_sim = cosine_similarity(vectorizer)

    to_drop = set()
    for row_idx in range(len(cosine_sim)):
        if row_idx not in to_drop:
            for compare_idx in range(row_idx + 1, len(cosine_sim)):
                if cosine_sim[row_idx][compare_idx] >= threshold:
                    to_drop.add(compare_idx)
                    break
    valid_to_drop = [index for index in to_drop if index in df.index]

    return df.drop(index=valid_to_drop).reset_index(drop=True)


def count_words(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column `text_length` to the DataFrame representing the length of each text.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'text' column with text data to be analyzed.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'text_length'.
    """
    df["text_length"] = df["text"].apply(len)
    return df


def remove_top_n_percent(df: pd.DataFrame, percent: int) -> pd.DataFrame:
    """
    Remove the top n percent of entries with the longest text lengths from the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'text_length' column.
        percent (int): The percentage of entries to remove (0-100)%.

    Returns:
        pd.DataFrame: The input DataFrame with the top n percent longest texts removed.
    """
    if not (0 <= percent <= 100):
        msg = "Percentage n must be between 0 and 100."
        raise ValueError(msg)

    n_to_drop = int(len(df) * (percent / 100))
    cutoff_length = df.nlargest(n_to_drop, "text_length")["text_length"].min()

    logger.info(f"Removing {n_to_drop} rows, which represents the top {percent}% of the longest entries.")
    logger.info(f"Cutoff length value: {cutoff_length}")
    logger.info(f"Size before removal: {len(df)}, size after removal: {len(df) - n_to_drop}")

    return df[df["text_length"] < cutoff_length].reset_index(drop=True)


def filter_repeated_rows(text: str, threshold: int) -> str | None:
    """
    Filter texts containing consecutive repeated words.

    Args:
        text (str): The input text.
        threshold (int): The minimum number of consecutive repetitions of word.

    Returns:
        str or None: Returns the original text if it doesn't contain repeated words or returns None to indicate spam.
    """
    pattern = re.compile(rf"\b(\w+)\b(?:\s+\1){{{threshold - 1},}}")
    return None if pattern.search(text) else text


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by removing whitespace, dropping duplicate texts, and eliminating rows with missing values.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'text' column.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df["text"] = df["text"].str.strip()
    df = df.drop_duplicates(subset="text")
    return df.dropna().reset_index(drop=True)


def preprocessing(dataset: Dataset) -> Dataset:
    """
    Preprocessing the input dataset.

    Function splits the dataset into training, testing, and validation sets,
    counts the number of words in each review, and then performs various processing
    operations such as removing duplicates and removing the longest entries. It also
    plots histograms for both raw and processed data.

    Args:
        dataset (Dataset): The input dataset to be processed.

    Returns:
        Dataset: A combined dataset after processing, containing training, testing, and validation data.
    """
    train_raw, test_raw, validation_raw = split_datasets(dataset)

    train_raw = count_words(train_raw)
    test_raw = count_words(test_raw)
    validation_raw = count_words(validation_raw)

    # TODO: Add raw shape measurement to data analysis file
    # print(train_raw.shape)
    # print(test_raw.shape)
    # print(validation_raw.shape)
    # print()
    # Draw raw histograms
    Plotter.plot_histograms(
        train_raw,
        test_raw,
        validation_raw,
        FilePath.hitogram_raw_path,
    )

    # TODO: count how many rows were dropped
    # train_raw = train_raw.drop_duplicates().reset_index(drop=True)
    # test_raw = test_raw.drop_duplicates().reset_index(drop=True)
    # validation_raw = validation_raw.drop_duplicates().reset_index(drop=True)

    # TODO: count how many rows were dropped
    remove_threshold = 0.8
    train_removed_similarities = remove_similar_rows(train_raw, threshold=remove_threshold)
    test_removed_similarities = remove_similar_rows(test_raw, threshold=remove_threshold)
    validation_removed_similarities = remove_similar_rows(validation_raw, threshold=remove_threshold)

    # print(train_removed_similarities.shape)
    # print(test_removed_similarities.shape)
    # print(validation_removed_similarities.shape)
    # print()

    # TODO: count how many rows were dropped
    top_n_percetage = 2
    train_removed_top = remove_top_n_percent(train_removed_similarities, top_n_percetage)
    test_removed_top = remove_top_n_percent(test_removed_similarities, top_n_percetage)
    validation_removed_top = remove_top_n_percent(validation_removed_similarities, top_n_percetage)

    # print(train_removed_top.shape)
    # print(test_removed_top.shape)
    # print(validation_removed_top.shape)
    # print()
    train_removed_top["text"] = train_removed_top["text"].apply(filter_repeated_rows)
    test_removed_top["text"] = test_removed_top["text"].apply(filter_repeated_rows)
    validation_removed_top["text"] = validation_removed_top["text"].apply(filter_repeated_rows)

    train_filtered = prepare_data(train_removed_top)
    test_filtered = prepare_data(test_removed_top)
    validation_filtered = prepare_data(validation_removed_top)

    mean_length_train = calculate_mean(train_filtered)
    mean_length_test = calculate_mean(test_filtered)
    mean_length_validation = calculate_mean(validation_filtered)

    # TODO: Add means measurement to data analysis file
    logger.info(f"Średnia długość recenzji w zbiorze treningowym: {mean_length_train:.2f}")
    logger.info(f"Średnia długość recenzji w zbiorze testowym: {mean_length_test:.2f}")
    logger.info(f"Średnia długość recenzji w zbiorze walidacyjnym: {mean_length_validation:.2f}")

    # Draw filtered histograms
    Plotter.plot_histograms(
        train_filtered,
        test_filtered,
        validation_filtered,
        FilePath.hitogram_filtered_path,
    )

    return combine_datasets(train_filtered, test_filtered, validation_filtered)
