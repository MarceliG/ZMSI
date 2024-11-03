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
    train_raw = train_raw.drop_duplicates().reset_index(drop=True)
    test_raw = test_raw.drop_duplicates().reset_index(drop=True)
    validation_raw = validation_raw.drop_duplicates().reset_index(drop=True)

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

    mean_length_train = calculate_mean(train_removed_top)
    mean_length_test = calculate_mean(test_removed_top)
    mean_length_validation = calculate_mean(validation_removed_top)

    # TODO: Add means measurement to data analysis file
    logger.info(f"Średnia długość recenzji w zbiorze treningowym: {mean_length_train:.2f}")
    logger.info(f"Średnia długość recenzji w zbiorze testowym: {mean_length_test:.2f}")
    logger.info(f"Średnia długość recenzji w zbiorze walidacyjnym: {mean_length_validation:.2f}")

    # Draw filtered histograms
    Plotter.plot_histograms(
        train_removed_top,
        test_removed_top,
        validation_removed_top,
        FilePath.hitogram_filtered_path,
    )

    return combine_datasets(train_removed_top, test_removed_top, validation_removed_top)
