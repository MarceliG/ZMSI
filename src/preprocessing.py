import re

import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import FilePath
from src.logs import logger
from src.resources import Plotter, combine_datasets, save_dataframe_as_markdown


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
    return round(df["text_length"].mean(), 2)


def calculate_std(df: pd.DataFrame) -> float:
    """
    Calculate the average length of text in a given dataset.

    Args:
        df (pd.DataFrame): A DataFrame containing column named 'text'.

    Returns:
        float: The average length of the text entries in the dataset.
    """
    return round(df["text_length"].std(), 2)


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


def create_statistics_dataframe() -> pd.DataFrame:
    """
    Create dataframe with bacis columns.

    Returns:
        pd.DataFrame: A dataframe with colums: mean, std and row_count.
    """
    return pd.DataFrame(
        columns=[
            "mean_text_length",
            "std_text_length",
            "row_count",
        ],
        index=["train", "test", "validation"],
    )


def basic_statistics(
    statistics_df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add basic statistical to a DataFrame.

    The function calculates the following statistics for the given train, test, and validation datasets:
    - Mean text length
    - Standard deviation of text length
    - Row count

    Args:
        statistics_df (pd.DataFrame): A DataFrame that will be updated with the calculated statistics.
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The testing dataset.
        validation (pd.DataFrame): The validation dataset.

    Returns:
        tuple: A tuple containing the updated train, test, and validation DataFrames.
    """
    statistics_df["mean_text_length"] = [calculate_mean(df) for df in (train, test, validation)]
    statistics_df["std_text_length"] = [calculate_std(df) for df in (train, test, validation)]
    statistics_df["row_count"] = [df.shape[0] for df in (train, test, validation)]

    return train, test, validation


def count_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count unique value in the 'rating' column.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'rating' column.

    Returns:
        pd.DataFrame: A DataFrame with 'rating' The count of unique value, sorted in descending order.
    """
    counts_df = df["rating"].value_counts().reset_index()
    return counts_df.sort_values(by="rating", ascending=False).reset_index(drop=True)


def split_dataset_by_half(df: pd.DataFrame, rating_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the input DataFrame into two DataFrames by dividing each group of rows into halves.

    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        rating_col (str): The column name used to group rows before splitting.

    Returns:
        tuple[*pd.DataFrame]: Two DataFrames with approximately equal distribution of rows per group.
    """
    df1_groups = []
    df2_groups = []

    for _, group in df.groupby(rating_col):
        half = len(group) // 2

        df1_groups.append(group.iloc[:half])
        df2_groups.append(group.iloc[half:])

    df1 = pd.concat(df1_groups).sample(frac=1, random_state=42).reset_index(drop=True)
    df2 = pd.concat(df2_groups).sample(frac=1, random_state=42).reset_index(drop=True)

    return df1, df2


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
    statistic_before = create_statistics_dataframe()
    statistic_during = pd.DataFrame()
    statistic_after = create_statistics_dataframe()

    train, test, validation = split_datasets(dataset)

    train_raw_rating, test_raw_rating, validation_raw_rating = [count_rating(df) for df in (train, test, validation)]

    save_dataframe_as_markdown(train_raw_rating, FilePath.train_raw_rating_value_counts)
    save_dataframe_as_markdown(test_raw_rating, FilePath.test_raw_rating_value_counts)
    save_dataframe_as_markdown(validation_raw_rating, FilePath.validation_raw_rating_value_counts)

    test, validation = split_dataset_by_half(validation, "rating")

    train_after_spliting_rating, test_after_spliting_rating, validation_after_spliting_rating = [
        count_rating(df) for df in (train, test, validation)
    ]

    save_dataframe_as_markdown(train_after_spliting_rating, FilePath.train_after_spliting_rating_value_counts)
    save_dataframe_as_markdown(test_after_spliting_rating, FilePath.test_after_spliting_rating_value_counts)
    save_dataframe_as_markdown(validation_after_spliting_rating, FilePath.validation_after_spliting_rating_value_counts)

    train, test, validation = [count_words(df) for df in (train, test, validation)]

    train, test, validation = basic_statistics(statistic_before, train, test, validation)
    save_dataframe_as_markdown(statistic_before, FilePath.statistic_before_preprocessing_path)

    # Draw raw histograms
    Plotter.plot_histograms(train, test, validation, FilePath.histogram_raw_path)

    # drop duplicates
    train, test, validation = [df.drop_duplicates().reset_index(drop=True) for df in (train, test, validation)]
    statistic_during["row_count_after_drop_duplicates"] = [df.shape[0] for df in (train, test, validation)]

    # Remove similar rows
    train, test, validation = [remove_similar_rows(df, threshold=0.8) for df in (train, test, validation)]
    statistic_during["row_count_after_remove_similar_rows"] = [df.shape[0] for df in (train, test, validation)]

    # Remove the longest lines (top 2%)
    train, test, validation = [remove_top_n_percent(df, 2) for df in (train, test, validation)]
    statistic_during["row_count_after_remove_top_n_percent"] = [df.shape[0] for df in (train, test, validation)]

    # Filters for repeating words
    train, test, validation = [
        df.assign(text=df["text"].apply(lambda x: filter_repeated_rows(x, 3))) for df in (train, test, validation)
    ]
    statistic_during["row_count_after_filter_repeated_rows"] = [df.shape[0] for df in (train, test, validation)]

    # Remove None rows
    train, test, validation = [df.dropna().reset_index(drop=True) for df in (train, test, validation)]

    # Clean rows
    train["text"], test["text"], validation["text"] = [df["text"].str.strip() for df in (train, test, validation)]

    statistic_during["row_count_after_drop_NaN"] = [df.shape[0] for df in (train, test, validation)]
    save_dataframe_as_markdown(statistic_during, FilePath.statistic_during_preprocessing_path)

    train, test, validation = basic_statistics(statistic_after, train, test, validation)
    save_dataframe_as_markdown(statistic_after, FilePath.statistic_after_preprocessing_path)
    Plotter.plot_histograms(train, test, validation, FilePath.histogram_filtered_path)

    return combine_datasets(train, test, validation)
