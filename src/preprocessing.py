import re

import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import FilePath
from src.resources import Plotter, combine_datasets, save_dataframe_as_markdown


class DatasetPreprocessor:
    def __init__(self, dataset: Dataset):
        """
        Class for preprocessing datasets.

        Including splitting data into training, test, and validation sets,
        and generating statistics before, during, and after processing.

        Attributes:
            dataset (Dataset): The input dataset to preprocess.
            statistics_before (pd.DataFrame): Statistics calculated before preprocessing.
            statistics_during (pd.DataFrame): Statistics calculated during preprocessing.
            statistics_after (pd.DataFrame): Statistics calculated after preprocessing.
            statistic_rating (pd.DataFrame): Statistics of the rating column.
            train (pd.DataFrame): The training subset of the dataset.
            test (pd.DataFrame): The testing subset of the dataset.
            validation (pd.DataFrame): The validation subset of the dataset.
        """
        self.dataset = dataset
        self.statistics_before = self._create_statistics_dataframe()
        self.statistics_during = self._create_statistics_during_dataframe()
        self.statistics_after = self._create_statistics_dataframe()
        self.statistic_rating = self._create_statistics_rating_dataframe()
        self.train, self.test, self.validation = self._split_datasets()

    def _split_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        return (
            pd.DataFrame(self.dataset["train"]),
            pd.DataFrame(self.dataset["test"]),
            pd.DataFrame(self.dataset["validation"]),
        )

    @staticmethod
    def _calculate_mean(df: pd.DataFrame) -> float:
        """
        Calculate the average length of text in a given dataset.

        Args:
            df (pd.DataFrame): A DataFrame containing column named 'text'.

        Returns:
            float: The average length of the text entries in the dataset.
        """
        return round(df["text_length"].mean(), 2)

    @staticmethod
    def _calculate_std(df: pd.DataFrame) -> float:
        """
        Calculate the average length of text in a given dataset.

        Args:
            df (pd.DataFrame): A DataFrame containing column named 'text'.

        Returns:
            float: The average length of the text entries in the dataset.
        """
        return round(df["text_length"].std(), 2)

    def _remove_similar_rows(self, df: pd.DataFrame, threshold: float = 0.8) -> tuple[pd.DataFrame, int]:
        """
        Find and remove rows that are very similar based on text similarity.

        Args:
            df (pd.DataFrame): A DataFrame containing a column named 'text'.
            threshold (float): Similarity threshold for considering two texts as similar.

        Returns:
            tuple: A tuple containing the processed DataFrame and the number of removed rows.
        """
        initial_count = len(df)
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
        df_cleaned = df.drop(index=valid_to_drop).reset_index(drop=True)

        removed_count = initial_count - len(df_cleaned)

        return df_cleaned, removed_count

    @staticmethod
    def _count_words(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a column `text_length` to the DataFrame representing the length of each text.

        Args:
            df (pd.DataFrame): A DataFrame containing a 'text' column with text data to be analyzed.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'text_length'.
        """
        df["text_length"] = df["text"].apply(len)
        return df

    def _remove_top_n_percent(self, df: pd.DataFrame, percent: int) -> tuple[pd.DataFrame, int]:
        """
        Remove the top n percent of entries with the longest text lengths from the DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame containing a 'text_length' column.
            percent (int): The percentage of entries to remove (0-100)%.

        Returns:
        tuple: A tuple containing the processed DataFrame and the number of removed rows.
        """
        if not (0 <= percent <= 100):
            msg = "Percentage n must be between 0 and 100."
            raise ValueError(msg)

        initial_count = len(df)
        n_to_drop = int(len(df) * (percent / 100))
        cutoff_length = df.nlargest(n_to_drop, "text_length")["text_length"].min()
        df_cleaned = df[df["text_length"] < cutoff_length].reset_index(drop=True)

        removed_count = initial_count - len(df_cleaned)

        return df_cleaned, removed_count

    def _filter_repeated(self, df: pd.DataFrame, threshold: int) -> tuple[pd.DataFrame, int]:
        """
        Remove rows with repeated words (consecutive repetitions) in the 'text' column based on a threshold.

        Args:
            df (pd.DataFrame): Input DataFrame with a 'text' column.
            threshold (int): Minimum number of consecutive repetitions of a word to consider it repeated.

        Returns:
            tuple: A DataFrame with rows removed and the count of removed rows.
        """
        initial_count = len(df)

        df_filtered = df.assign(
            text=df["text"].apply(
                lambda text: None if re.search(rf"\b(\w+)\b(?:\s+\1){{{threshold - 1},}}", text) else text
            )
        )

        removed_count = initial_count - df_filtered.dropna(subset=["text"]).shape[0]
        df_cleaned = df_filtered.dropna(subset=["text"]).reset_index(drop=True)

        return df_cleaned, removed_count

    def _undersampling(self, df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, int]:
        """
        Perform undersampling by balancing the dataset based on the least frequent class.

        Args:
            df (pd.DataFrame): Input DataFrame containing a target column.
            target_column (str): Name of the column containing class labels.

        Returns:
            tuple: A DataFrame after undersampling and the number of removed rows.
        """
        class_counts = df[target_column].value_counts()
        min_class_count = class_counts.min()

        df_balanced = (
            df.groupby(target_column)
            .apply(lambda group: group.sample(n=min_class_count, random_state=42))
            .reset_index(drop=True)
        )

        removed_count = len(df) - len(df_balanced)
        return df_balanced, removed_count

    def _create_statistics_rating_dataframe(self) -> pd.DataFrame:
        """
        Create dataframe with rating columns.

        Returns:
            pd.DataFrame: A dataframe with colums: rating, train, test, validation.
        """
        return pd.DataFrame(columns=["rating", "train", "test", "validation"])

    def _create_statistics_dataframe(self) -> pd.DataFrame:
        """
        Create dataframe with bacis columns.

        Returns:
            pd.DataFrame: A dataframe with colums: mean, std and row_count.
        """
        return pd.DataFrame(
            columns=["mean_text_length", "std_text_length", "row_count"], index=["train", "test", "validation"]
        )

    def _create_statistics_during_dataframe(self) -> pd.DataFrame:
        """
        Create a dataframe with columns describing the steps performed during preprocessing.

        Returns:
            pd.DataFrame: A dataframe with colums: mean, std and row_count.
        """
        return pd.DataFrame(
            columns=[
                "remove_nan",
                "remove_duplicates",
                "remove_similarities",
                "remove_longest_texts",
                "remove_spam",
                "undersampling",
            ],
            index=["train", "test", "validation"],
        )

    def _update_statistics(self) -> None:
        """Update `stats_df` with statistics from `df`."""
        for name, dataset in zip(["train", "test", "validation"], [self.train, self.test, self.validation]):
            self.statistics_before.loc[name, ["mean_text_length", "std_text_length", "row_count"]] = [
                self._calculate_mean(dataset),
                self._calculate_std(dataset),
                len(dataset),
            ]

    def _update_statistics_during(
        self, column_name: str, removed_count_train: int, removed_count_test: int, removed_count_validation: int
    ) -> None:
        """Update `stats_df` with statistics from `df`."""
        self.statistics_during[column_name] = [removed_count_train, removed_count_test, removed_count_validation]

    def _remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Remove duplicate rows based on the "text" column and calculates the number of removed rows.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: A tuple containing the processed DataFrame and the number of removed rows.
        """
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset="text").reset_index(drop=True)
        removed_count = initial_count - len(df_cleaned)
        return df_cleaned, removed_count

    def _remove_nans(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Remove nan rows and calculates the number of removed rows.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: A tuple containing the processed DataFrame and the number of removed rows.
        """
        initial_count = len(df)
        df_cleaned = df.dropna().reset_index(drop=True)
        removed_count = initial_count - len(df_cleaned)
        return df_cleaned, removed_count

    def split_dataset_by_half(self, df: pd.DataFrame, rating_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    def count_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count unique value in the 'rating' column.

        Args:
            df (pd.DataFrame): The input DataFrame containing a 'rating' column.

        Returns:
            pd.DataFrame: A DataFrame with 'rating' The count of unique value, sorted in descending order.
        """
        counts_df = df["rating"].value_counts().reset_index()
        return counts_df.sort_values(by="rating", ascending=False).reset_index(drop=True)

    def _update_statistic_rating(self) -> None:
        """
        Count unique values in the 'rating' column for train, test, and validation datasets and update self.statistic_rating.

        The resulting DataFrame will have the structure:
        pd.DataFrame(columns=["rating", "train", "test", "validation"])
        """
        train_raw_rating, test_raw_rating, validation_raw_rating = [
            self.count_rating(df) for df in (self.train, self.test, self.validation)
        ]
        train_raw_rating = train_raw_rating.rename(columns={"count": "train"})
        test_raw_rating = test_raw_rating.rename(columns={"count": "test"})
        validation_raw_rating = validation_raw_rating.rename(columns={"count": "validation"})

        combined_df = train_raw_rating.merge(test_raw_rating, on="rating", how="outer")
        combined_df = combined_df.merge(validation_raw_rating, on="rating", how="outer")

        self.statistic_rating = combined_df.sort_values(by="rating", ascending=False).reset_index(drop=True)

    def preprocess(self) -> Dataset:
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
        self._update_statistic_rating()
        self.test, self.validation = self.split_dataset_by_half(self.validation, "rating")
        save_dataframe_as_markdown(self.statistic_rating, FilePath.statistic_rating_before_preprocessing_path)

        self.train, self.test, self.validation = [
            self._count_words(df) for df in (self.train, self.test, self.validation)
        ]
        self._update_statistics()
        save_dataframe_as_markdown(self.statistics_before, FilePath.statistic_before_preprocessing_path)

        (
            (self.train, removed_count_train),
            (self.test, removed_count_test),
            (self.validation, removed_count_validation),
        ) = [self._remove_nans(df) for df in (self.train, self.test, self.validation)]
        self._update_statistics_during("remove_nan", removed_count_train, removed_count_test, removed_count_validation)

        (
            (self.train, removed_count_train),
            (self.test, removed_count_test),
            (self.validation, removed_count_validation),
        ) = [self._remove_duplicates(df) for df in (self.train, self.test, self.validation)]
        self._update_statistics_during(
            "remove_duplicates", removed_count_train, removed_count_test, removed_count_validation
        )

        (
            (self.train, removed_count_train),
            (self.test, removed_count_test),
            (self.validation, removed_count_validation),
        ) = [self._remove_similar_rows(df, threshold=0.8) for df in (self.train, self.test, self.validation)]
        self._update_statistics_during(
            "remove_similarities", removed_count_train, removed_count_test, removed_count_validation
        )

        (
            (self.train, removed_count_train),
            (self.test, removed_count_test),
            (self.validation, removed_count_validation),
        ) = [self._remove_top_n_percent(df, 2) for df in (self.train, self.test, self.validation)]
        self._update_statistics_during(
            "remove_longest_texts", removed_count_train, removed_count_test, removed_count_validation
        )

        (
            (self.train, removed_count_train),
            (self.test, removed_count_test),
            (self.validation, removed_count_validation),
        ) = [self._filter_repeated(df, threshold=3) for df in (self.train, self.test, self.validation)]
        self._update_statistics_during("remove_spam", removed_count_train, removed_count_test, removed_count_validation)

        (
            (self.train, removed_count_train),
            (self.test, removed_count_test),
            (self.validation, removed_count_validation),
        ) = [self._undersampling(df, target_column="rating") for df in (self.train, self.test, self.validation)]
        self._update_statistics_during(
            "undersampling", removed_count_train, removed_count_test, removed_count_validation
        )

        save_dataframe_as_markdown(self.statistics_during, FilePath.statistic_during_preprocessing_path)

        self._update_statistics()
        save_dataframe_as_markdown(self.statistics_during, FilePath.statistic_after_preprocessing_path)
        self._update_statistic_rating()
        save_dataframe_as_markdown(self.statistic_rating, FilePath.statistic_rating_after_preprocessing_path)

        save_dataframe_as_markdown(self.statistics_before, FilePath.statistic_after_preprocessing_path)
        Plotter.plot_histograms(self.train, self.test, self.validation, FilePath.histogram_filtered_path)
        return combine_datasets(self.train, self.test, self.validation)
