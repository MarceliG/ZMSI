import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    @staticmethod
    def plot_histograms(train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame, file_path: str) -> None:
        """
        Plot three histograms of text lengths for train, test, and validation datasets on a single figure.

        Args:
            train (pd.DataFrame): DataFrame containing the train dataset.
            test (pd.DataFrame): DataFrame containing the test dataset.
            validation (pd.DataFrame): DataFrame containing the validation dataset.
            file_path (str): Path where the generated plot will be saved.
        """
        min_length = min(
            train["text_length"].min(),
            test["text_length"].min(),
            validation["text_length"].min(),
        )
        max_length = max(
            train["text_length"].max(),
            test["text_length"].max(),
            validation["text_length"].max(),
        )

        x_ticks = range(min_length, max_length + 1, (max_length - min_length) // 10)

        plt.figure(figsize=(10, 12))

        xlabel_name = "Długość recenzji [słowa]"
        ylabel_name = "Liczba recenzji"
        bins = 100
        # Histogram for the train dataset
        plt.subplot(3, 1, 1)
        plt.hist(train["text_length"], bins=bins, alpha=0.7, label="Train")
        plt.title("Histogram długości recenzji - Train")
        plt.xlabel(xlabel_name)
        plt.ylabel(ylabel_name)
        plt.xticks(x_ticks)
        plt.grid()

        # Histogram for the test dataset
        plt.subplot(3, 1, 2)
        plt.hist(test["text_length"], bins=bins, alpha=0.7, label="Test")
        plt.title("Histogram długości recenzji - Test")
        plt.xlabel(xlabel_name)
        plt.ylabel(ylabel_name)
        plt.xticks(x_ticks)
        plt.grid()

        # Histogram for the validation dataset
        plt.subplot(3, 1, 3)
        plt.hist(validation["text_length"], bins=bins, alpha=0.7, label="Validation")
        plt.title("Histogram długości recenzji - Validation")
        plt.xlabel(xlabel_name)
        plt.ylabel(ylabel_name)
        plt.xticks(x_ticks)
        plt.grid()

        plt.tight_layout()

        plt.savefig(file_path)
        plt.close()
