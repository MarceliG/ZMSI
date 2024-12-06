import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.logs import logger


class Tokenizer:
    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        """
        Initialize the Tokenizer class.

        Args:
            model_name (str): Name of the pre-trained model for tokenization.
                Defaults to 'bert-base-uncased'.
        """
        self.tokenization = AutoTokenizer.from_pretrained(model_name)

    def tokenize(
        self,
        dataset: Dataset,
        truncation: bool = True,
        padding: str = "max_length",
        max_length: int = 128,
    ) -> Dataset:
        """
        Tokenizes and preprocesses the dataset for the BERT model.

        Args:
            dataset (Dataset): The dataset to preprocess. Should include 'text' and 'rating' columns.
            truncation (bool): Whether to truncate sequences to the maximum length. Defaults to True.
            padding (str): Padding strategy. Defaults to 'max_length'.
            max_length (int): Maximum sequence length. Defaults to 128.

        Returns:
            Dataset: The preprocessed dataset, ready for training/evaluation.
        """
        logger.info("Start tokenization...")
        tokenized_data = self.tokenization(
            dataset["text"],
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )
        tokenized_df = pd.DataFrame(
            {
                "input_ids": tokenized_data["input_ids"],
                "attention_mask": tokenized_data["attention_mask"],
                "labels": dataset["rating"],
            }
        )
        tokenized_df["labels"] = tokenized_df["labels"].astype(int)
        logger.info("Finish tokenization")
        return Dataset.from_pandas(tokenized_df)
