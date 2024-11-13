import pandas as pd
import torch
from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

from src.config import FilePath
from src.logs import logger
from src.resources import load_dataset_from_path


class Train:
    def __init__(self, dataset_path: str) -> None:
        """
        Initialize the Train class.

        Args:
            dataset_path (str): The path to the dataset.

        Returns:
            None
        """
        dataset = load_dataset_from_path(dataset_path)
        if dataset is None:
            msg = "Dataset not found"
            raise ValueError(msg)

        self.dataset = dataset
        self.pretrained_model_name_or_path = "bert-base-uncased"

    def tokenization(self, dataset: Dataset) -> Dataset:
        """
        Tokenizes and preprocesses the dataset.

        Args:
            self.pretrained_model_name_or_path (str): The name or path of the pre-trained model.
            dataset (Dataset): The dataset to preprocess.

        Returns:
            Dataset: The preprocessed dataset.
        """
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        tokenized_data = tokenizer(dataset["text"], truncation=True, padding="max_length", max_length=128)
        tokenized_df = pd.DataFrame(
            {
                "input_ids": tokenized_data["input_ids"],
                "attention_mask": tokenized_data["attention_mask"],
                "labels": dataset["rating"],
            }
        )

        tokenized_df["labels"] = tokenized_df["labels"].astype(int) - 1

        return Dataset.from_pandas(tokenized_df)

    def train_model(self) -> None:
        """
        Train the model using the dataset at the given path.

        Args:
            self.dataset (Dataset): The dataset to use for training.
            self.pretrained_model_name_or_path (str): The name or path of the pre-trained model.

        Returns:
            None
        """
        logger.info("Start training model")

        tokenized_train_dataset = self.tokenization(self.dataset["train"])
        tokenized_test_dataset = self.tokenization(self.dataset["test"])

        # Initialize the model
        model = BertForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path, num_labels=5)

        training_args = TrainingArguments(
            output_dir=FilePath.results,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            fp16=torch.cuda.is_available(),
        )

        # Train the model
        trainer = Trainer(
            model=model, args=training_args, train_dataset=tokenized_train_dataset, eval_dataset=tokenized_test_dataset
        )
        trainer.train()

        model.save_pretrained(FilePath.models)

        logger.info("Model training completed")
