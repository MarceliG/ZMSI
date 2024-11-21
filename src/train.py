import logging

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import FilePath
from src.logs import logger


class TrainerModel:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
    ) -> None:
        """
        Initialize the TrainerModel class.

        Args:
            model_name (str): Name of the pre-trained model to be used for training.
                Defaults to 'bert-base-uncased'.
        """
        self.model_name = model_name

    def train(
        self,
        tokenized_train_dataset: Dataset,
        tokenized_validation_dataset: Dataset,
        model_path_to_save: str,
        num_labels: int = 5,
        epoch: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 5e-5,
    ) -> None:
        """
        Train the BERT model using the preprocessed dataset.

        Args:
            tokenized_train_dataset (Dataset): The tokenized train dataset.
            tokenized_validation_dataset (Dataset): The tokenized validation dataset.
            model_path_to_save (str): The path where model will be save.
            num_labels (int): Number of unique labels (classes) for classification. Defaults to 5.
            epoch (int): Number of training epochs. Defaults to 3.
            batch_size (int): Batch size per device during training. Defaults to 8.
            gradient_accumulation_steps (int): Number of steps to accumulate gradients
                before performing a backward/update pass. Defaults to 2.
            learning_rate (float): The learning rate used by the optimizer to update model weights during training.
                It controls how big each step will be in the gradient descent process. Typical values are in the range
                of 1e-5 to 5e-5 for transformer models like BERT. Defaults to 5e-5.
        """
        logger.info("Start training model")

        # Initialize the model
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logging.info(f"Set model to use device: {device}")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=FilePath.results,
            num_train_epochs=epoch,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            save_steps=500,
            eval_strategy="steps",
            eval_steps=500,
        )

        # Train the model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_validation_dataset,
        )
        trainer.train()

        # Save the trained model
        model.save_pretrained(model_path_to_save)

        logger.info("Model training completed")

    def evaluate_trained_model(
        self,
        model: BertForSequenceClassification,
        tokenized_dataset: Dataset,
        result_path_to_save: str,
        batch_size: int = 8,
    ) -> None:
        """
        Evaluate the trained model on the test dataset and generate a classification report.

        Args:
            model (str): Path where the trained model is saved.
            tokenized_dataset (Dataset): The tokenized test dataset.
            result_path_to_save (str): The path wrehe result will be saved.
            batch_size (int): Batch size per device during training. Defaults to 8.
        """
        logger.info("Start evaluation")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        data_collator = DataCollatorWithPadding(
            tokenizer=BertTokenizer.from_pretrained(self.model_name), return_tensors="pt"
        )
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_predictions, digits=4, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_markdown(result_path_to_save)
        logger.info("Classification Report:\n", report_df)
