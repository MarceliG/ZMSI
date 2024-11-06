import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from src.resources import FilePath, load_dataset_from_path
from datasets import Dataset

from src.logs import logger

def preprocess_data(dataset: Dataset) -> Dataset:
    """Tokenizes and preprocesses the dataset."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_data = tokenizer(dataset["text"], truncation=True, padding="max_length", max_length=128)
    
    # Konwertowanie tokenizowanych danych do ramki danych
    tokenized_df = pd.DataFrame(
        {
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
            "labels": dataset["rating"],
        }
    )

    tokenized_df["labels"] = tokenized_df["labels"].astype(int) - 1

    return Dataset.from_pandas(tokenized_df)

def train_model(dataset_path: str) -> None:
    """Train the model using the dataset at the given path."""
    logger.info("Start training model")
    
    # Load the dataset
    dataset = load_dataset_from_path(dataset_path)
    if dataset is None:
        raise ValueError("Dataset not found")
    
    tokenized_datasets = preprocess_data(dataset['train']) # @TODO czy na pewno tak ma być?

    # Initialize the model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    training_args = TrainingArguments(
        output_dir=FilePath().results, num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=2, fp16=torch.cuda.is_available()
    )

    # Train the model
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets)
    trainer.train()

    model.save_pretrained(FilePath().models)

    logger.info("Model training completed")