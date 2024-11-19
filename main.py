from src import Tokenizer, TrainerModel
from src.config import FilePath, parser
from src.logs import logger
from src.preprocessing import preprocessing
from src.resources import download_dataset, load_dataset_from_path, load_model_from_disc, save_dataset


def main() -> None:
    """Execute The main function."""
    logger.info("Start application")
    args = parser()

    if args.download:
        dataset = download_dataset(name="legacy-datasets/allegro_reviews")
        save_dataset(dataset, FilePath.datasets_raw)

    if args.preprocessing:
        dataset_raw = load_dataset_from_path(FilePath.datasets_raw)
        dataset_preprocessed = preprocessing(dataset_raw)
        save_dataset(dataset_preprocessed, FilePath.dataset_preprocessed)

    if args.train:
        dataset = load_dataset_from_path(FilePath.dataset_preprocessed)
        tokenizer = Tokenizer()

        max_length = 128
        tokenized_train_dataset = tokenizer.tokenize(dataset["train"], max_length=max_length)  # 512 za długo się uczy
        tokenized_test_dataset = tokenizer.tokenize(dataset["test"], max_length=max_length)  # 512 za długo się uczy
        tokenized_validation_dataset = tokenizer.tokenize(
            dataset["validation"], max_length=max_length
        )  # 512 za długo się uczy
        trainer = TrainerModel()
        trainer.train(
            tokenized_train_dataset=tokenized_train_dataset,
            tokenized_validation_dataset=tokenized_validation_dataset,
            model_path_to_save=FilePath.models,
            num_labels=5,
            epoch=5,
            batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
        )
        model = load_model_from_disc(FilePath.models)
        trainer.evaluate_trained_model(
            model,
            tokenized_dataset=tokenized_test_dataset,
            result_path_to_save=FilePath.classification_report_path,
            batch_size=8,
        )

    logger.info("Finish application")


if __name__ == "__main__":
    main()
