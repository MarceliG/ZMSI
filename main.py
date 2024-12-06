from src import (
    DatasetPreprocessor,
    FilePath,
    Tokenizer,
    TrainerModel,
    download_dataset,
    load_dataset_from_path,
    load_model_from_disc,
    logger,
    parser,
    save_dataset,
)


def main() -> None:
    """Execute The main function."""
    logger.info("Start application")
    args = parser()

    if args.download:
        dataset = download_dataset(name="legacy-datasets/allegro_reviews")
        save_dataset(dataset, FilePath.datasets_raw)

    if args.preprocessing:
        dataset_raw = load_dataset_from_path(FilePath.datasets_raw)
        preprocessor = DatasetPreprocessor(dataset_raw)
        dataset_preprocessed = preprocessor.preprocess()
        save_dataset(dataset_preprocessed, FilePath.dataset_preprocessed)

    if args.train or args.evaluate:
        loaded_dataset = load_dataset_from_path(FilePath.dataset_preprocessed)
        max_length = 512
        batch_size = 64
        model_name = "distilbert/distilbert-base-uncased"
        tokenizer = Tokenizer(model_name)
        trainer = TrainerModel(model_name)

        tokenized_train_dataset = tokenizer.tokenize(loaded_dataset["train"], max_length=max_length)
        tokenized_test_dataset = tokenizer.tokenize(loaded_dataset["test"], max_length=max_length)
        tokenized_validation_dataset = tokenizer.tokenize(loaded_dataset["validation"], max_length=max_length)

    if args.train:
        trainer.train(
            tokenized_train_dataset=tokenized_train_dataset,
            tokenized_validation_dataset=tokenized_validation_dataset,
            model_path_to_save=FilePath.model_2_classes,
            num_labels=2,
            epoch=10,
            batch_size=batch_size,
            learning_rate=2e-5,
        )

    if args.evaluate:
        model = load_model_from_disc(FilePath.model_2_classes)

        trainer.evaluate_trained_model(
            model,
            tokenized_dataset=tokenized_test_dataset,
            result_path_to_save=FilePath.classification_report_path,
            batch_size=8,
        )

    logger.info("Finish application")


if __name__ == "__main__":
    main()
