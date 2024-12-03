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

    if args.train:
        dataset = load_dataset_from_path(FilePath.dataset_preprocessed)
        tokenizer = Tokenizer()

        max_length = 512
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
            epoch=10,
            batch_size=8,
            learning_rate=5e-5,
        )

    if args.evaluate:
        dataset = load_dataset_from_path(FilePath.dataset_preprocessed)
        tokenizer = Tokenizer()
        model = load_model_from_disc(FilePath.models)
        max_length = 512
        tokenized_test_dataset = tokenizer.tokenize(dataset["test"], max_length=max_length)
        trainer = TrainerModel()
        trainer.evaluate_trained_model(
            model,
            tokenized_dataset=tokenized_test_dataset,
            result_path_to_save=FilePath.classification_report_path,
            batch_size=8,
        )

    logger.info("Finish application")


if __name__ == "__main__":
    main()
