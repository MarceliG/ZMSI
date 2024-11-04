from src.logs import logger
from src.preprocessing import preprocessing
from src.resources import FilePath, download_dataset, load_dataset_from_path, parser, save_dataset


def main() -> None:
    """Execute The main function."""
    logger.info("Start application")
    args = parser()

    if args.download:
        dataset = download_dataset(name="legacy-datasets/allegro_reviews")
        save_dataset(dataset, FilePath().datasets_raw)

    if args.preprocessing:
        dataset_raw = load_dataset_from_path(FilePath().datasets_raw)
        dataset_preprocessed = preprocessing(dataset_raw)
        save_dataset(dataset_preprocessed, FilePath().dataset_preprocessed)

    logger.info("Finish application")


if __name__ == "__main__":
    main()