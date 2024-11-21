import os


class FilePath:
    current_path = os.getcwd()
    # directories
    data_path = os.path.join(current_path, "data")
    datasets = os.path.join(data_path, "datasets")
    dataset_preprocessed = os.path.join(datasets, "preprocessed")
    datasets_raw = os.path.join(datasets, "raw")
    models = os.path.join(data_path, "models")
    plots = os.path.join(data_path, "plots")
    results = os.path.join(data_path, "results")
    classification_report = os.path.join(results, "classification_reports")
    statistics = os.path.join(data_path, "statistics")

    # files
    histogram_raw_path = os.path.join(plots, "histograms_raw.png")
    histogram_filtered_path = os.path.join(plots, "histograms_filtered.png")

    statistic_before_preprocessing_path = os.path.join(statistics, "before_preprocessing.md")
    statistic_during_preprocessing_path = os.path.join(statistics, "during_preprocessing.md")
    statistic_after_preprocessing_path = os.path.join(statistics, "after_preprocessing.md")

    train_raw_rating_value_counts = os.path.join(statistics, "train_raw_rating_value_counts.md")
    test_raw_rating_value_counts = os.path.join(statistics, "test_raw_rating_value_counts.md")
    validation_raw_rating_value_counts = os.path.join(statistics, "validation_raw_rating_value_counts.md")

    train_after_spliting_rating_value_counts = os.path.join(statistics, "train_after_spliting_rating_value_counts.md")
    test_after_spliting_rating_value_counts = os.path.join(statistics, "test_after_spliting_rating_value_counts.md")
    validation_after_spliting_rating_value_counts = os.path.join(
        statistics, "validation_after_spliting_rating_value_counts.md"
    )

    classification_report_path = os.path.join(classification_report, "classification_report.md")

    # Create neccesery folders
    for directory in [
        data_path,
        models,
        datasets,
        datasets_raw,
        dataset_preprocessed,
        plots,
        results,
        classification_report,
        statistics,
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)
