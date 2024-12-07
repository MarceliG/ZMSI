import os


class FilePath:
    current_path = os.getcwd()
    # directories
    data_path = os.path.join(current_path, "data")
    datasets = os.path.join(data_path, "datasets")
    dataset_preprocessed = os.path.join(datasets, "preprocessed")
    datasets_raw = os.path.join(datasets, "raw")
    models = os.path.join(data_path, "models")
    model_2_classes = os.path.join(models, "model_2_classes")
    plots = os.path.join(data_path, "plots")
    results = os.path.join(data_path, "results")
    classification_report = os.path.join(results, "classification_reports")
    statistics = os.path.join(data_path, "statistics")
    statistics_before = os.path.join(statistics, "statistics_before")
    statistics_during = os.path.join(statistics, "statistics_during")
    statistics_after = os.path.join(statistics, "statistics_after")

    # files
    histogram_raw_path = os.path.join(plots, "histograms_raw.png")
    histogram_filtered_path = os.path.join(plots, "histograms_filtered.png")

    statistic_before_preprocessing_path = os.path.join(statistics_before, "before.md")
    statistic_during_preprocessing_path = os.path.join(statistics_during, "during.md")
    statistic_after_preprocessing_path = os.path.join(statistics_after, "after.md")

    statistic_rating_before_preprocessing_path = os.path.join(statistics_before, "rating.md")
    statistic_rating_after_preprocessing_path = os.path.join(statistics_after, "rating.md")
    statistic_rating_after_change_label_path = os.path.join(statistics_after, "rating_after_change_label.md")

    classification_report_path = os.path.join(classification_report, "classification_report.md")

    # Create neccesery folders
    for directory in [
        data_path,
        models,
        model_2_classes,
        datasets,
        datasets_raw,
        dataset_preprocessed,
        plots,
        results,
        classification_report,
        statistics,
        statistics_before,
        statistics_during,
        statistics_after,
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)
