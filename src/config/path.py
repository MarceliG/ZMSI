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
    statistics = os.path.join(data_path, "statistics")

    # files
    histogram_raw_path = os.path.join(plots, "histograms_raw.png")
    histogram_filtered_path = os.path.join(plots, "histograms_filtered.png")

    statistic_before_preprocessing_path = os.path.join(statistics, "before_preprocessing.md")
    statistic_during_preprocessing_path = os.path.join(statistics, "during_preprocessing.md")
    statistic_after_preprocessing_path = os.path.join(statistics, "after_preprocessing.md")

    # Create neccesery folders
    for directory in [
        data_path,
        models,
        datasets,
        datasets_raw,
        dataset_preprocessed,
        plots,
        results,
        statistics,
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)
