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

    # files
    hitogram_raw_path = os.path.join(plots, "histograms_raw.png")
    hitogram_filtered_path = os.path.join(plots, "histograms_filtered.png")

    # Create neccesery folders
    for directory in [data_path, models, datasets, datasets_raw, dataset_preprocessed, plots, results]:
        if not os.path.exists(directory):
            os.makedirs(directory)
