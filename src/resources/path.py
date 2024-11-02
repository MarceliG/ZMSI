import os


class FilePath:
    current_path = os.getcwd()
    data_path = os.path.join(current_path, "data")
    hitogram_path = os.path.join(data_path, "histograms.png")

    # Create neccesery folders
    for directory in [data_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
