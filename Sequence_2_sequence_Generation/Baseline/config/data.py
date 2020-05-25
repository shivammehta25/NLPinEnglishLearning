"""
The Configurations Related to and Dataset and their properties
like local filenames and URLs
"""

DATA_FOLDER = "data"
DATA_FOLDER_RAW = "raw"
DATA_FOLDER_PROCESSED = "processed"

RAW_FILENAMES = {
    "SQUAD": {
        "train": "squad_train.json",
        "test": "squad_test.json",
        "valid": "squad_valid.json",
    }
}

SQUAD_NAME = "SQUAD"

DATASETS = {
    "SQUAD": {
        "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "test": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "valid": None,
    }
}
