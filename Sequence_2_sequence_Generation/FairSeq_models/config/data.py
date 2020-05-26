"""
Configuration for Datasets
"""
import os


RAW_DATASET_FILENAMES = {
    "train": ("src-train.txt", "tgt-train.txt"),
    "valid": ("src-dev.txt", "tgt-dev.txt"),
    "test": ("src-test.txt", "tgt-test.txt"),
}
DATASET_FOLDER = "data"
RAW_DATASET_FOLDER = "raw"
PROCESSED_DATASET_FOLDER = "processed"
PROCESSED_DATASET_TRAIN_FILENAME = "train"
PROCESSED_DATASET_VALID_FILENAME = "valid"
PROCESSED_DATASET_TEST_FILENAME = "test"
FAIRSEQ_PREPROCESSED_DATASET = os.path.join(DATASET_FOLDER, "fairseq_binaries")
RAW_DATASET_LOCATION = os.path.join(DATASET_FOLDER, RAW_DATASET_FOLDER)
PROCESSED_DATASET = {
    "train": os.path.join(
        DATASET_FOLDER, PROCESSED_DATASET_FOLDER, PROCESSED_DATASET_TRAIN_FILENAME
    ),
    "valid": os.path.join(
        DATASET_FOLDER, PROCESSED_DATASET_FOLDER, PROCESSED_DATASET_VALID_FILENAME
    ),
    "test": os.path.join(
        DATASET_FOLDER, PROCESSED_DATASET_FOLDER, PROCESSED_DATASET_TEST_FILENAME
    ),
}

TEMP_DIR = ".temp"
