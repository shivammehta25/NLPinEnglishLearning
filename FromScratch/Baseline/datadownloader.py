"""
File used to download dataset Use python datadownloader.py -h
for more information
"""
import os
import tqdm
import sys
import argparse
import logging

import urllib.request as req

from config.data import DATA_FOLDER, DATA_FOLDER_RAW, RAW_FILENAMES, DATASETS
from config.root import LOGGING_LEVEL, LOGGING_FORMAT

CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_WORKING_DIRECTORY = os.path.abspath(os.getcwd())

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def download_dataset(dataset_name):
    """
    Downloads the dataset which is passed as parameter
    Input:
    dataset_name: string
    Returns: None
    """

    logger.info("Downloading {}".format(dataset_name))
    output_path = os.path.join(DATA_FOLDER, DATA_FOLDER_RAW)

    if not os.path.exists(output_path):
        logger.debug("Folders doesn't exists creating it")
        os.makedirs(output_path)

    dataset_name = dataset_name.upper()
    if DATASETS[dataset_name]["train"]:
        train_filename, _ = req.urlretrieve(
            url=DATASETS[dataset_name]["train"],
            filename=os.path.join(output_path, RAW_FILENAMES[dataset_name]["train"]),
        )
        logger.debug("Downloaded Train set -> {}".format(train_filename))

    if DATASETS[dataset_name]["test"]:
        test_filename, _ = req.urlretrieve(
            url=DATASETS[dataset_name]["test"],
            filename=os.path.join(output_path, RAW_FILENAMES[dataset_name]["test"]),
        )
        logger.debug("Downloaded Test set -> {}".format(test_filename))

    if DATASETS[dataset_name]["valid"]:
        valid_filename, _ = req.urlretrieve(
            url=DATASETS[dataset_name]["valid"],
            filename=os.path.join(output_path, RAW_FILENAMES[dataset_name]["valid"]),
        )
        logger.debug("Downloaded Valid Set -> {}".format(valid_filename))

    logger.info("Files Downloaded Successfully!")


def already_exists(dataset_name):
    """
    Checks if the raw data exists already
    Returns:
    True if raw data present otherwise false
    """
    raw_data_directory = os.path.join(DATA_FOLDER, DATA_FOLDER_RAW)

    if not os.path.exists(raw_data_directory):
        logger.debug(
            "Directory already doesnt exists creating {}".format(raw_data_directory)
        )
        os.makedirs(raw_data_directory)
    else:
        logger.debug("Directory already exists {}".format(raw_data_directory))

    if (
        os.path.exists(
            os.path.join(raw_data_directory, RAW_FILENAMES[dataset_name]["train"])
        )
        or os.path.exists(
            os.path.join(raw_data_directory, RAW_FILENAMES[dataset_name]["test"])
        )
        or os.path.exists(
            os.path.join(
                raw_data_directory,
                RAW_FILENAMES[dataset_name]["valid"]
                if RAW_FILENAMES[dataset_name]["valid"]
                else "valid",
            )
        )
    ):
        logger.warn(
            "Train: {} or Test: {} or Valid: {} files already exists!".format(
                RAW_FILENAMES[dataset_name]["train"],
                RAW_FILENAMES[dataset_name]["test"],
                RAW_FILENAMES[dataset_name]["valid"],
            )
        )

        return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to download datasets currently available datasets: {}".format(
            ",".join(DATASETS.keys())
        )
    )

    parser.add_argument("-d", "--dataset", default="SQUAD", help="Name of Dataset")
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Skip Directory check and force override download",
    )

    args = parser.parse_args()

    assert args.dataset.upper() in DATASETS.keys(), "Invalid Dataset Selected"

    logger.debug("All Arguments: {}".format(args))
    if args.force:
        download_dataset(args.dataset)
    else:
        if already_exists(args.dataset):
            ch = input(
                "Are you sure you want to discard your present files and override? [y/n]: "
            )
            if ch.lower() == "y" or ch.lower() == "yes":
                download_dataset(args.dataset)
            else:
                logger.info("Not Downloading dataset as dataset already present")
                sys.exit(0)
        else:
            logger.debug("Downloading : {}".format(args.dataset))
            download_dataset(args.dataset)
