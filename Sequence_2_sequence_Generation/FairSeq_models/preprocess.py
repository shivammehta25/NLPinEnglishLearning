"""
Preprocessing of raw data run this file by

```
    >>> python preprocessdata.py  
    >>> python preproecssdata.py --location <datasetLocation>
```
"""
import argparse
import csv
import logging
import os
import re
import time
import shutil

import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from utility import run_command
from config.data import (
    DATASET_FOLDER,
    PROCESSED_DATASET,
    PROCESSED_DATASET_FOLDER,
    FAIRSEQ_PREPROCESSED_DATASET,
    RAW_DATASET_LOCATION,
    RAW_DATASET_FILENAMES,
)
from config.hyperparameters import SRC_WORDS, TRG_WORDS
from config.root import LOGGING_FORMAT, LOGGING_LEVEL, SEED, seed_all

nlp = spacy.load("en")
seed_all(SEED)

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


class PreProcessDataset:
    """
    Class to preprocess dataset takes input location as input
    otherwise will use configuration location 
    Uses fairseq Preprocessing to generate dataset to binaries
    """

    def __init__(self, location):
        if location:
            self.dataset_location = location
        else:
            self.dataset_location = RAW_DATASET_LOCATION

    def preprocess(self):
        """Preprocesses the dataset"""

        # shutil.copyfile(original, target)

        if not os.path.exists(os.path.join(DATASET_FOLDER, PROCESSED_DATASET_FOLDER)):
            os.mkdir(os.path.join(DATASET_FOLDER, PROCESSED_DATASET_FOLDER))

        shutil.copyfile(
            os.path.join(self.dataset_location, RAW_DATASET_FILENAMES["train"][0]),
            PROCESSED_DATASET["train"] + ".sentence",
        )
        shutil.copyfile(
            os.path.join(self.dataset_location, RAW_DATASET_FILENAMES["train"][1]),
            PROCESSED_DATASET["train"] + ".question",
        )

        shutil.copyfile(
            os.path.join(self.dataset_location, RAW_DATASET_FILENAMES["valid"][0]),
            PROCESSED_DATASET["valid"] + ".sentence",
        )
        shutil.copyfile(
            os.path.join(self.dataset_location, RAW_DATASET_FILENAMES["valid"][1]),
            PROCESSED_DATASET["valid"] + ".question",
        )

        shutil.copyfile(
            os.path.join(self.dataset_location, RAW_DATASET_FILENAMES["test"][0]),
            PROCESSED_DATASET["test"] + ".sentence",
        )
        shutil.copyfile(
            os.path.join(self.dataset_location, RAW_DATASET_FILENAMES["test"][1]),
            PROCESSED_DATASET["test"] + ".question",
        )

        logger.info(
            "Running FairSeq Preprocessing to convert files into fairseq binaries"
        )

        if os.path.exists(FAIRSEQ_PREPROCESSED_DATASET):
            logger.debug("Old Binaries present deleting them")
            shutil.rmtree(FAIRSEQ_PREPROCESSED_DATASET)
            logger.debug("Deleted old binaries now generating new one's")

        pre_process_command = "fairseq-preprocess --source-lang sentence --target-lang question \
                              --trainpref {} --testpref {} \
                              --validpref {} --destdir {} --seed {} \
                              --nwordssrc {} --nwordstgt {}".format(
            PROCESSED_DATASET["train"],
            PROCESSED_DATASET["test"],
            PROCESSED_DATASET["valid"],
            FAIRSEQ_PREPROCESSED_DATASET,
            SEED,
            SRC_WORDS,
            TRG_WORDS,
        )

        run_command(pre_process_command)


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description="Utility to preprocess the dataset")

    parser.add_argument(
        "-l",
        "--location",
        default=None,
        help="Location of Dataset if left empty configuration will be used",
    )

    args = parser.parse_args()

    preprocessor = PreProcessDataset(args.location)

    preprocessor.preprocess()

    logger.debug(
        "Utility Finished Execution in: {:.4f}ms".format(time.time() - start_time)
    )
