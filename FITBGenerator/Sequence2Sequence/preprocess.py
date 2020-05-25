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
    RAW_DATASET,
    FAIRSEQ_PREPROCESSED_DATASET,
)
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
    """

    def __init__(self, location):
        if location:
            self.dataset_location = location
        else:
            self.dataset_location = RAW_DATASET

        self.dataset = {"sentence": [], "question": []}

    def preprocess(self):
        """Preprocesses the dataset"""
        with open(RAW_DATASET) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter="\t")
            for row in csv_reader:
                if "_" in row["Question"]:
                    row["Question"] = re.sub(r"[_]{2,}", "_", row["Question"])

                    self.dataset["sentence"].append(row["answer"].lstrip().strip())
                    self.dataset["question"].append(row["Question"].strip())

        logger.debug("DataSet Preprocessed Successfully!")

        self.dataset = pd.DataFrame.from_dict(self.dataset)

        # Test Split
        self.trainset, self.testset = train_test_split(
            self.dataset, test_size=0.15, random_state=SEED
        )

        # Valid Split
        self.trainset, self.validset = train_test_split(
            self.trainset, test_size=0.10, random_state=SEED
        )

        self.trainset["sentence"].to_csv(
            "{}.sentence".format(PROCESSED_DATASET["train"]),
            index=False,
            sep="\t",
            header=False,
        )
        self.validset["sentence"].to_csv(
            "{}.sentence".format(PROCESSED_DATASET["valid"]),
            index=False,
            sep="\t",
            header=False,
        )
        self.testset["sentence"].to_csv(
            "{}.sentence".format(PROCESSED_DATASET["test"]),
            index=False,
            sep="\t",
            header=False,
        )
        self.trainset["question"].to_csv(
            "{}.question".format(PROCESSED_DATASET["train"]),
            index=False,
            sep="\t",
            header=False,
        )
        self.validset["question"].to_csv(
            "{}.question".format(PROCESSED_DATASET["valid"]),
            index=False,
            sep="\t",
            header=False,
        )

        self.testset["question"].to_csv(
            "{}.question".format(PROCESSED_DATASET["test"]),
            index=False,
            sep="\t",
            header=False,
        )

        logger.debug(
            "Trainset Size: {}, Validset Size: {}, Tesetset Size: {}".format(
                self.trainset.shape, self.validset.shape, self.testset.shape
            )
        )

        logger.debug(
            "Saving the file preprocessed files to : {}".format(
                PROCESSED_DATASET_FOLDER
            )
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
                              --nwordssrc 5000 --nwordstgt 5000".format(
            PROCESSED_DATASET["train"],
            PROCESSED_DATASET["test"],
            PROCESSED_DATASET["valid"],
            FAIRSEQ_PREPROCESSED_DATASET,
            SEED,
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
