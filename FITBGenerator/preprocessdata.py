"""
Preprocessing of raw data run this file by

```
    >>> python preprocessdata.py  
    >>> python preproecssdata.py --location <datasetLocation>
```
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from config.data import (
    DATASET_FOLDER,
    PROCESSED_DATASET,
    PROCESSED_DATASET_FOLDER,
    RAW_DATASET,
)
from config.root import LOGGING_FORMAT, LOGGING_LEVEL, SEED

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

        self.dataset = pd.read_csv(self.dataset_location, delimiter="\t")

    def preprocess(self):
        """Preprocesses the dataset"""
        # Changing ____ to <blank/> tags
        self.dataset["Question"] = self.dataset["Question"].str.replace(
            r"[_]{2,}", "<blank>"
        )
        # Removing brackets
        self.dataset["Question"] = self.dataset["Question"].str.replace(r"[\)\(]", "")
        # Stripping whitespaces
        self.dataset["Question"] = self.dataset["Question"].apply(lambda x: x.strip())
        # Replacing / to <slash/> tags
        self.dataset["Question"] = self.dataset["Question"].str.replace(
            r"\/", "<slash>"
        )

        if not os.path.exists(os.path.join(DATASET_FOLDER, PROCESSED_DATASET_FOLDER)):
            os.mkdir(os.path.join(DATASET_FOLDER, PROCESSED_DATASET_FOLDER))

        # self.dataset["label"] = (
        #     " <Q_S> "
        #     + self.dataset["Question"]
        #     + " </Q_S> "
        #     + " <K_S> "
        #     + self.dataset["key"]
        #     + " </K_S> "
        #     + " <A_S> "
        #     + self.dataset["answer"]
        #     + " </A_S> "
        # )

        self.trainset, self.testset = train_test_split(
            self.dataset, test_size=0.15, random_state=SEED
        )

        self.trainset.to_csv(PROCESSED_DATASET["train"], index=False, sep="\t")
        self.testset.to_csv(PROCESSED_DATASET["test"], index=False, sep="\t")

        logger.debug(
            "Saving the file preprocessed files to : {}".format(
                PROCESSED_DATASET_FOLDER
            )
        )


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
