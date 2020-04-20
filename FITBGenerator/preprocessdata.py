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
import csv

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
from tqdm.auto import tqdm

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

        self.dataset = {"feature": [], "key": []}

    def write_to_dataset(row):
        row["Question"] = [t.text for t in nlp(row["Question"].replace(".","").strip())]
        answer = [t.text for t in nlp(row["answer"].replace(".","").strip())]
        i = 0
        label = []

        while row["Question"] and answer:
            if row["Question"][-1] == answer[-1]:
                row["Question"].pop()
                answer.pop()
            elif row["Question"][-1] == "_":
                row["Question"].pop()
                while row["Question"] and answer and (row["Question"][-1] != answer[-1]):
                    print(row["Question"], answer)
                    label.append(answer.pop())
                break

        if not label:
            while answer:
                label.append(answer.pop())

        dataset["feature"].append(row["answer"].lstrip().strip())
        dataset["key"].append(" ".join(reversed(label)))

    def preprocess(self):
        """Preprocesses the dataset"""

        with open(self.dataset_location) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t')
            for row in tqdm(csv_reader, total=len(csv_reader):
                if "_" in row["Question"]:
                    row["Question"] = re.sub(r"[_]{2,}", "_", row["Question"])

                self.write_to_dataset(row)

        logger.debug("DataSet Preprocessed Successfully!")


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
