import logging
import os

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import tqdm
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator


from config.data import DATA_FOLDER, DATA_FOLDER_PROCESSED, DATASETS, SQUAD_NAME
from config.root import LOGGING_FORMAT, LOGGING_LEVEL, seed_all, device
from config.hyperparameters import VANILLA_SEQ2SEQ

from utils import word_tokenizer

# TODO: Move this to main menu to seed in the starting of application
seed_all()

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


FILE_PATH = os.path.join(DATA_FOLDER, DATA_FOLDER_PROCESSED)


def load_dataset(
    dataset_name="SQUAD",
    tokenizer=word_tokenizer,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
    use_glove=True,
    source_vocab=45000,
    target_vocab=28000,
    batch_size=VANILLA_SEQ2SEQ["BATCHSIZE"],
):
    """
    Method Loads the dataset from location and returns three iterators and SRC and TRG fields
    """
    logger.debug("Loading {} dataset".format(dataset_name))
    SRC = data.Field(
        tokenize=tokenizer,
        init_token=init_token,
        eos_token=eos_token,
        lower=True,
        include_lengths=True,
    )
    TRG = data.Field(
        tokenize=tokenizer, init_token=init_token, eos_token=eos_token, lower=True
    )

    location = os.path.join(FILE_PATH, dataset_name)

    logger.debug("Loading from location: {}".format(location))
    start_time = time.time()
    train_dataset, valid_dataset, test_dataset = TranslationDataset.splits(
        exts=(".paragraphs", ".questions"),
        fields=(SRC, TRG),
        path=location,
        train="train",
        validation="valid",
        test="test",
    )

    logger.debug(
        "Number of Samples: Training = {} | Validation = {} | Testing = {}".format(
            len(train_dataset.examples),
            len(valid_dataset.examples),
            len(test_dataset.examples),
        )
    )
    logger.debug("Time Taken: {:.6f}s".format(time.time() - start_time))
    logger.debug("Building Vocab")

    start_time = time.time()
    if use_glove:
        logger.debug("Using Glove vectors")
        SRC.build_vocab(train_dataset, max_size=source_vocab, vectors="glove.6B.300d")
        TRG.build_vocab(train_dataset, max_size=target_vocab, vectors="glove.6B.300d")
    else:
        SRC.build_vocab(train_dataset, max_size=source_vocab)
        TRG.build_vocab(train_dataset, max_size=target_vocab)

    logger.info(
        "Vocabulary Built! Source Tokens = {} | Target Tokens = {}  \nCreating Iterators".format(
            len(SRC.vocab), len(TRG.vocab)
        )
    )
    logger.debug("Time Taken: {:.6f}s".format(time.time() - start_time))

    return (
        BucketIterator.splits(
            (train_dataset, valid_dataset, test_dataset),
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device,
        ),
        SRC,
        TRG,
    )


if __name__ == "__main__":
    print(load_dataset())
