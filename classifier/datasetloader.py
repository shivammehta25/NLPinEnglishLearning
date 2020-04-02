"""
Load Dataset
"""

import logging
import os

import torch
from torchtext import data, datasets
from config.data import DATASET_FOLDER, PROCESSED_DATASET, PROCESSED_DATASET_FOLDER
from config.hyperparameters import MAX_VOCAB, BATCH_SIZE
from config.root import LOGGING_FORMAT, LOGGING_LEVEL, device
from utility import tokenizer

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


class GrammarDaset:
    def __init__(self):

        self.dataset_location = PROCESSED_DATASET

        self.question = data.Field(
            tokenize=tokenizer, include_lengths=True, eos_token="</q>", init_token="<q>"
        )
        self.key = data.Field(
            tokenize=tokenizer, include_lengths=True, eos_token="</k>", init_token="<k>"
        )
        self.answer = data.Field(
            tokenize=tokenizer, include_lengths=True, eos_token="</a>", init_token="<a>"
        )
        self.label = data.LabelField()

        self.fields = None
        self.trainset = None
        self.testset = None
        self.train_iterator, self.test_iterator = None, None

    @classmethod
    def get_iterators(cls, batch_size):
        """
        Load dataset and return iterators
        """
        grammar_dataset = cls()

        grammar_dataset.fields = [
            ("question", grammar_dataset.question),
            ("key", grammar_dataset.key),
            ("answer", grammar_dataset.answer),
            ("label", grammar_dataset.label),
            (None, None),
        ]

        if not os.path.exists(PROCESSED_DATASET["train"]) or not os.path.exists(
            PROCESSED_DATASET["test"]
        ):
            raise FileNotFoundError(
                "Please run the preprocessdata.py first by executing python preprocessdata.py"
            )

        grammar_dataset.trainset, grammar_dataset.testset = data.TabularDataset.splits(
            path=os.path.join(DATASET_FOLDER, PROCESSED_DATASET_FOLDER),
            train="train.tsv",
            test="test.tsv",
            format="tsv",
            fields=grammar_dataset.fields,
            skip_header=True,
        )

        logger.debug("Data Loaded Successfully!")

        grammar_dataset.question.build_vocab(
            grammar_dataset.trainset,
            max_size=MAX_VOCAB,
            vectors="glove.6B.300d",
            unk_init=torch.Tensor.normal_,
        )

        grammar_dataset.key.vocab = grammar_dataset.question.vocab
        grammar_dataset.answer.vocab = grammar_dataset.question.vocab

        grammar_dataset.label.build_vocab(grammar_dataset.trainset)

        logger.debug("Vocabulary Loaded")
        grammar_dataset.train_iterator, grammar_dataset.test_iterator = data.BucketIterator.splits(
            (grammar_dataset.trainset, grammar_dataset.testset),
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.question) + len(x.key) + len(x.answer),
            device=device,
        )
        logger.debug("Created Iterators")

        return grammar_dataset
