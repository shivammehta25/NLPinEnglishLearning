"""
Training script for the model
"""
import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from config.hyperparameters import (
    BATCH_SIZE,
    BIDIRECTION,
    DROPOUT,
    EMBEDDING_DIM,
    EPOCHS,
    FREEZE_EMBEDDINGS,
    HIDDEN_DIM,
    LR,
    N_LAYERS,
    WEIGHT_DECAY,
    CNN_N_FILTER,
    CNN_FILTER_SIZES,
    LINEAR_HIDDEN_DIM,
)
from config.root import (
    LOGGING_FORMAT,
    LOGGING_LEVEL,
    TRAINED_CLASSIFIER_FOLDER,
    TRAINED_CLASSIFIER_RNNHIDDEN,
    device,
    seed_all,
    SEED,
)
from datasetloader import GrammarDasetAnswerKey
from helperfunctions import evaluate, train
from model import RNNHiddenClassifier
from utility import categorical_accuracy, epoch_time
from lossfunction import BCEWithLogitLossWithMask

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def count_parameters(model):
    """Method to count the number of parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_new_model(
    classifier_type,
    dataset,
    embedding_dim,
    hidden_dim,
    n_layers,
    bidirectional,
    dropout,
    freeze_embeddings,
    linear_hidden_dim,
):
    """Method to initialise new model, takes in dataset object and hyperparameters as parameter"""
    logger.debug("Initializing Model")
    VOCAB_SIZE = len(dataset.answer.vocab)
    PAD_IDX = dataset.answer.vocab.stoi[dataset.answer.pad_token]
    pretrained_embeddings = dataset.answer.vocab.vectors
    UNK_IDX = dataset.answer.vocab.stoi[dataset.answer.unk_token]
    OUTPUT_LAYERS = 1
    if classifier_type == "RNNHiddenClassifier":

        model = RNNHiddenClassifier(
            VOCAB_SIZE,
            embedding_dim,
            hidden_dim,
            OUTPUT_LAYERS,
            n_layers,
            bidirectional,
            dropout,
            PAD_IDX,
        )
    else:
        raise TypeError("Invalid Classifier selected")

    if freeze_embeddings:
        model.embedding.weight.requires_grad = False

    logger.debug(
        "Freeze Embeddings Value {}: {}".format(
            freeze_embeddings, model.embedding.weight.requires_grad
        )
    )

    logger.info(
        "Model Initialized with {:,} trainiable parameters".format(
            count_parameters(model)
        )
    )

    # Initialize pretrained word embeddings

    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Initialize Padding and Unknown as 0
    model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)

    logger.debug("Copied PreTrained Embeddings")
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Utility to train the Model")

    parser.add_argument(
        "-s",
        "--seed",
        default=SEED,
        help="Set custom seed for reproducibility",
        type=int,
    )

    parser.add_argument(
        "-loc",
        "--model-location",
        default=None,
        help="Give an already trained model location to use and train more epochs on it",
    )

    parser.add_argument(
        "-b",
        "--bidirectional",
        default=BIDIRECTION,
        help="Makes the model Bidirectional",
        type=bool,
    )
    parser.add_argument(
        "-d",
        "--dropout",
        default=DROPOUT,
        help="Dropout count for the model",
        type=float,
    )
    parser.add_argument(
        "-e",
        "--embedding-dim",
        default=EMBEDDING_DIM,
        help="Embedding Dimensions",
        type=int,
    )
    parser.add_argument(
        "-hd",
        "--hidden-dim",
        default=HIDDEN_DIM,
        help="Hidden dimensions of the RNN",
        type=int,
    )
    parser.add_argument(
        "-l", "--n-layers", default=N_LAYERS, help="Number of layers in RNN", type=int
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=LR,
        help="Learning rate of Adam Optimizer",
        type=float,
    )
    parser.add_argument(
        "-n",
        "--epochs",
        default=EPOCHS,
        help="Number of Epochs to train model",
        type=int,
    )
    parser.add_argument(
        "-batch",
        "--batch_size",
        default=BATCH_SIZE,
        help="Number of Epochs to train model",
        type=int,
    )

    parser.add_argument(
        "-f",
        "--freeze-embeddings",
        default=FREEZE_EMBEDDINGS,
        help="Freeze Embeddings of Model",
        type=int,
    )

    parser.add_argument(
        "-l2",
        "--l2-regularization",
        default=WEIGHT_DECAY,
        help="Value of alpha in l2 regularization 0 means no regularization ",
        type=float,
    )

    parser.add_argument(
        "-m",
        "--model",
        default="RNNHiddenClassifier",
        choices=["RNNHiddenClassifier"],
        help="select the classifier to train on",
    )

    parser.add_argument(
        "-lhd",
        "--linear-hidden-dim",
        default=LINEAR_HIDDEN_DIM,
        help="Freeze Embeddings of Model",
        type=int,
    )

    args = parser.parse_args()

    seed_all(args.seed)
    logger.debug(args)
    logger.debug("Custom seed set with: {}".format(args.seed))

    logger.info("Loading Dataset")

    dataset = GrammarDasetAnswerKey.get_iterators(args.batch_size)

    logger.info("Dataset Loaded Successfully")

    if args.model_location:
        model = torch.load(args.model_location)
    else:
        model = initialize_new_model(
            args.model,
            dataset,
            args.embedding_dim,
            args.hidden_dim,
            args.n_layers,
            args.bidirectional,
            args.dropout,
            args.freeze_embeddings,
            args.linear_hidden_dim,
        )

    criterion = BCEWithLogitLossWithMask()
    optimizer = optim.Adam(
        model.parameters(), lr=LR, weight_decay=args.l2_regularization
    )

    model = model.to(device)
    criterion = criterion.to(device)

    logger.info(model)

    if not os.path.exists(TRAINED_CLASSIFIER_FOLDER):
        os.mkdir(TRAINED_CLASSIFIER_FOLDER)

    best_test_loss = float("inf")

    for epoch in range(int(args.epochs)):

        start_time = time.time()
        train_loss, train_acc, train_f1, train_precision, train_recall = train(
            model, dataset.train_iterator, optimizer, criterion
        )
        test_loss, test_acc, test_f1, test_precision, test_recall = evaluate(
            model, dataset.test_iterator, criterion
        )

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(
                model,
                os.path.join(TRAINED_CLASSIFIER_FOLDER, TRAINED_CLASSIFIER_RNNHIDDEN),
            )

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%")
        print(f"\t Train. F1: {train_f1:.2f} |  Val. F1: {test_f1:.2f}")
        print(
            f"\t Train. Precision: {train_precision:.2f} |  Val. Precision: {test_precision:.2f}"
        )
        print(f"\t Train. Recall: {train_recall:.2f} |  Val. Recall: {test_recall:.2f}")
