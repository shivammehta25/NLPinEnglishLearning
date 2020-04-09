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
from datasetloader import GrammarDasetMultiTag, GrammarDasetSingleTag
from helperfunctions import evaluate, train, train_tag_model, evaluate_tag_model
from model import (
    RNNHiddenClassifier,
    RNNMaxpoolClassifier,
    CNN2dClassifier,
    CNN1dClassifier,
    RNNFieldClassifer,
    CNN1dExtraLayerClassifier,
)
from utility import categorical_accuracy, epoch_time

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
    dataset_tag,
    linear_hidden_dim,
):
    """Method to initialise new model, takes in dataset object and hyperparameters as parameter"""
    logger.debug("Initializing Model")
    if dataset_tag == "multi":
        VOCAB_SIZE = len(dataset.question.vocab)
        PAD_IDX = dataset.question.vocab.stoi[dataset.question.pad_token]
        pretrained_embeddings = dataset.question.vocab.vectors
        UNK_IDX = dataset.question.vocab.stoi[dataset.question.unk_token]
    else:
        VOCAB_SIZE = len(dataset.text.vocab)
        PAD_IDX = dataset.text.vocab.stoi[dataset.text.pad_token]
        pretrained_embeddings = dataset.text.vocab.vectors
        UNK_IDX = dataset.text.vocab.stoi[dataset.text.unk_token]

    OUTPUT_LAYERS = len(dataset.label.vocab)

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

    elif classifier_type == "RNNMaxpoolClassifier":
        model = RNNMaxpoolClassifier(
            VOCAB_SIZE,
            embedding_dim,
            hidden_dim,
            OUTPUT_LAYERS,
            n_layers,
            bidirectional,
            dropout,
            PAD_IDX,
        )
    elif classifier_type == "CNN2dClassifier":
        model = CNN2dClassifier(
            VOCAB_SIZE,
            embedding_dim,
            CNN_N_FILTER,
            CNN_FILTER_SIZES,
            OUTPUT_LAYERS,
            dropout,
            PAD_IDX,
        )
    elif classifier_type == "CNN1dClassifier":
        model = CNN1dClassifier(
            VOCAB_SIZE,
            embedding_dim,
            CNN_N_FILTER,
            CNN_FILTER_SIZES,
            OUTPUT_LAYERS,
            dropout,
            PAD_IDX,
        )
    elif classifier_type == "RNNFieldClassifer":
        model = RNNFieldClassifer(
            VOCAB_SIZE,
            embedding_dim,
            hidden_dim,
            OUTPUT_LAYERS,
            n_layers,
            bidirectional,
            dropout,
            PAD_IDX,
            dataset.tags,
        )
    elif classifier_type == "CNN1dExtraLayerClassifier":
        model = CNN1dExtraLayerClassifier(
            VOCAB_SIZE,
            embedding_dim,
            CNN_N_FILTER,
            CNN_FILTER_SIZES,
            OUTPUT_LAYERS,
            dropout,
            PAD_IDX,
            linear_hidden_dim,
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
        "-t",
        "--tag",
        default="multi",
        choices=["multi", "single"],
        help="Use two different dataset type, multi type and single type where all are merged into same key ",
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
        choices=[
            "RNNHiddenClassifier",
            "RNNMaxpoolClassifier",
            "RNNFieldClassifier",
            "CNN2dClassifier",
            "CNN1dClassifier",
            "RNNFieldClassifer",
            "CNN1dExtraLayerClassifier",
        ],
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

    if args.tag == "multi":
        dataset = GrammarDasetMultiTag.get_iterators(args.batch_size)
    else:
        dataset = GrammarDasetSingleTag.get_iterators(args.batch_size)

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
            args.tag,
            args.linear_hidden_dim,
        )

    criterion = nn.CrossEntropyLoss()
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
        if args.model == "RNNFieldClassifer":
            train_loss, train_acc = train_tag_model(
                model, dataset.train_iterator, optimizer, criterion, dataset.tags
            )
            test_loss, test_acc = evaluate_tag_model(
                model, dataset.test_iterator, criterion, dataset.tags
            )

        else:
            train_loss, train_acc = train(
                model, dataset.train_iterator, optimizer, criterion, args.tag
            )
            test_loss, test_acc = evaluate(
                model, dataset.test_iterator, criterion, args.tag
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
