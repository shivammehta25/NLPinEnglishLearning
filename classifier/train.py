
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from config.hyperparameters import (BIDIRECTION, DROPOUT, EMBEDDING_DIM,
                                    HIDDEN_DIM, N_LAYERS, LR, EPOCHS)
from config.root import (LOGGING_FORMAT, LOGGING_LEVEL,
                         TRAINED_CLASSIFIER_FOLDER,
                         TRAINED_CLASSIFIER_RNNHIDDEN, seed_all)
from datasetloader import GrammarDaset
from model import RNNHiddenClassifier
from utility import categorical_accuracy
from helperfunctions import train

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def count_parameters(model):
    """Method to count the number of parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_new_model(dataset, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTION, DROPOUT):
    """Method to initialise new model, takes in dataset object and hyperparameters as parameter"""
    logger.debug("Initializing Model")
    
    VOCAB_SIZE = len(dataset.question.vocab)
    OUTPUT_LAYERS = len(dataset.label.vocab)
    PAD_IDX = dataset.question.vocab.stoi[dataset.question.pad_token]

    model = RNNHiddenClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_LAYERS, N_LAYERS, BIDIRECTION, DROPOUT, PAD_IDX)

    logger.info("Model Initialized with {:,} trainiable parameters".format(count_parameters(model)))

    # Initialize pretrained word embeddings
    pretrained_embeddings = dataset.question.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Initialize Padding and Unknown as 0
    UNK_IDX = dataset.question.vocab.stoi[dataset.question.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    logger.debug("Copied PreTrained Embeddings")
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Utility to train the Model")

    parser.add_argument(
        "-s",
        "--seed",
        default=1234,
        help="Set custom seed for reproducibility",
    )

    parser.add_argument(
        "-m"
        "--model-location",
        default=None,
        help="Give an already trained model location to use and train more epochs on it"
    )

    parser.add_argument("-b", "--bidirectional", default=BIDIRECTION, help="Makes the model Bidirectional")
    parser.add_argument("-d", "--dropout", default=DROPOUT, help="Dropout count for the model")    
    parser.add_argument("-e", "--embedding-dim", default=EMBEDDING_DIM, help="Embedding Dimensions")
    parser.add_argument("-hd", "--hidden-dim", default=HIDDEN_DIM, help="Hidden dimensions of the RNN")
    parser.add_argument("-l", "--n-layers", default=N_LAYERS, help="Number of layers in RNN")
    parser.add_argument("-lr", "--learning-rate", default=LR, help="Learning rate of Adam Optimizer")
    parser.add_argument("-n", "--epochs", default=EPOCHS, help="Number of Epochs to train model")

    args = parser.parse_args()

    seed_all(args.seed)
    logger.debug("Custom seed set with: {}".format(args.seed))

    logger.info("Loading Dataset")

    dataset = GrammarDaset.get_iterators()

    logger.info("Dataset Loaded Successfully")

    if args.model_location:
        model = torch.load(args.model_location)
    else:
        model = initialize_new_model(dataset, args.embedding_dim, args.hidden_dim, args.n_layers, args.bidirectional, args.dropout)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, dataset.train_iterator, optimizer, criterion)



    



