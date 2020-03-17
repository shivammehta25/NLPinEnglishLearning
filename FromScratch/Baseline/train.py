"""
Training Code
"""
import argparse
import logging
import time

import os
import math
import torch
import torch.optim as optim
from tqdm import tqdm

from config.hyperparameters import VANILLA_SEQ2SEQ
from config.root import (
    LOGGING_FORMAT,
    LOGGING_LEVEL,
    device,
    models,
    seed_all,
    TRAINED_MODEL_PATH,
)
from dataloader import load_dataset
from models.VanillaSeq2Seq import *

seed_all()


logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip, teacher_forcing=0.5):
    """
    Generic Training Method
    """

    model.train()

    epoch_loss = 0

    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):

        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, src_len, trg, teacher_forcing)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.detach().item()

        # Throwing GPU out of memory on Colab
        del output
        del loss

    # Force emptying the GPU caches reduces runtime but effective to
    # save space on GPU at Colab
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """
    Generic Evaluation Method
    """

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def initialize_vanillaSeq2Seq(dataset_name):
    logger.debug("Initializing Datasets...")
    (train_iterator, valid_iterator, test_iterator), SRC, TRG = load_dataset(
        dataset_name,
        source_vocab=VANILLA_SEQ2SEQ["INPUT_DIM"],
        target_vocab=VANILLA_SEQ2SEQ["OUTPUT_DIM"],
        batch_size=VANILLA_SEQ2SEQ["BATCHSIZE"],
    )

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    logger.debug("Initializing Models on {}".format(device))
    enc = Encoder(
        INPUT_DIM,
        VANILLA_SEQ2SEQ["ENC_EMB_DIM"],
        VANILLA_SEQ2SEQ["HID_DIM"],
        VANILLA_SEQ2SEQ["DROPOUT"],
    )
    attn = Attention(VANILLA_SEQ2SEQ["HID_DIM"])
    dec = Decoder(
        OUTPUT_DIM,
        VANILLA_SEQ2SEQ["DEC_EMB_DIM"],
        VANILLA_SEQ2SEQ["HID_DIM"],
        VANILLA_SEQ2SEQ["DROPOUT"],
        attn,
    )
    model = VanillaSeq2Seq(enc, dec, device).to(device)
    return model, SRC, TRG, train_iterator, valid_iterator, test_iterator


def train_vanilla_seq2seq(
    dataset_name, clip, lr, validation, epochs, train_model_path, teacher_forcing
):
    """
    Method to train the Vanilla Seq2Seq
    """

    logger.debug("Data Loading")

    model, SRC, TRG, train_iterator, valid_iterator, _ = initialize_vanillaSeq2Seq(
        dataset_name
    )

    if train_model_path:
        logger.debug("Loading Pretrained model")
        model = torch.load(train_model_path)
        model = model.to(device)
    else:
        model.apply(init_weights)

    logger.info(
        "The model has {:,} trainable parameters".format(count_parameters(model))
    )

    logger.debug(model)

    optimizer = optim.Adam(model.parameters())

    TRG_PADDING = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PADDING)

    best_valid_loss = float("inf")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                model, os.path.join(TRAINED_MODEL_PATH, "{}.pt".format(models[1]))
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Epoch: {:02} | Time: {}m {}s".format(epoch + 1, epoch_mins, epoch_secs)
        )
        logger.info(
            "\tTrain Loss: {:.3f} | Train PPL: {:7.3f}".format(
                train_loss, math.exp(train_loss)
            )
        )
        logger.info(
            "\t Val. Loss: {:.3f} |  Val. PPL: {:7.3f}".format(
                valid_loss, math.exp(valid_loss)
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to Train datasets {}".format(models)
    )
    parser.add_argument(
        "-d", "--dataset", default="SQUAD", help="which dataset to train on"
    )
    parser.add_argument(
        "-m", "--model", default=1, help="Which Model to Train", type=int
    )
    parser.add_argument(
        "-c", "--clipnorm", default=1, help="Value to clip gradients", type=int
    )
    parser.add_argument(
        "-l",
        "--learningrate",
        help="Learning rate of Adam Optmizer",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "-v",
        "--validation",
        help="Flag to turn validation on and off",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "-e", "--epochs", default=5, help="Number of Epochs to train", type=int
    )
    parser.add_argument(
        "-t", "--teacherforcing", default=0.5, help="Teacher Forcing", type=int
    )
    parser.add_argument(
        "-tmp",
        "--trained-model-path",
        default="",
        help="Load the model from the directory",
    )

    args = parser.parse_args()

    if args.model == 1:
        train_vanilla_seq2seq(
            args.dataset,
            args.clipnorm,
            args.learningrate,
            args.validation,
            args.epochs,
            args.trained_model_path,
            args.teacherforcing,
        )
