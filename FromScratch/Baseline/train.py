"""
Training Code
"""
import argparse
import logging


from tqdm import tqdm
import torch
import torch.optim as optim
from config.root import LOGGING_FORMAT, LOGGING_LEVEL, models, seed_all, device
from dataloader import load_dataset
from models.VanillaSeq2Seq import VanillaSeq2Seq
from config.hyperparameters import VANILLA_SEQ2SEQ

from models.VanillaSeq2Seq import *

seed_all()


logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def train(model, iterator, optimizer, criterion, clip):
    """
    Generic Training Method
    """

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, src_len, trg)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

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


def train_vanilla_seq2seq(
    dataset_name="SQUAD",
    clip=1,
    lr=0.001,
    validation=True,
    epochs=10,
    teacher_forcing=0.0,
):
    """
    Method to train the Vanilla Seq2Seq
    """

    logger.debug("Data Loading")

    (train_iterator, valid_iterator, test_iterator), SRC, TRG = load_dataset(
        dataset_name,
        source_vocab=VANILLA_SEQ2SEQ["INPUT_DIM"],
        target_vocab=VANILLA_SEQ2SEQ["OUTPUT_DIM"],
    )

    logger.debug("Initializing Models on {}".format(device))
    enc = Encoder(
        VANILLA_SEQ2SEQ["INPUT_DIM"],
        VANILLA_SEQ2SEQ["ENC_EMB_DIM"],
        VANILLA_SEQ2SEQ["HID_DIM"],
        VANILLA_SEQ2SEQ["N_LAYERS"],
        VANILLA_SEQ2SEQ["DROPOUT"],
    )
    dec = Decoder(
        VANILLA_SEQ2SEQ["OUTPUT_DIM"],
        VANILLA_SEQ2SEQ["DEC_EMB_DIM"],
        VANILLA_SEQ2SEQ["HID_DIM"],
        VANILLA_SEQ2SEQ["N_LAYERS"],
        VANILLA_SEQ2SEQ["DROPOUT"],
    )

    model = VanillaSeq2Seq(enc, dec, device).to(device)

    logger.info(
        "The model has {:,} trainable parameters".format(count_parameters(model))
    )

    logger.debug(model)

    optimizer = optim.Adam(model.parameters())

    TRG_PADDING = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PADDING)

    best_valid_loss = float("inf")

    for epoch in tqdm(range(epochs)):
        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "trained_model.pt")

    print(f"Epoch: {epoch+1:02}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Utility to Train datasets {}".format(models)
    # )
    # parser.add_argument(
    #     "-d", "--dataset", default="SQUAD", help="which dataset to train on"
    # )
    # parser.add_argument(
    #     "-m", "--model", default=1, help="Which Model to Train", type=int
    # )
    # parser.add_argument(
    #     "-c", "--clipnorm", default=1, help="Value to clip gradients", type=int
    # )
    # parser.add_argument(
    #     "-l",
    #     "--learningrate",
    #     help="Learning rate of Adam Optmizer",
    #     type=float,
    #     default=0.001,
    # )
    # parser.add_argument(
    #     "-v",
    #     "--validation",
    #     help="Flag to turn validation on and off",
    #     default=True,
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "-e", "--epochs", default=5, help="Number of Epochs to train", type=int
    # )
    # parser.add_argument(
    #     "-t", "--teacherforcing", default=0.5, help="Teacher Forcing", type=int
    # )

    # parser.add_argument()

    # args = parser.parse_args()

    # if args.model == 1:
    #     train_vanilla_seq2seq(
    #         args.dataset,
    #         args.clipnorm,
    #         args.learningrate,
    #         args.validation,
    #         args.epochs,
    #         args.teacherforcing,
    #     )

    train_vanilla_seq2seq()
