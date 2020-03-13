import logging
import os

import torch
from utils import nlp_word

from config.root import (
    LOGGING_FORMAT,
    LOGGING_LEVEL,
    TRAINED_MODEL_PATH,
    device,
    models,
    seed_all,
)
from train import initialize_vanillaSeq2Seq
from tqdm import tqdm
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def generate_questions_vanilla_seq2Seq(max_len=50, dataset="SQUAD"):

    logger.debug("Loading Model")
    model, TRG, train_iterator, valid_iterator, test_iterator = initialize_vanillaSeq2Seq(
        dataset
    )
    logger.debug("Model Initialized")

    model.load_state_dict(
        torch.load(
            os.path.join(TRAINED_MODEL_PATH, "{}.pt".format(models[1])),
            map_location=device,
        )
    )

    logger.debug("Model Loaded")
    model.eval()

    outputs = []

    with torch.no_grad():

        for i, batch in tqdm(enumerate(test_iterator), total=len(test_iterator)):

            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0)
            outputs.append(output.detach())

    with open("generated_questions.txt", "o") as file:
        for i, output in enumerate(outputs):
            sentence = [TRG.vocab.itos[i] for i in output]
            outputs[i] = " ".join(sentence)
            file.write(" ".join(sentence) + "\n")

    logger.debug("Generated questionsi nto generated_questions.txt")


if __name__ == "__main__":
    generate_questions_vanilla_seq2Seq()
