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
import spacy

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def generate_questons(sentence, src_field, trg_field, model, max_len):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load("en")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]


def generate_questions_vanilla_seq2Seq(
    max_len=50,
    dataset="SQUAD",
    model_location=os.path.join(TRAINED_MODEL_PATH, "{}.pt".format(models[1])),
):

    logger.debug("Loading Model")
    model, SRC, TRG, train_iterator, valid_iterator, test_iterator = initialize_vanillaSeq2Seq(
        dataset
    )
    logger.debug("Model Initialized")

    model = torch.load(model_location, map_location=device)

    logger.debug("Model Loaded")
    model.eval()

    outputs = []

    with torch.no_grad(), open("generated_questions.txt", "o") as file:

        for i, batch in tqdm(enumerate(test_iterator), total=len(test_iterator)):

            src, src_len = batch.src
            trg = batch.trg

            file.write(
                " ".join(generate_questons(src, SRC, TRG, model, max_len)) + "\n"
            )

    logger.debug("Generated questions into generated_questions.txt")


if __name__ == "__main__":
    generate_questions_vanilla_seq2Seq()
