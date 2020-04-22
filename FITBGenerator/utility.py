"""
Utility methods to be used for training and dataloading purposes
"""

import time

import spacy
import torch

nlp = spacy.load("en")



def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def tokenizer(text):
    results = []
    text = text.replace("<blank>", " BLANKK ")
    text = text.replace("<slash>", " SLASHH ")
    text = text.replace("<Q>", " QSTARTIT ")
    text = text.replace("</Q>", " QENDIT ")
    text = text.replace("<K>", " KSTARTIT ")
    text = text.replace("</K>", " KENDIT ")
    text = text.replace("<A>", " ASTARTIT ")
    text = text.replace("</A>", " AENDIT ")
    for token in nlp(text):
        if token.text == "BLANKK":
            results.append("<blank>")
        elif token.text == "SLASHH":
            results.append("<slash>")
        elif token.text == "QSTARTIT":
            results.append("<Q>")
        elif token.text == "QENDIT":
            results.append("</Q>")
        elif token.text == "KSTARTIT":
            results.append("<K>")
        elif token.text == "KENDIT":
            results.append("</K>")
        elif token.text == "ASTARTIT":
            results.append("<A>")
        elif token.text == "AENDIT":
            results.append("</A>")
        else:
            results.append(token.text)

    return results


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])
