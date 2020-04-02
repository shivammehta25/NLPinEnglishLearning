import spacy
import torch

nlp = spacy.load("en")


def tokenizer(text):
    results = []
    text = text.replace("<blank>", " BLANKK ")
    text = text.replace("<slash>", " SLASHH ")
    for token in nlp(text):
        if token.text == "BLANKK":
            results.append("<blank>")
        elif token.text == "SLASHH":
            results.append("<slash>")
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
