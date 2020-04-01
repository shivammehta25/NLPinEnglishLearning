import spacy

nlp = spacy.load("en")


def tokenizer(text):
    results = []
    text = text.replace("<blank>", " BLANK ")
    text = text.replace("<slash>", " SLASHH ")
    for token in nlp(text):
        if token.text == "BLANK":
            results.append("<blank>")
        elif token.text == "SLASHH":
            results.append("<slash>")
        else:
            results.append(token.text)

    return results
