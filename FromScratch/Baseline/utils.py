"""
All Utilities methods and classes and objects are available here
"""
from spacy.lang.en import English


nlp_word = English()
nlp_sentence = English()
nlp_sentence.add_pipe(nlp_sentence.create_pipe("sentencizer"))


def word_tokenizer(sentence):
    return [word.text for word in nlp_word(sentence)]

