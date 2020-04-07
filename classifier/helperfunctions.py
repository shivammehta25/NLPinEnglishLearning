"""
Helper Functions containing training and evaluation methods
"""

import torch
from tqdm.auto import tqdm
from utility import categorical_accuracy


def train(model, iterator, optimizer, criterion, dataset_tag):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator, total=len(iterator)):

        optimizer.zero_grad()

        text, text_lengths = get_batch_data(batch, dataset_tag)

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_batch_data(batch, dataset_tag):

    if dataset_tag == "multi":
        (question, question_len), (key, key_len), (answer, answer_len) = (
            batch.question,
            batch.key,
            batch.answer,
        )

        text = torch.cat((question, key, answer), dim=0)
        text_lengths = question_len + key_len + answer_len
    else:

        text, text_lengths = batch.text

    return text, text_lengths


def evaluate(model, iterator, criterion, dataset_tag):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in tqdm(iterator, total=len(iterator)):

            text, text_lengths = get_batch_data(batch, dataset_tag)

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_batch_data_with_field_embeddings(batch):
    (question, question_len), (key, key_len), (answer, answer_len) = (
        batch.question,
        batch.key,
        batch.answer,
    )
    print(
        question.shape,
        question_len.shape,
        key.shape,
        key_len.shape,
        answer.shape,
        answer_len.shape,
    )
    print("Batch shape: {}".format(batch.shape))

    text = torch.cat((question, key, answer), dim=0)
    text_lengths = question_len + key_len + answer_len


def train_field_embeddings(model, iterator, optimizer, criterion, dataset_tag):

    epoch_loss = 0
    epoch_acc = 0

    # field_label = data.Field(tokenize=tokenizer)
    # field_label.build_vocab(["Q", "K", "A"])

    model.train()

    for batch in tqdm(iterator, total=len(iterator)):

        optimizer.zero_grad()

        text, text_lengths = get_batch_data(batch, dataset_tag)

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
