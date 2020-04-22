"""
Helper Functions containing training and evaluation methods
"""

import torch
import numpy as np
from tqdm.auto import tqdm
from utility import categorical_accuracy, binary_accuracy
from config.root import device


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator, total=len(iterator)):

        optimizer.zero_grad()

        text, text_lengths = batch.answer
        max_len = text.shape[1]

        print("Text Input: {}".format(text.shape))

        predictions = model(text, text_lengths)

        print("Predictions: {}".format(predictions.shape))

        key, _ = batch.key

        key = (
            torch.from_numpy(
                np.where(np.isin(text.cpu().numpy(), key.cpu().numpy()), 1, 0)
            )
            .to(device)
            .unsqueeze(2)
        )

        mask = (
            (
                torch.arange(max_len, device=device).expand(len(text_lengths), max_len)
                < text_lengths.unsqueeze(1)
            )
            .float()
            .unsqueeze(2)
        )

        print(f"Mask: {mask.shape}")

        print(f"key: {key.shape}")

        loss = criterion(predictions, key, weight=mask)

        acc = binary_accuracy(predictions, predictions)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in tqdm(iterator, total=len(iterator)):

            text, text_lengths = batch.answer

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
