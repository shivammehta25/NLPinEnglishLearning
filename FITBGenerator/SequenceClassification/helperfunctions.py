"""
Helper Functions containing training and evaluation methods
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from utility import categorical_accuracy, binary_accuracy, f1_measure
from config.root import device


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()

    for batch in tqdm(iterator, total=len(iterator)):

        optimizer.zero_grad()

        text, text_lengths = batch.answer
        max_len = text.shape[1]

        predictions = model(text, text_lengths)

        mask, key = get_mask_key_from_batch(batch, text, max_len, text_lengths)

        loss = criterion(predictions, key, mask)

        acc = binary_accuracy(predictions, key, mask)
        f1_score = f1_measure(predictions, key, mask)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1_score

    return (
        epoch_loss / len(iterator),
        epoch_acc / len(iterator),
        epoch_f1 / len(iterator),
    )


def get_mask_key_from_batch(batch, text, max_len, text_lengths):
    key, _ = batch.key

    key = (
        torch.from_numpy(np.where(np.isin(text.cpu().numpy(), key.cpu().numpy()), 1, 0))
        .float()
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
    return mask, key


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.eval()

    with torch.no_grad():

        for batch in tqdm(iterator, total=len(iterator)):

            text, text_lengths = batch.answer
            max_len = text.shape[1]
            predictions = model(text, text_lengths)

            mask, key = get_mask_key_from_batch(batch, text, max_len, text_lengths)

            loss = criterion(predictions, key, mask)

            acc = binary_accuracy(predictions, key, mask)
            f1_score = f1_measure(predictions, key, mask)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1_score

    return (
        epoch_loss / len(iterator),
        epoch_acc / len(iterator),
        epoch_f1 / len(iterator),
    )
