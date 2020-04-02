"""
Model Architecture
"""

import logging

import torch
import torch.nn as nn

from config.root import LOGGING_FORMAT, LOGGING_LEVEL

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


class RNNHiddenClassifier(nn.Module):
    """
    This classifier concatenates the last hidden 
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
        freeze_embeddings,
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        logger.debug(
            "Freeze Embeddings Value {}: {}".format(
                freeze_embeddings, self.embedding.weight.requires_grad
            )
        )

        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        if self.bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        if self.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            )
        else:
            hidden = self.dropout(hidden[-1, :, :])

        return self.fc(hidden)
