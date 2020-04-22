"""
Model Architectures of RNN based Classifiers
"""

import logging

import torch
import torch.nn as nn

from config.root import LOGGING_FORMAT, LOGGING_LEVEL

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


class LSTMWithPackPaddedSequences(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, bidirectional, dropout):

        super().__init__()

        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, embedded, text_lengths):

        embedded = embedded.permute(1, 0, 2)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, _) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        output = output.permute(1, 0, 2)

        return output, output_lengths, hidden


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
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.bidirectional = bidirectional

        self.rnn = LSTMWithPackPaddedSequences(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        if bidirectional:
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))

        output, output_lengths, hidden = self.rnn(embedded, text_lengths)

        output = self.fc(output)

        return output
