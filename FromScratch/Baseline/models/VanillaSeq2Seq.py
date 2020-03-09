"""
Vanilla Seq2Seq with BiDirection LSTM encoder and decoder
"""
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.root import LOGGING_FORMAT, LOGGING_LEVEL


logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


class Encoder(nn.Module):
    """
    A bidirectional LSTM Encoder
    Input:
        input_dim: Vocab length of input
        embedding_dim: Dimension of Embeddings
        hidden_dim: Dimension of Hidden vectors of LSTM
        n_layers: Layers of LSTM
        dropout: Dropout applied
    Returns:
        hidden: hidden layers of LSTM
        cell: cell state of LSTM
    """

    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src_len, batch_size]
        # src_len = [batch_size]

        embedded = self.dropout(self.embedding(src))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        return hidden, cell


class Decoder(nn.Module):
    """
    A Decoder LSTM Decoder
    Input:
        output_dim: Vocab length of the output
        embedding_dim: Decoder Embedding Dimension
        hidden_dim: Hidden Dimensions of the LSTM Layer
        n_layer: Number of layer for LSTM
        dropout: Dropout Applied
    Output:
        prediction: Output of the Fully connected layer
    """

    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.dropout(self.fc_out(output))
        prediction = output.squeeze(0)
        return prediction, hidden, cell


class VanillaSeq2Seq(nn.Module):
    """
    Final EncoderDecoderModel
    """

    def __init__(self, encoder, decoder, device):
        super(VanillaSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing=0.0):
        encoder_hidden, encoder_cell = self.encoder(src, src_len)

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Takse first letter of the input
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, encoder_hidden, encoder_cell)

            outputs[t] = output

            teacher_forcing = random.random() < teacher_forcing

            if teacher_forcing:
                input = trg[t]
            else:
                input = torch.argmax(output, dim=1)

        return outputs
