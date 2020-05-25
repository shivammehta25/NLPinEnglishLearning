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
    A bidirectional GRU Encoder
    Input:
        input_dim: Vocab length of input
        embedding_dim: Dimension of Embeddings
        hidden_dim: Dimension of Hidden vectors of LSTM
        dropout: Dropout applied
    Returns:
        output: Output of GRU
        hidden: Hidden state of GRU to act as initial input state of Decoder
    """

    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        packed_output, hidden = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )
        return output, hidden


class Attention(nn.Module):
    """
    Class for Attention Mechanism it is the softmax over the weight of
    which part of the sentences should be focused while generating next
    token.
    Input: 
        Hidden dim: Size of the hidden dimension of Encoder and Decoder
    Output:
        Attention over the source sentence
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.attn = nn.Linear(3 * (hidden_dim), hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    A Decoder GRU Decoder
    Input:
        output_dim: Vocab length of the output
        embedding_dim: Decoder Embedding Dimension
        hidden_dim: Hidden Dimensions of the GRU Layer
        dropout: Dropout Applied
        attn: Instance of the Attention Class
    Output:
        prediction: Output of the Fully connected layer
        hidden: Hidden state of this Decoder to work as
         an input for the next initial hidden state for decoder
    """

    def __init__(self, output_dim, embedding_dim, hidden_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention = attention
        if not isinstance(attention, Attention):
            raise TypeError(
                "The attention Parameter must be an instance of Attention class"
            )
        self.embedding = nn.Embedding(output_dim, embedding_dim)

        self.rnn = nn.GRU((2 * hidden_dim) + embedding_dim, hidden_dim)
        self.fc_out = nn.Linear(3 * hidden_dim + embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        attention = self.attention(hidden, encoder_outputs)

        attention = attention.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(attention, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0)


class VanillaSeq2Seq(nn.Module):
    """
    Final EncoderDecoderModel
    """

    def __init__(self, encoder, decoder, device):
        super(VanillaSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing=0.5):

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src, src_len)
        # Take first letter of the input
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            outputs[t] = output

            teacher_forcing = random.random() < teacher_forcing

            top1 = output.argmax(1)

            input = trg[t] if teacher_forcing else top1

        return outputs
