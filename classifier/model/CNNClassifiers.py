import torch
import torch.nn as nn
import torch.nn.functional as F
from config.root import device


class CNN2dClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_len):

        text = text.permute(1, 0)

        embedded = self.embedding(text)

        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)


class CNN1dClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs
                )
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_len):

        text = text.permute(1, 0)

        embedded = self.embedding(text)

        embedded = embedded.permute(0, 2, 1)

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)


class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):

        super().__init__()

        self.convlayer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, embedded):

        embedded = embedded.permute(0, 2, 1)

        post_conv = self.convlayer(embedded)

        return post_conv.permute(0, 2, 1)


class CNN1dExtraLayerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        linear_hidden_dim,
        output_dim,
        dropout,
        pad_idx,
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                CustomConv1d(
                    in_channels=embedding_dim,
                    out_channels=n_filters,
                    kernel_size=fs,
                    padding=((fs - 1) // 2),
                )
                for fs in filter_sizes
            ]
        )

        self.hidden_layer = nn.Linear(len(filter_sizes) * n_filters, linear_hidden_dim)

        self.fc = nn.Linear(linear_hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_len):

        text = text.permute(1, 0)
        max_len = text.shape[1]
        print(max_len)
        embedded = self.embedding(text)

        print("Embedding Size: {}".format(embedded.shape))

        conved = [conv(embedded) for conv in self.convs]

        print([conv.shape for conv in conved])

        cnns = F.relu(torch.cat([conv for conv in conved], -1))
        print("Concat CNN shape: {}".format(cnns.shape))

        hidden_output = self.hidden_layer(cnns)

        print(hidden_output.shape)
        mask = (
            torch.arange(max_len, device=device).expand(len(text_len), max_len)
            < text_len.unsqueeze(1)
        ).float()

        print(mask.shape)
        print(mask)

        # mask = (torch.arange(max_len) < text_len).float().cuda()

        vec, _ = torch.max(hidden_output - (1.0 - mask) * 1e23, dim=1)

        print("Vector shapes : {}".format(vec.shape))

        exit(0)

        # print([conv.shape for conv in conved])

        # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # for pool in pooled:
        #     print(pool.shape)

        cat = self.dropout(vec)

        return self.fc(cat)
