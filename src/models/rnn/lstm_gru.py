import torch
import torch.nn as nn

from src.models.rnn.embeddings import TextEmbedding


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.embedding = TextEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_dim, num_classes),
        )

    def forward(self, input_ids, lengths):
        embeddings = self.embedding(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return self.classifier(hidden)
