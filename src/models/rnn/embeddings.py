import torch
import torch.nn as nn


class TextEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len)
        return: (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(input_ids)
