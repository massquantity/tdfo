import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == False, -1e9)

    p_attn = nn.functional.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_model: int,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert dim_model % num_heads == 0
        self.d_k = dim_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList(
            [nn.Linear(dim_model, dim_model, device=device) for _ in range(3)]
        )
        self.output_linear = nn.Linear(dim_model, dim_model, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(dim=0)
        query, key, value = [
            linearLayer(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
            for linearLayer, x in zip(self.linear_layers, (query, key, value))
        ]

        # x shape: batch_size, num_heads, seq_len, d_k
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        return self.output_linear(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        d_ff = dim_model * 4
        self.w_1 = nn.Linear(dim_model, d_ff, device=device)
        self.w_2 = nn.Linear(d_ff, dim_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(
        self,
        size: int,
        dropout: float,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(size, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        # self.norm(x + self.dropout(sublayer(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.attention = MultiHeadedAttention(num_heads, embed_dim, dropout, device)
        self.feed_forward = FeedForward(embed_dim, dropout, device)
        self.input_sublayer = SublayerConnection(embed_dim, dropout, device)
        self.output_sublayer = SublayerConnection(embed_dim, dropout, device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class HistoryArch(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        history_len: int,
        embed_dim: int,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_len = history_len
        self.positional_encoding = nn.Parameter(
            torch.randn(history_len, embed_dim, device=device)
        )
        self.layernorm = nn.LayerNorm([history_len, embed_dim], device=device)
        self.dropout = nn.Dropout(p=dropout)

        item_embedding_config = EmbeddingConfig(
            name="item_embedding",
            embedding_dim=embed_dim,
            num_embeddings=vocab_size,
            feature_names=["item"],
            weight_init_max=1.0,
            weight_init_min=-1.0,
        )
        self.embed_collection = EmbeddingCollection(
            tables=[item_embedding_config],
            device=device,
        )

    def forward(self, id_list_features: KeyedJaggedTensor) -> torch.Tensor:
        jagged_tensor_dict = self.embed_collection(id_list_features)

        # (BT) * E -> B * T * E
        padded_embeddings = [
            torch.ops.fbgemm.jagged_2d_to_dense(
                values=jagged_tensor_dict[e].values(),
                offsets=jagged_tensor_dict[e].offsets(),
                max_sequence_length=self.history_len,
            ).view(-1, self.history_len, self.embed_dim)
            for e in id_list_features.keys()
        ]
        item_output = torch.cat(padded_embeddings, dim=1)
        batch_size = id_list_features.stride()
        x = item_output + self.positional_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        return self.dropout(self.layernorm(x))


class Bert4Rec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = embed_dim
        self.max_len = max_len
        self.pad_id = 0
        self.history = HistoryArch(
            vocab_size, max_len, embed_dim, dropout=dropout, device=device
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, dropout, device=device)
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(self.emb_dim, self.vocab_size, device=device)

    def forward(self, inputs: KeyedJaggedTensor) -> torch.Tensor:
        # B * T
        dense_tensor = inputs["item"].to_padded_dense(
            desired_length=self.max_len, padding_value=self.pad_id
        )
        # B * H * T * T
        mask = (
            (dense_tensor != self.pad_id)
            .unsqueeze(1)
            .repeat(1, dense_tensor.size(1), 1)
            .unsqueeze(1)
        )
        x = self.history(inputs)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return self.out(x)
