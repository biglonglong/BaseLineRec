import math
import torch
from torch import nn
import torch.nn.functional as F
from models.baserec import PositionalEncoding, BaseSequentialRecModel


class MultiHeadAttentionwithPreLNwithResidual(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.atten_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_length, d_model] -> [batch_size, num_heads, seq_length, d_head]
        # mask: [batch_size, num_heads, seq_length, seq_length]

        batch_size, seq_length, _ = x.size()
        residual = x

        # Pre-LN
        x_norm = self.norm(x)

        # Linear projections
        query = (
            self.query_linear(x_norm)
            .view(batch_size, seq_length, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        key = (
            self.key_linear(x_norm)
            .view(batch_size, seq_length, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        value = (
            self.value_linear(x_norm)
            .view(batch_size, seq_length, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Ateention Matrix
        if mask is not None:
            # Use a large negative value instead of -inf. Setting -inf can
            # produce NaNs from softmax when an entire row is masked.
            scores = scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        dropout_attn_weights = self.atten_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(dropout_attn_weights, value)

        # Concatenate heads and pass through final linear layer
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.d_model)
        )
        output = self.out_dropout(self.out_linear(attn_output))

        return output + residual, attn_weights


class PositionwiseFeedForwardwithPreLNwithResidual(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch_size, seq_length, d_model]

        residual = x
        x_norm = self.norm(x)
        out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))

        return out + residual


class SASRecBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttentionwithPreLNwithResidual(
            d_model=d_model, num_heads=n_heads, dropout_rate=dropout
        )
        self.ffn = PositionwiseFeedForwardwithPreLNwithResidual(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout
        )

    def forward(self, x, mask=None):
        out, attn = self.attention(x, mask)
        out = self.ffn(out)
        return out, attn


class SASRec(BaseSequentialRecModel):
    def __init__(
        self,
        num_items: int,
        d_model: int = 128,
        max_len: int = 10,
        dropout: float = 0.1,
        pad_idx: int = 0,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
    ):
        super().__init__(num_items, d_model, max_len, dropout, pad_idx)
        self.pad_idx = pad_idx

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [SASRecBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Layer norm for final output
        self.final_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def make_causal_mask_with_padding(self, seq):
        # seq: [batch_size, seq_len]
        batch_size, seq_len = seq.size()

        # Causal mask (upper triangular): [1, 1, seq_len, seq_len]
        causal_mask = (
            torch.triu(
                torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Padding mask: [batch_size, 1, 1, seq_len]
        pad_mask = (seq == self.pad_idx).unsqueeze(1).unsqueeze(2)

        return causal_mask | pad_mask

    def _encode(self, seq):
        # Embedding
        seq_embs = self.item_embedding(seq) * math.sqrt(
            self.d_model
        )  # avoiding interference with position embeddings
        seq_embs = self.pos_encoding(seq_embs)
        seq_embs = self.embed_dropout(seq_embs)

        # Attention
        mask = self.make_causal_mask_with_padding(seq)
        x = seq_embs
        # attentions = []
        for block in self.blocks:
            x, _ = block(x, mask)
            # attentions.append(attn)

        return self.final_norm(x)  # , attentions
