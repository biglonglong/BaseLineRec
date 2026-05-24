import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)

    def forward(self, x):
        # x: [batch_size, seq_length, embedding_dim]
        # positions: [1, seq_length]
        # pos_embeddings: [1, seq_length, embedding_dim]

        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)


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
            scores = scores.masked_fill(mask, float("-inf"))
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


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int = 128,
        max_len: int = 15,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_layers: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx

        # Item embedding
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=pad_idx)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_len, d_model)

        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [SASRecBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Layer norm for final output
        self.final_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Initialize position embeddings specially
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_encoding.pos_embedding.weight, std=0.02)

        with torch.no_grad():
            self.item_embedding.weight[self.pad_idx].zero_()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
            x, attn = block(x, mask)
            # attentions.append(attn)

        return self.final_norm(x)  # , attentions

    def forward(self, seq):
        return self._encode(seq)

    def calculate_loss(self, seq, target_ids, neg_target_ids=None):
        # seq: [batch_size, seq_len]
        # target_ids: [batch_size, seq_len]
        # neg_target_ids: [batch_size, seq_len, num_negatives]

        hidden = self._encode(seq)

        valid_mask = (target_ids != self.pad_idx).float()
        num_valid = valid_mask.sum()
        if num_valid == 0:
            return (hidden * 0).sum()

        if neg_target_ids is not None:
            # Sampled BCE Loss for large dataset
            pos_emb = self.item_embedding(target_ids)  # [B, L, D]
            pos_logits = (hidden * pos_emb).sum(dim=-1)  # [B, L]
            pos_loss = -F.logsigmoid(pos_logits)  # [B, L]

            neg_valid_mask = (neg_target_ids != self.pad_idx).float()
            neg_emb = self.item_embedding(neg_target_ids)  # [B, L, K, D]
            neg_logits = torch.einsum("bld,blkd->blk", hidden, neg_emb)  # [B, L, K]
            neg_loss_per_sample = -F.logsigmoid(-neg_logits)
            neg_loss = (neg_loss_per_sample * neg_valid_mask).sum(
                dim=-1
            ) / neg_valid_mask.sum(dim=-1).clamp(
                min=1e-8
            )  # [B, L]

            loss_per_step = pos_loss + neg_loss

        else:
            # Full CE Loss for small dataset
            logits = torch.matmul(hidden, self.item_embedding.weight.T)  # [B, L, I+1]
            logits[:, :, self.pad_idx] = float("-inf")

            loss_per_step = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [B*L, I+1]
                target_ids.view(-1),  # [B*L]
                reduction="none",
            ).view_as(target_ids)

        loss = (loss_per_step * valid_mask).sum() / num_valid
        return loss

    @torch.no_grad()
    def predict_next(self, seq, candidate_ids=None):
        hidden = self._encode(seq)[:, -1, :]  # Only take the last position's logits

        # logits: [batch_size, seq_len, num_candidates]
        if candidate_ids is not None:
            candidate_embs = self.item_embedding(candidate_ids)
            return torch.einsum("bd,bcd->bc", hidden, candidate_embs)
        else:
            return torch.matmul(hidden, self.item_embedding.weight.T)
