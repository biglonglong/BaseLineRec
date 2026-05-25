import math
import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


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


class BaseSequentialRecModel(nn.Module, ABC):
    def __init__(
        self,
        num_items: int,
        d_model: int = 128,
        max_len: int = 10,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx

        # Common components
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

    @abstractmethod
    def _encode(self, seq):
        """Encode sequence into hidden representations"""
        pass

    def forward(self, seq):
        """Forward pass, returns hidden states for all positions"""
        return self._encode(seq)

    def calculate_loss(self, seq, target_ids, neg_target_ids=None):
        """
        Calculate loss for training

        Args:
            seq: [batch_size, seq_len] - Input sequences
            target_ids: [batch_size, seq_len] - Target items
            neg_target_ids: [batch_size, seq_len, num_negatives] - Negative items (optional)

        Returns:
            loss: scalar tensor
        """
        hidden = self._encode(seq)  # Can be [B, D] or [B, L, D]

        # when hidden is [B, D], we are doing next-item prediction for the last position only
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)  # [B, 1, D]
            target_ids = target_ids[:, -1:]  # [B, 1]
            if neg_target_ids is not None:
                neg_target_ids = neg_target_ids[:, -1:, :]  # [B, 1, K]

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
            # Avoid using -inf here for the same reason as above. Use a large
            # negative finite value so log-softmax remains numerically stable.
            logits[:, :, self.pad_idx] = -1e9

            loss_per_step = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [B*L, I+1]
                target_ids.view(-1),  # [B*L]
                reduction="none",
            ).view_as(target_ids)

        loss = (loss_per_step * valid_mask).sum() / num_valid
        return loss

    @torch.no_grad()
    def predict_next(self, seq, candidate_ids=None):
        """
        Predict next item(s) for the last position in each sequence

        Args:
            seq: [batch_size, seq_len] - Input sequences
            candidate_ids: Optional [batch_size, num_candidates] or [batch_size, seq_len, num_candidates]

        Returns:
            If candidate_ids is None: [batch_size, num_items+1] - Logits for all items
            If candidate_ids is provided: [batch_size, num_candidates] - Scores for candidate items
        """
        hidden = self._encode(seq)
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]  # Only take the last position's logits

        # logits: [batch_size, seq_len, num_candidates]
        if candidate_ids is not None:
            if candidate_ids.dim() == 3:
                candidate_ids = candidate_ids[:, -1, :]
            candidate_embs = self.item_embedding(candidate_ids)
            return torch.einsum("bd,bcd->bc", hidden, candidate_embs)
        else:
            return torch.matmul(hidden, self.item_embedding.weight.T)

    def _init_weights(self):
        """Common weight initialization"""
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_encoding.pos_embedding.weight, std=0.02)

        with torch.no_grad():
            self.item_embedding.weight[self.pad_idx].zero_()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
