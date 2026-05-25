import math
import torch
from torch import nn
import torch.nn.functional as F
from models.baserec import PositionalEncoding, BaseSequentialRecModel


class Caser(BaseSequentialRecModel):
    def __init__(
        self,
        num_items: int,
        d_model: int = 128,
        max_len: int = 10,
        dropout: float = 0.1,
        pad_idx: int = 0,
        num_filters: int = 16,
        filter_sizes: list = [2, 3, 4],
    ):
        super().__init__(num_items, d_model, max_len, dropout, pad_idx)
        self.pad_idx = pad_idx

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (i, d_model)) for i in filter_sizes]
        )

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (max_len, 1))

        # Fully Connected Layer to project to hidden dimension
        self.num_filters_total = num_filters * len(filter_sizes)
        final_dim = d_model + self.num_filters_total
        self.projection = nn.Linear(final_dim, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm for final output
        self.final_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()
        self._rnn_init_weights()

    def _rnn_init_weights(self):
        # Initialize horizontal CNNs
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            if cnn.bias is not None:
                nn.init.constant_(cnn.bias, 0.1)

        # Initialize vertical CNN
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        if self.vertical_cnn.bias is not None:
            nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def _encode(self, seq):
        batch_size, seq_len = seq.shape

        # Embedding
        seq_embs = self.item_embedding(seq) * math.sqrt(
            self.d_model
        )  # avoiding interference with position embeddings
        seq_embs = self.pos_encoding(seq_embs)
        seq_embs = self.embed_dropout(seq_embs)

        # Apply mask for padding positions
        mask = (seq != self.pad_idx).float().unsqueeze(-1)
        seq_embs = seq_embs * mask

        # Add channel dimension: [batch, 1, seq_len, d_model]
        seq_embs = seq_embs.unsqueeze(1)

        # Horizontal convolutions
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            # cnn input: [batch, 1, seq_len, d_model]
            # output: [batch, num_filters, seq_len - filter_size + 1, 1]
            h_out = F.relu(cnn(seq_embs))
            h_out = h_out.squeeze(-1)  # [batch, num_filters, H]
            # Max pooling over time
            p_out = F.max_pool1d(h_out, h_out.size(2))  # [batch, num_filters, 1]
            p_out = p_out.squeeze(-1)  # [batch, num_filters]
            pooled_outputs.append(p_out)
        # Concatenate horizontal features
        h_pool = torch.cat(pooled_outputs, dim=1)  # [batch, num_filters_total]

        # Vertical convolution
        v_out = F.relu(self.vertical_cnn(seq_embs))  # [batch, 1, 1, d_model]
        v_flat = v_out.squeeze(1).squeeze(1)  # [batch, d_model]

        # Concatenate horizontal and vertical features
        out = torch.cat([h_pool, v_flat], dim=1)  # [batch, d_model + num_filters_total]
        out = self.dropout(out)

        # Project to d_model dimension to match base class expectations
        hidden = self.projection(out)  # [batch, d_model]

        return self.final_norm(hidden)
