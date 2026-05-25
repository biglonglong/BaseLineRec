import math
import torch
from torch import nn
from models.baserec import PositionalEncoding, BaseSequentialRecModel


class GRU4Rec(BaseSequentialRecModel):
    def __init__(
        self,
        num_items: int,
        d_model: int = 128,
        max_len: int = 10,
        dropout: float = 0.1,
        pad_idx: int = 0,
        n_layers: int = 2,
    ):
        super().__init__(num_items, d_model, max_len, dropout, pad_idx)
        self.pad_idx = pad_idx

        # GRU
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Layer norm for final output
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()
        self._gru_init_weights()

    def _gru_init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:  # input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:  # hidden-to-hidden weights
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # GRU bias has two parts: reset/update gates and new gate
                # Initialize the new gate bias to a negative value
                # to encourage information retention early in training
                n = param.size(0)
                param.data[n // 3 :].fill_(-1.0)

        # Initialize LayerNorm
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

    def get_sequence_lengths(self, seq):
        return (seq != self.pad_idx).sum(dim=1)

    def _encode(self, seq):
        batch_size, seq_len = seq.shape

        # Embedding
        seq_embs = self.item_embedding(seq) * math.sqrt(
            self.d_model
        )  # avoiding interference with position embeddings
        seq_embs = self.pos_encoding(seq_embs)
        seq_embs = self.embed_dropout(seq_embs)

        len_seq = self.get_sequence_lengths(seq)
        if len_seq.sum() > 0:
            packed_seq_embs = nn.utils.rnn.pack_padded_sequence(
                seq_embs, len_seq.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.gru(
                packed_seq_embs
            )  # all steps output: (batch, seq_len, hidden_size), last hidden: (num_layers, batch, hidden_size)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=seq_len
            )
        else:
            output, _ = self.gru(seq_embs)

        return self.final_norm(output)
