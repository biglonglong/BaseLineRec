import ast
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self, data_path, max_len=10, pad_idx=0, num_negatives=0, num_items=None
    ):
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.num_negatives = num_negatives
        self.num_items = num_items

        self.seqs, self.targets = self._load_data(data_path)

        if self.num_negatives > 0 and self.num_items is None:
            raise ValueError("num_items must be provided when num_negatives > 0")

    def _load_data(self, data_path):
        df = pd.read_csv(data_path)[["history_item_id", "item_id"]]
        df = df.rename(columns={"history_item_id": "seq", "item_id": "next"})

        all_seqs = []
        all_targets = []
        for _, row in df.iterrows():
            try:
                seq = (
                    ast.literal_eval(row["seq"])
                    if isinstance(row["seq"], str)
                    else row["seq"]
                )
            except Exception:
                continue

            if not seq:
                continue

            # left padding
            if len(seq) >= self.max_len:
                padded_seq = seq[-self.max_len :]
            else:
                padded_seq = [self.pad_idx] * (self.max_len - len(seq)) + seq

            next_item = row["next"]
            raw_target = seq[1:] + [next_item]
            if len(raw_target) >= self.max_len:
                padded_target = raw_target[-self.max_len :]
            else:
                padded_target = [self.pad_idx] * (
                    self.max_len - len(raw_target)
                ) + raw_target

            all_seqs.append(padded_seq)
            all_targets.append(padded_target)

        return np.array(all_seqs, dtype=np.int64), np.array(all_targets, dtype=np.int64)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.seqs[idx])
        target = torch.from_numpy(self.targets[idx])

        if self.num_negatives <= 0:
            return seq, target

        # neg per timestamp independently
        k = self.num_negatives
        neg_target = torch.full((self.max_len, k), self.pad_idx, dtype=torch.long)
        for t in range(self.max_len):
            if target[t] != self.pad_idx:
                neg_ids = []
                while len(neg_ids) < k:
                    candidate = random.randint(1, self.num_items - 1)
                    if candidate != target[t].item() and candidate not in neg_ids:
                        neg_ids.append(candidate)

                neg_target[t] = torch.tensor(neg_ids)

        return seq, target, neg_target

    def __len__(self):
        return len(self.seqs)

    def calc_propensity_score(self):
        flat_targets = self.targets.flatten()
        valid = flat_targets[flat_targets != self.pad_idx]
        counts = np.bincount(valid)
        total = len(valid)
        return {int(i): counts[i] / total for i in range(len(counts)) if counts[i] > 0}


if __name__ == "__main__":
    dataset = SequenceDataset(
        Path(
            "./BaseLineRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
        )
    )
    print(dataset[1])
