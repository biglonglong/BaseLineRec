import torch
from torch import nn


class SequenceModelTrainer:
    def __init__(
        self,
        model,  # GRU, SASRec ...
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: str = "cuda",
        use_scheduler: bool = True,
        gradient_clip: float = 1.0,
        num_epochs: int = 20,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_scheduler = use_scheduler
        self.gradient_clip = gradient_clip

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        else:
            self.scheduler = None

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            if len(batch) == 3:
                seq, targets, neg_targets = batch
                neg_targets = neg_targets.to(self.device)
            elif len(batch) == 2:
                seq, targets = batch
                neg_targets = None
            else:
                raise ValueError(
                    "Batch must contain either (seq, targets) or (seq, targets, neg_targets)"
                )

            seq = seq.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(seq, targets, neg_targets)
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if self.scheduler is not None:
            self.scheduler.step()

        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader, metrics=["hr", "ndcg"], k=5):
        self.model.eval()
        metric_sums = {m: 0.0 for m in metrics}
        total_samples = 0

        for batch in dataloader:
            if len(batch) == 3:
                seq, targets, _ = batch
            elif len(batch) == 2:
                seq, targets = batch
            else:
                raise ValueError(
                    "Batch must contain either (seq, targets) or (seq, targets, neg_targets)"
                )

            seq = seq.to(self.device)
            targets = targets.to(self.device)

            batch_prediction_logits = self.model.predict_next(seq)
            batch_targets = targets[:, -1]
            bs = batch_targets.size(0)

            if "hr" in metrics:
                metric_sums["hr"] += (
                    self._hit_rate(batch_prediction_logits, batch_targets, k) * bs
                )
            if "ndcg" in metrics:
                metric_sums["ndcg"] += (
                    self._ndcg(batch_prediction_logits, batch_targets, k) * bs
                )
            if "mrr" in metrics:
                metric_sums["mrr"] += (
                    self._mrr(batch_prediction_logits, batch_targets, k) * bs
                )

            total_samples += bs

        results = {}
        key_map = {"hr": "HR", "ndcg": "NDCG", "mrr": "MRR"}
        for m in metrics:
            results[f"{key_map.get(m, m.upper())}@{k}"] = metric_sums[m] / max(
                total_samples, 1
            )

        return results

    def _hit_rate(self, prediction_logits, targets, k=10):
        """exist in top-k"""
        top_k = torch.topk(prediction_logits, k, dim=1).indices
        is_hit = top_k == targets.unsqueeze(1)

        hit_per_sample = is_hit.any(dim=1).float()
        return hit_per_sample.mean().item()

    def _ndcg(self, prediction_logits, targets, k=10):
        """1.0 / log2_position in top-k"""
        top_k = torch.topk(prediction_logits, k, dim=1).indices
        is_hit = top_k == targets.unsqueeze(1)

        ranks = is_hit.float().argmax(dim=1).float() + 1
        ranks[~is_hit.any(dim=1)] = float("inf")

        # idcg = 1 / torch.log2(torch.arange(2, k + 2, device=targets.device).float())
        # ndcg = dcg / idcg[: len(dcg)]
        dcg = torch.where(
            ranks <= k,
            1.0 / torch.log2(ranks + 1),
            torch.zeros_like(ranks, dtype=torch.float),
        )
        idcg = 1.0 / torch.log2(torch.tensor(2.0, device=prediction_logits.device))

        return (dcg / idcg).mean().item()

    def _mrr(self, prediction_logits, targets, k=10):
        """1.0 / position in top-k"""
        top_k = torch.topk(prediction_logits, k, dim=1).indices
        is_hit = top_k == targets.unsqueeze(1)

        ranks = is_hit.float().argmax(dim=1) + 1
        ranks[~is_hit.any(dim=1)] = float("inf")

        mrr = torch.where(
            ranks <= k, 1.0 / ranks, torch.zeros_like(ranks, dtype=torch.float)
        )
        return mrr.mean().item()

    def save_checkpoint(self, filepath):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
