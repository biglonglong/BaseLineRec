import time
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from models.sasrec import SASRec
from trainer import SequenceModelTrainer
from dataset import SequenceDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_items(info_path: Path) -> int:
    with open(info_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return len(lines)


# ==================== CONFIG ====================
# RAW DATA PATHS
DATA_INFO = Path("./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt")
TRAIN_DATA = Path("./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv")
VALID_DATA = Path("./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv")
TEST_DATA = Path("./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv")

# TRAINING CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
USE_SCHEDULER = True

# OTHER CONFIG
SEED = 42
PAD_IDX = 0
MAX_LEN = 10
NUM_ITEMS = get_num_items(DATA_INFO)
NUM_NEGATIVES = 4
TOP_KS = [1, 3, 5, 10]

# MODEL CONFIG
model_name = "SASRec"
sasrec_config = {
    "num_items": NUM_ITEMS,
    "d_model": 128,
    "max_len": MAX_LEN,
    "n_heads": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "n_layers": 2,
    "pad_idx": PAD_IDX,
}

if __name__ == "__main__":
    set_seed(SEED)
    print(f"num_items: {NUM_ITEMS}, device: {DEVICE}")

    train_dataset = SequenceDataset(
        TRAIN_DATA,
        max_len=MAX_LEN,
        pad_idx=PAD_IDX,
        num_negatives=NUM_NEGATIVES,
        num_items=NUM_ITEMS,
    )
    valid_dataset = SequenceDataset(
        VALID_DATA,
        max_len=MAX_LEN,
        pad_idx=PAD_IDX,
        num_negatives=0,
        num_items=NUM_ITEMS,
    )
    test_dataset = SequenceDataset(
        TEST_DATA,
        max_len=MAX_LEN,
        pad_idx=PAD_IDX,
        num_negatives=0,
        num_items=NUM_ITEMS,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if model_name == "SASRec":
        model = SASRec(**sasrec_config).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    trainer = SequenceModelTrainer(
        model,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        use_scheduler=USE_SCHEDULER,
        gradient_clip=GRADIENT_CLIP,
        num_epochs=EPOCHS,
    )

    best_hr = 0.0
    for epoch in range(EPOCHS):
        # training
        loss = trainer.train_epoch(dataloader=train_loader)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # evaluation
        val_metrics = trainer.evaluate(dataloader=valid_loader, k=max(TOP_KS))
        hr = val_metrics.get(f"HR@{max(TOP_KS)}", 0)
        ndcg = val_metrics.get(f"NDCG@{max(TOP_KS)}", 0)
        print(f"  Valid HR@{max(TOP_KS)}: {hr:.4f}, NDCG@{max(TOP_KS)}: {ndcg:.4f}")

        if hr > best_hr:
            best_hr = hr
            trainer.save_checkpoint(f"{model_name}_best_hr.pth")
            print(f"  ✔ New best model saved (HR@{max(TOP_KS)}={hr:.4f})")

    trainer.load_checkpoint(f"{model_name}_best_hr.pth")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    results_file = f"SASRec_final_results_{timestamp}.txt"

    with open(results_file, "w") as f:
        for k in TOP_KS:
            metrics = trainer.evaluate(dataloader=test_loader, k=k)
            hr = metrics.get(f"HR@{k}", 0)
            ndcg = metrics.get(f"NDCG@{k}", 0)
            line = f"HR@{k}: {hr:.4f}, NDCG@{k}: {ndcg:.4f}"
            print(f"Test {line}")
            f.write(line + "\n")

    print(f"\nResults saved to {results_file}")
