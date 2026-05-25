import time
import logging
import torch
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.sasrec import SASRec
from models.gru4rec import GRU4Rec
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


def plot_metrics(all_loss, all_hr, all_ndcg, timestamp):
    epochs = range(1, len(all_loss) + 1)

    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, all_loss, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot HR
    plt.subplot(1, 3, 2)
    plt.plot(epochs, all_hr, label="HR", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("HR")
    plt.title("Validation Hit Rate")
    plt.legend()

    # Plot NDCG
    plt.subplot(1, 3, 3)
    plt.plot(epochs, all_ndcg, label="NDCG", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("NDCG")
    plt.title("Validation NDCG")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"metrics_{timestamp}.png")
    # plt.show()


# ==================== CONFIG ====================
# RAW DATA PATHS
DATA_INFO = Path("./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt")
TRAIN_DATA = Path("./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv")
VALID_DATA = Path("./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv")
TEST_DATA = Path("./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv")

# TRAINING CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
USE_SCHEDULER = True

# OTHER CONFIG
SEED = 42
NUM_ITEMS = get_num_items(DATA_INFO)
MAX_LEN = 10
PAD_IDX = NUM_ITEMS
NUM_NEGATIVES = 0
TOP_KS = [1, 3, 5, 10]

# MODEL CONFIG
# model_name = "SASRec"
model_name = "GRU4Rec"
"""
EPOCHS = 30
BATCH_SIZE = 128
LR = 2e-4
WEIGHT_DECAY = 0.01
NUM_NEGATIVES = 3
"""
sasrec_config = {
    "num_items": NUM_ITEMS,
    "d_model": 512,
    "max_len": MAX_LEN,
    "dropout": 0.2,
    "pad_idx": PAD_IDX,
    "n_heads": 8,
    "d_ff": 2048,
    "n_layers": 4,
}
"""
EPOCHS = 30
BATCH_SIZE = 128
LR = 2e-4
WEIGHT_DECAY = 0.01
NUM_NEGATIVES = 3
"""
gru4rec_config = {
    "num_items": NUM_ITEMS,
    "d_model": 256,
    "max_len": MAX_LEN,
    "dropout": 0.2,
    "pad_idx": PAD_IDX,
    "n_layers": 2,
}

if __name__ == "__main__":
    set_seed(SEED)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("baseline_rec")
    fh = logging.FileHandler(f"training_{timestamp}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

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
    elif model_name == "GRU4Rec":
        model = GRU4Rec(**gru4rec_config).to(DEVICE)
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
    all_loss = []
    all_hr = []
    all_ndcg = []
    for epoch in range(EPOCHS):
        # training
        loss = trainer.train_epoch(dataloader=train_loader)
        all_loss.append(loss)
        logger.info(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # evaluation
        val_metrics = trainer.evaluate(dataloader=valid_loader, k=max(TOP_KS))
        hr = val_metrics.get(f"HR@{max(TOP_KS)}", 0)
        ndcg = val_metrics.get(f"NDCG@{max(TOP_KS)}", 0)
        all_hr.append(hr)
        all_ndcg.append(ndcg)
        logger.info(
            f"  Valid HR@{max(TOP_KS)}: {hr:.4f}, NDCG@{max(TOP_KS)}: {ndcg:.4f}"
        )

        if hr > best_hr:
            best_hr = hr
            trainer.save_checkpoint(f"{model_name}_best_hr_{timestamp}.pth")
            logger.info(f"  ✔ New best model saved (HR@{max(TOP_KS)}={hr:.4f})")

    plot_metrics(all_loss, all_hr, all_ndcg, timestamp)
    trainer.load_checkpoint(f"{model_name}_best_hr_{timestamp}.pth")
    results_file = f"{model_name}_final_{timestamp}.res"

    with open(results_file, "w") as f:
        for k in TOP_KS:
            metrics = trainer.evaluate(dataloader=test_loader, k=k)
            hr = metrics.get(f"HR@{k}", 0)
            ndcg = metrics.get(f"NDCG@{k}", 0)
            line = f"HR@{k}: {hr:.4f}, NDCG@{k}: {ndcg:.4f}"
            print(f"Test {line}")
            f.write(line + "\n")

    print(f"\nResults saved to {results_file}")
