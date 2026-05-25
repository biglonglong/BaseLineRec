import time
import torch
import random
import logging
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.sasrec import SASRec
from models.gru4rec import GRU4Rec
from models.caser import Caser
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


def plot_metrics(all_loss, all_hr, all_ndcg, timestamp, model_name):
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
    plt.savefig(f"{model_name}_metrics_{timestamp}.png")
    # plt.show()


# ==================== CONFIG ====================
# RAW DATA PATHS
DATA_INFO = Path("./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt")
TRAIN_DATA = Path("./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv")
VALID_DATA = Path("./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv")
TEST_DATA = Path("./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv")

# TRAINING CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_ITEMS = get_num_items(DATA_INFO)
MAX_LEN = 10
PAD_IDX = NUM_ITEMS
TOP_KS = [1, 3, 5, 10]

# MODEL CONFIG
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
gru4rec_config = {
    "num_items": NUM_ITEMS,
    "d_model": 128,
    "max_len": MAX_LEN,
    "dropout": 0.3,
    "pad_idx": PAD_IDX,
    "n_layers": 2,
}
caser_config = {
    "num_items": NUM_ITEMS,
    "d_model": 256,
    "max_len": MAX_LEN,
    "dropout": 0.2,
    "pad_idx": PAD_IDX,
    "num_filters": 16,
    "filter_sizes": [2, 3, 4],
}


def main(
    model_name,
    num_negatives,
    epochs,
    batch_size,
    lr,
    weight_decay,
    gradient_clip,
    use_scheduler,
):
    set_seed(SEED)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("baseline_rec")
    fh = logging.FileHandler(f"{model_name}_training_{timestamp}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    print(f"num_items: {NUM_ITEMS}, device: {DEVICE}")

    train_dataset = SequenceDataset(
        TRAIN_DATA,
        max_len=MAX_LEN,
        pad_idx=PAD_IDX,
        num_negatives=num_negatives,
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if model_name == "SASRec":
        model = SASRec(**sasrec_config).to(DEVICE)
    elif model_name == "GRU4Rec":
        model = GRU4Rec(**gru4rec_config).to(DEVICE)
    elif model_name == "Caser":
        model = Caser(**caser_config).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    trainer = SequenceModelTrainer(
        model,
        learning_rate=lr,
        weight_decay=weight_decay,
        device=DEVICE,
        use_scheduler=use_scheduler,
        gradient_clip=gradient_clip,
        num_epochs=epochs,
    )

    best_hr = 0.0
    all_loss = []
    all_hr = []
    all_ndcg = []
    for epoch in range(epochs):
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

    plot_metrics(all_loss, all_hr, all_ndcg, timestamp, model_name)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument(
        "--model", type=str, default="SASRec", choices=["SASRec", "GRU4Rec", "Caser"]
    )
    parser.add_argument("--num_negatives", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--disable_scheduler", action="store_true")
    args = parser.parse_args()

    main(
        model_name=args.model,
        num_negatives=args.num_negatives,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        use_scheduler=not args.disable_scheduler,
    )
    # python3 main.py --model SASRec
    # python3 main.py --model GRU4Rec --epochs 30 --lr 5e-4 --weight_decay 5e-5
    # python3 main.py --model Caser --epochs 50 --lr 1e-3 --weight_decay 1e-5
