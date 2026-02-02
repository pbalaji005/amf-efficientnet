"""
train.py

Trains an EfficientNet-B3 classifier using transfer learning.

Expected dataset layout (local, not committed):
data/
  train/
    ClassA/
    ClassB/
  test/
    ClassA/
    ClassB/

This script:
- loads data/train
- creates an internal train/val split (per-class split)
- trains EfficientNet-B3 (head training by default)
- optionally fine-tunes the last blocks
- saves best model to runs/<run_name>/best_model.pt
- logs metrics to runs/<run_name>/train_log.csv
"""

import os
import csv
import time
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


# -------------------------
# Helpers
# -------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def check_train_dir(data_dir: str) -> str:
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        raise RuntimeError(
            f"Missing folder: {train_dir}\n"
            "Create data/train/<ClassName>/ and add images later."
        )
    return train_dir

def build_transforms(img_size: int):
    # ImageNet normalization (matches pretrained weights)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tfm, val_tfm

def split_indices_by_class(dataset: datasets.ImageFolder, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """
    Per-class split so every class appears in validation.
    This is safer than naive random_split when classes are imbalanced.
    """
    g = torch.Generator().manual_seed(seed)
    targets = torch.tensor(dataset.targets)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for c in range(len(dataset.classes)):
        class_idx = (targets == c).nonzero(as_tuple=False).view(-1)
        class_idx = class_idx[torch.randperm(class_idx.numel(), generator=g)]

        n_val = int(round(class_idx.numel() * val_ratio))
        val_part = class_idx[:n_val].tolist()
        train_part = class_idx[n_val:].tolist()

        val_idx.extend(val_part)
        train_idx.extend(train_part)

    return train_idx, val_idx

def build_model(num_classes: int, dropout: float = 0.3) -> nn.Module:
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)

    # Freeze backbone (head training)
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model

def unfreeze_last_blocks(model: nn.Module, n: int = 2) -> None:
    """
    Optionally unfreezes last N blocks inside model.features for fine-tuning.
    """
    if not hasattr(model, "features"):
        return
    blocks = list(model.features.children())
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data", help="Base dir containing train/ and test/")
    parser.add_argument("--run-name", default="effnet_b3_baseline", help="Folder name under runs/")
    parser.add_argument("--img-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)

    # Optional fine-tune
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    run_dir = os.path.join("runs", args.run_name)
    ensure_dir(run_dir)

    train_dir = check_train_dir(args.data)

    train_tfm, val_tfm = build_transforms(args.img_size)

    # Load training set (with train transforms)
    full_ds = datasets.ImageFolder(train_dir, transform=train_tfm)
    if len(full_ds) == 0:
        raise RuntimeError(
            f"No images found in {train_dir}\n"
            "Add images to data/train/<ClassName>/ first."
        )

    # Same files, different transform for validation
    val_ds_raw = datasets.ImageFolder(train_dir, transform=val_tfm)

    print("Classes:", full_ds.classes)
    num_classes = len(full_ds.classes)

    train_idx, val_idx = split_indices_by_class(full_ds, args.val_ratio, args.seed)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(val_ds_raw, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Head training optimizer (only classifier params)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "epoch", "train_loss", "val_loss", "val_acc", "seconds"])

    best_val_acc = -1.0
    best_path = os.path.join(run_dir, "best_model.pt")

    def train_phase(num_epochs: int, phase: str):
        nonlocal best_val_acc, optimizer

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            model.train()

            total_loss = 0.0
            total = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total += x.size(0)

            train_loss = total_loss / max(total, 1)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            secs = time.time() - t0

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": full_ds.classes,
                        "img_size": args.img_size,
                        "arch": "efficientnet_b3",
                        "val_acc": best_val_acc,
                    },
                    best_path,
                )

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([phase, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{secs:.2f}"])

            print(
                f"[{phase}] Epoch {epoch}/{num_epochs} | "
                f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
                f"ValAcc {val_acc:.4f} | Best {best_val_acc:.4f} | {secs:.1f}s"
            )

    # Phase 1: train head
    train_phase(args.epochs, "head")

    # Phase 2: optional fine-tuning
    if args.fine_tune:
        print(f"Fine-tune enabled: unfreezing last {args.unfreeze_blocks} blocks...")
        unfreeze_last_blocks(model, n=args.unfreeze_blocks)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.ft_lr, weight_decay=args.weight_decay)

        train_phase(args.ft_epochs, "finetune")

    print("\n✅ Training complete.")
    print("Best model saved to:", best_path)
    print("Training log saved to:", log_path)


if __name__ == "__main__":
    main()
