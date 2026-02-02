"""
eval.py

Evaluates a trained checkpoint on the held-out test set.

Expected dataset layout:
data/
  test/
    ClassA/
    ClassB/

Usage:
python src/eval.py --data ./data --weights runs/effnet_b3_baseline/best_model.pt
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import efficientnet_b3

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data", help="Base dir containing test/")
    parser.add_argument("--weights", required=True, help="Path to runs/<run>/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    test_dir = os.path.join(args.data, "test")
    if not os.path.isdir(test_dir):
        raise RuntimeError(
            f"Missing folder: {test_dir}\n"
            "Create data/test/<ClassName>/ and add images later."
        )

    ckpt = torch.load(args.weights, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 300)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    test_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        normalize,
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=test_tfm)
    if len(test_ds) == 0:
        raise RuntimeError(
            f"No images found in {test_dir}\n"
            "Add images to data/test/<ClassName>/ first."
        )

    # Warn if class ordering differs
    if test_ds.classes != classes:
        print("⚠️ Warning: Test folder class order does not match checkpoint classes.")
        print("Checkpoint classes:", classes)
        print("Test dataset classes:", test_ds.classes)
        print("Fix by ensuring the same class folder names exist in train and test.")

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(classes))
    )

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    print("\n✅ Test Accuracy:", round(acc, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=test_ds.classes, digits=4))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
