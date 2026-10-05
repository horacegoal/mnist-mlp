import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import device, get_loaders
from model import build_model

RUNS_DIR = Path("runs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run folder name under runs/")
    p.add_argument("--device", default="")
    return p.parse_args()


def confusion_matrix_np(preds, targets, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def main():
    args = parse_args()
    run_dir = RUNS_DIR / args.run
    history = json.loads((run_dir / "history.json").read_text())
    a = history["args"]

    dev = torch.device(args.device) if args.device else device()
    model = build_model(a["model"], hidden=tuple(a["hidden"]), activation=a["activation"]).to(dev)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev))
    model.eval()

    _, _, test_loader = get_loaders(batch_size=a["batch_size"], seed=a["seed"])

    all_preds, all_targets, all_images = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(dev))
            all_preds.append(logits.argmax(1).cpu())
            all_targets.append(y)
            all_images.append(x.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    images = torch.cat(all_images)

    acc = (preds == targets).mean()
    print(f"run: {args.run}")
    print(f"test accuracy: {acc:.4f}")

    cm = confusion_matrix_np(preds, targets)
    np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"loss - {args.run}")
    plt.legend()
    plt.savefig(run_dir / "loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"accuracy - {args.run}")
    plt.legend()
    plt.savefig(run_dir / "acc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"confusion matrix - {args.run}")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.savefig(run_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    wrong = np.where(preds != targets)[0]
    n = min(16, len(wrong))
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    mean, std = 0.1307, 0.3081
    for k in range(16):
        ax = axes[k // 4][k % 4]
        ax.axis("off")
        if k < n:
            idx = wrong[k]
            img = images[idx].squeeze().numpy() * std + mean
            ax.imshow(img, cmap="gray")
            ax.set_title(f"t={targets[idx]} p={preds[idx]}", fontsize=8)
    fig.suptitle(f"misclassified examples - {args.run}")
    fig.savefig(run_dir / "misclassified.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote plots + confusion_matrix.csv to {run_dir}")


if __name__ == "__main__":
    main()
