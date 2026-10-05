import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import device, get_loaders
from model import build_model

RUNS_DIR = Path("runs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--hidden", nargs="+", type=int, default=[128, 64])
    p.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="relu")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="")
    p.add_argument("--device", default="")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, dev):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device) if args.device else device()
    print(f"device: {dev}")

    train_loader, val_loader, test_loader = get_loaders(batch_size=args.batch_size, seed=args.seed)

    model = build_model(args.model, hidden=tuple(args.hidden), activation=args.activation).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    if args.run_name:
        run_name = args.run_name
    elif args.model == "linear":
        run_name = f"linear_lr{args.lr}"
    else:
        run_name = f"{args.model}_{args.activation}_{'-'.join(map(str, args.hidden))}_lr{args.lr}"
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, dev)
        va_loss, va_acc = evaluate(model, val_loader, criterion, dev)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_dir / "best.pt")
        print(f"epoch {epoch:2d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.1f}s")

    model.load_state_dict(torch.load(out_dir / "best.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, dev)
    print(f"best val acc {best_val_acc:.4f} | test loss {test_loss:.4f} | test acc {test_acc:.4f}")

    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["args"] = vars(args)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
