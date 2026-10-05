# MNIST MLP

Learning exercise: go from a linear classifier to a multi-layer perceptron on MNIST.

## Setup

```
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```
.venv/bin/python data.py                       # sanity-check the data pipeline
.venv/bin/python train.py --model linear --epochs 5
.venv/bin/python train.py --model mlp --hidden 128 64 --activation relu --epochs 10
.venv/bin/python evaluate.py --run <run-name>
```

Each `train.py` run writes to `runs/<run-name>/`:

- `best.pt` — weights at best val accuracy
- `history.json` — per-epoch train/val loss + acc, plus test metrics

`evaluate.py` adds loss/accuracy curves, a confusion matrix (png + csv), and a
grid of misclassified digits.

## Layout

| File | Purpose |
|---|---|
| `data.py` | MNIST loaders, train/val split, device helper |
| `model.py` | `LinearModel`, `MLP` |
| `train.py` | training/eval loop, argparse entrypoint |
| `evaluate.py` | metrics + plots for a saved run |

## Milestones

1. `data.py` sanity check — batch shapes `[B, 1, 28, 28]`.
2. Linear baseline — target ~92% test.
3. MLP (128, 64) + ReLU — target ~97-98% test.
4. Evaluate: confusion matrix, curves, misclassified grid.
5. Ablations (depth, width, activation, lr).

## Results

| run | model | hidden | activation | lr | epochs | test acc |
|---|---|---|---|---|---|---|
| `linear_lr0.001` | linear | - | - | 1e-3 | 5 | 0.9213 |
| `mlp_relu_128-64_lr0.001` | mlp | 128,64 | relu | 1e-3 | 10 | 0.9774 |

## Notes

- The python.org build lacks CA certs, so `data.py` points `SSL_CERT_FILE` at
  `certifi` (installed via `requirements.txt`) to let the first MNIST download
  verify TLS. Without it you get `CERTIFICATE_VERIFY_FAILED`.
- Training runs on the Apple GPU (`mps`) automatically; pass `--device cpu` to
  force CPU.
