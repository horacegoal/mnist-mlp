# LEARN — build MNIST MLP yourself

You write every line. The reference lives in `answer_key/` — **do not open it until M6.**
Run everything with `.venv/bin/python`.

## Rules

- One checkpoint per module. Don't advance until its acceptance test passes.
- Stuck? Ask for a hint (question / docs pointer) before asking for code.
- Review happens on your pasted code or error, at line level.
- Full scope: M1–M5. PyTorch only (no manual backprop).

## Progress

- [ ] M0 — tensor warm-up
- [ ] M1 — `data.py`
- [ ] M2 — `model.py`
- [ ] M3 — `train.py` core
- [ ] M4 — `train.py` driver
- [ ] M5 — `evaluate.py`
- [ ] M6 — compare + ablate

---

## M0 — Tensor warm-up

No file. In a REPL or throwaway script, get comfortable with:

- creating tensors, reading `.shape`, `.dtype`, `.device`
- matrix multiply with `@`
- `requires_grad=True`, calling `.backward()`, reading `.grad`
- moving to GPU with `.to("mps")`

Done when autograd feels like: "forward makes a number, backward fills in `.grad`."

## M1 — `data.py`

Spec (must expose exactly these names):

```
device() -> torch.device
get_loaders(batch_size=128, val_size=5000, seed=42, num_workers=2)
    -> (train_loader, val_loader, test_loader)
```

Behavior:

- MNIST downloaded to `./data`.
- train split into train + val (`val_size` examples); test stays untouched.
- each batch is images `[B, 1, 28, 28]` and labels `[B]`.
- transform: `ToTensor`, then normalize with MNIST mean 0.1307 / std 0.3081.
- `device()` returns `mps` if available, else `cuda`, else `cpu`.

API to learn: `torchvision.datasets.MNIST`, `transforms.Compose/ToTensor/Normalize`,
`torch.utils.data.random_split/DataLoader`.

Acceptance: a `__main__` that prints one batch's shape `(128, 1, 28, 28)`,
dtype, label shape, and the device.

## M2 — `model.py`

Spec:

```
class LinearModel(nn.Module)   # flatten -> one Linear(784, 10)
class MLP(nn.Module)           # flatten -> [784 -> *hidden] with activations -> Linear(., 10)
build_model(name, hidden=(128, 64), activation="relu") -> nn.Module
```

Behavior:

- `activation` in {`relu`, `tanh`, `gelu`}; unknown name raises `ValueError`.
- `build_model("linear")` / `build_model("mlp", ...)`; unknown name raises `ValueError`.
- `hidden` is a tuple, e.g. `(128, 64)`.

API to learn: `nn.Module`, `nn.Flatten`, `nn.Linear`, `nn.Sequential`, `nn.ReLU/Tanh/GELU`.

Acceptance: `MLP()(torch.randn(4, 1, 28, 28)).shape == (4, 10)` and
`LinearModel()(torch.randn(4, 1, 28, 28)).shape == (4, 10)`.

## M3 — `train.py` core

Spec:

```
run_epoch(model, loader, criterion, optimizer, dev) -> (loss, acc)
evaluate(model, loader, criterion, dev) -> (loss, acc)
```

Behavior:

- `run_epoch`: train mode; the 5 steps per batch — zero_grad, forward, loss,
  backward, step. Averages loss over all examples; accuracy = fraction correct.
- `evaluate`: no gradients (`@torch.no_grad()`), eval mode, same loss/acc math.
- correctness measured with `argmax(1) == y`.

API to learn: `nn.CrossEntropyLoss`, `torch.optim.Adam`, `@torch.no_grad()`,
`Tensor.argmax`, `.item()`.

Acceptance: a temporary script trains on a subset for ~3 epochs and loss
visibly drops.

## M4 — `train.py` driver

Spec: `main()` with argparse:

```
--model {linear,mlp}   default mlp
--hidden int [int ...] default [128, 64]
--activation {relu,tanh,gelu} default relu
--epochs int           default 10
--lr float             default 1e-3
--batch-size int       default 128
--seed int             default 42
--run-name str         default ""
--device str           default ""
```

Behavior:

- seed init; pick device (arg overrides `data.device()`).
- loop epochs: `run_epoch` then `evaluate` on val; record history.
- save `runs/<run-name>/best.pt` whenever val acc improves.
- after training, reload best and evaluate on **test once**.
- write `runs/<run-name>/history.json` (per-epoch train/val loss+acc, test
  metrics, args).
- default run name: `linear_lr<lr>` or `<model>_<activation>_<h1-h2>_lr<lr>`.

API to learn: `argparse`, `model.state_dict()`, `torch.save/load`, `json`, `pathlib.Path`.

Acceptance:

```
.venv/bin/python train.py --model linear --epochs 5
.venv/bin/python train.py --model mlp --hidden 128 64 --activation relu --epochs 10
```

linear ≈ 0.92, mlp ≈ 0.97–0.98 test accuracy.

## M5 — `evaluate.py`

Spec: `--run` required (folder under `runs/`), optional `--device`.

Behavior: rebuild the model from `history.json` args, load `best.pt`, run test
set, then write into the run folder:

- `confusion_matrix.csv` (rows=true, cols=pred)
- `confusion_matrix.png`
- `loss_curve.png` (train vs val)
- `acc_curve.png` (train vs val)
- `misclassified.png` (grid of wrong predictions, with true/pred labels)

Behavior detail: de-normalize images for display (`img * std + mean`).
Use a non-interactive matplotlib backend so it runs headless.

API to learn: `matplotlib.pyplot`, `numpy`, per-class counting.

Acceptance: running it on your `mlp` run produces all five artifacts and prints
test accuracy matching `history.json`.

## M6 — Compare + ablate

- [ ] Diff your files against `answer_key/` — note anything surprising.
- [ ] Ablations, one knob at a time, record `test_acc`:

| run | change | test acc |
|---|---|---|
| baseline mlp | — | |
| depth | `--hidden 128` | |
| depth | `--hidden 256 128 64` | |
| width | `--hidden 512 512` | |
| activation | `--activation tanh` | |
| activation | `--activation gelu` | |
| lr | `--lr 1e-2` | |
| lr | `--lr 1e-4` | |
| epochs | `--epochs 1` | |
| epochs | `--epochs 20` | |
| batch | `--batch-size 16` | |
| batch | `--batch-size 512` | |

Observations to write down: which changes help, which hurt, signs of
under/overfitting.
