"""dl/train.py — Training loop for the DL comparison module.

Usage
-----
  python -m dl.train --arch gru --config raw-seq --seeds 0 1 2 3 4
  python -m dl.train --arch lstm --config raw-seq --seeds 0 1 2 3 4
  python -m dl.train --arch gru --config feat110-seq --seeds 0 1 2

All artefacts go under dl/artifacts/.
Never modifies the production RF, splits, or data files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.optim import AdamW

from .config import (
    BATCH_SIZE,
    DL_ARTIFACTS,
    EPOCHS,
    LR,
    PATIENCE,
    SEQ_LEN_FEAT110,
    SEQ_LEN_RAW,
    get_device_str,
)
from .data import load_splits
from .model import build_model

# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Val PR-AUC helper
# ---------------------------------------------------------------------------


def _val_prauc(model: nn.Module, val_dl, device: torch.device) -> float:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y, c in val_dl:
            x, y, c = x.to(device), y.to(device), c.to(device)
            logit = model(x, c).squeeze(1)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs.append(prob)
            labels.append(y.cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, probs))


# ---------------------------------------------------------------------------
# Single-run trainer
# ---------------------------------------------------------------------------


def train_one_run(
    arch: str,
    input_config: str,
    seed: int,
    device: torch.device,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    save_dir: Path | None = None,
) -> dict:
    """
    Train one model configuration with one seed.
    Returns metrics dict with val_prauc, checkpoint path, and timing.
    """
    set_seed(seed)

    seq_len = SEQ_LEN_RAW if input_config == "raw-seq" else SEQ_LEN_FEAT110
    tag = f"{arch}_{input_config.replace('-','_')}_seed{seed}"

    if save_dir is None:
        save_dir = DL_ARTIFACTS / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = save_dir / "scaler.pkl"
    ckpt_path = save_dir / "best.pt"
    run_info_path = save_dir / "run_info.json"

    print(f"\n{'='*60}")
    print(f"  Training: arch={arch}  config={input_config}  seed={seed}  device={device}")
    print(f"{'='*60}")

    # Load data
    data = load_splits(
        input_config=input_config,
        seq_len=seq_len,
        batch_size=batch_size,
        refit_scaler=True,
        scaler_path=scaler_path,
    )
    train_dl = data["train_dl"]
    val_dl = data["val_dl"]
    n_feat = data["n_features"]
    pos_w = data["pos_weight"]

    # Model
    model = build_model(n_features=n_feat, arch=arch).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=device)
    )

    # Training loop
    best_val_prauc = -1.0
    best_epoch = 0
    wait = 0
    train_losses: list[float] = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y, c in train_dl:
            x, y, c = x.to(device), y.to(device), c.to(device)
            optimizer.zero_grad()
            logit = model(x, c).squeeze(1)
            loss = criterion(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_prauc = _val_prauc(model, val_dl, device)
        train_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:03d}  loss={avg_loss:.4f}  val_PR-AUC={val_prauc:.4f}")

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"  Early stop at epoch {epoch} (best={best_epoch}, val_PR-AUC={best_val_prauc:.4f})"
                )
                break

    elapsed = time.time() - t_start
    print(
        f"  Finished in {elapsed:.1f}s  best_val_PR-AUC={best_val_prauc:.4f} (epoch {best_epoch})"
    )

    run_info = {
        "arch": arch,
        "input_config": input_config,
        "seed": seed,
        "seq_len": seq_len,
        "n_features": n_feat,
        "pos_weight": round(pos_w, 4),
        "best_epoch": best_epoch,
        "best_val_prauc": round(best_val_prauc, 6),
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "scaler": str(scaler_path),
        "device": str(device),
    }
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return run_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DL comparison models")
    parser.add_argument("--arch", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--config", choices=["raw-seq", "feat110-seq"], default="raw-seq")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda:0)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(get_device_str())
    print(f"Using device: {device}")

    all_infos = []
    for seed in args.seeds:
        info = train_one_run(
            arch=args.arch,
            input_config=args.config,
            seed=seed,
            device=device,
        )
        all_infos.append(info)

    # Save summary
    summary_path = DL_ARTIFACTS / f"summary_{args.arch}_{args.config.replace('-','_')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_infos, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")

    val_praucs = [i["best_val_prauc"] for i in all_infos]
    print(f"Val PR-AUC across seeds: {[round(v,4) for v in val_praucs]}")
    print(f"  mean={np.mean(val_praucs):.4f}  std={np.std(val_praucs):.4f}")


if __name__ == "__main__":
    main()
