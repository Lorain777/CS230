"""Train and test U-Net baselines with patient-level cross-validation.

    python -m fibroid.train --slices 1 --model standard --wandb
    python -m fibroid.train --slices 1 --model dropout --wandb
    python -m fibroid.train --slices 1 --model dropblock --wandb
    python -m fibroid.train --slices 1 --model skip_hardship --wandb

    python -m fibroid.train --slices 5 --model standard --wandb
    ...

    Each fold trains on the other folds (minus a validation subset used for early
    stopping), then writes test predictions' metrics to <out>/fold<k>/test.json.
Finished folds are skipped, so an interrupted run can simply be restarted.

Weights & Biases logs to CoreWeave Forge by default
(https://forge.coreweave.com/wandb). Override with --wandb-base-url if needed.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SliceDataset, augment, context_indices, load_cases, load_meta, split_patients
from .losses import DiceBCELoss
from .metrics import THRESHOLDS, dice_score, false_positive_counts, lesion_scores
from .model import VARIANTS, UNet

# CoreWeave Forge multi-tenant W&B API. UI lives at https://forge.coreweave.com/wandb
FORGE_WANDB_BASE_URL = "https://api.forge.coreweave.com"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_wandb(args):
    """Point the W&B SDK at CoreWeave Forge (or a custom host)."""
    if args.wandb_base_url:
        os.environ["WANDB_BASE_URL"] = args.wandb_base_url
    else:
        os.environ.setdefault("WANDB_BASE_URL", FORGE_WANDB_BASE_URL)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)


def maybe_wandb(args, fold):
    """Start a W&B run when --wandb is set; returns the run or None."""
    if not args.wandb:
        return None
    import wandb

    configure_wandb(args)
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": f"{args.model}_s{args.slices}_fold{fold}",
        "group": f"{args.model}_s{args.slices}",
        "job_type": "train",
        "config": {
            "model": args.model,
            "slices": args.slices,
            "fold": fold,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropblock_keep_prob": args.dropblock_keep_prob,
            "dropblock_size": args.dropblock_size,
            "skip_drop_prob": args.skip_drop_prob,
            "dropout_p": args.dropout_p,
            "wandb_base_url": os.environ.get("WANDB_BASE_URL"),
        },
        "reinit": True,
    }
    # Prefer Settings when the installed wandb build supports base_url.
    try:
        init_kwargs["settings"] = wandb.Settings(base_url=os.environ.get("WANDB_BASE_URL"))
    except TypeError:
        pass
    return wandb.init(**init_kwargs)


def scheduled_keep_prob(epoch, epochs, final_keep_prob):
    """Linear schedule from 1.0 to final_keep_prob over training (DropBlock paper)."""
    if epochs <= 1:
        return final_keep_prob
    progress = (epoch - 1) / (epochs - 1)
    return 1.0 - progress * (1.0 - final_keep_prob)


@torch.no_grad()
def predict_volume(model, case, n_slices, device, batch_size=32):
    """Fibroid probability for every slice of one patient, shape (Z, H, W)."""
    model.eval()
    context = context_indices(case["image"].shape[0], n_slices)
    probs = []
    for rows in context.split(batch_size):
        inputs = case["image"][rows].float().to(device)
        probs.append(torch.sigmoid(model(inputs))[:, 0].cpu())
    return torch.cat(probs).numpy()


def mean_dice(model, cases, n_slices, device):
    scores = [
        dice_score(predict_volume(model, case, n_slices, device) >= 0.5, case["myoma"].numpy())
        for case in cases
    ]
    return float(np.mean(scores))


@torch.no_grad()
def mean_val_loss(model, cases, n_slices, device, criterion, batch_size=32):
    """Average Dice+BCE loss over validation patients (slice minibatches)."""
    model.eval()
    total, n = 0.0, 0
    center = n_slices // 2
    for case in cases:
        context = context_indices(case["image"].shape[0], n_slices)
        target_vol = case["myoma"].float()
        for rows in context.split(batch_size):
            images = case["image"][rows].float().to(device)
            targets = target_vol[rows[:, center]].unsqueeze(1).to(device)
            total += criterion(model(images), targets).item() * rows.shape[0]
            n += rows.shape[0]
    return total / max(n, 1)


def test_metrics(model, cases, n_slices, device):
    patients, lesions = [], []
    for case in cases:
        prob = predict_volume(model, case, n_slices, device)
        truth = case["myoma"].numpy()
        patients.append(
            {
                "id": case["id"],
                "dice": dice_score(prob >= 0.5, truth),
                "false_positives": false_positive_counts(prob, truth),
            }
        )
        for voxels, score in lesion_scores(prob, truth):
            lesions.append({"patient": case["id"], "volume_ml": voxels * case["voxel_ml"], "score": score})
    return {"thresholds": THRESHOLDS.tolist(), "patients": patients, "lesions": lesions}


def build_model(args):
    return UNet(
        args.slices,
        variant=args.model,
        dropblock_keep_prob=args.dropblock_keep_prob,
        dropblock_size=args.dropblock_size,
        skip_drop_prob=args.skip_drop_prob,
        dropout_p=args.dropout_p,
    )


def train_fold(args, fold, meta, device):
    out_dir = args.out / f"fold{fold}"
    if (out_dir / "test.json").exists():
        print(f"fold {fold}: already finished, skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + fold)

    splits = split_patients(meta, fold, exclude_irregular=args.exclude_irregular)
    if args.limit_patients:
        splits = [ids[: args.limit_patients] for ids in splits]
    train_cases, val_cases, test_cases = (load_cases(args.cache, ids) for ids in splits)
    loader = DataLoader(
        SliceDataset(train_cases, args.slices), batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    print(
        f"fold {fold}: model={args.model}  {len(train_cases)} train / {len(val_cases)} val / "
        f"{len(test_cases)} test patients, {len(loader.dataset)} training slices",
        flush=True,
    )

    model = build_model(args).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run = maybe_wandb(args, fold)

    best_dice, best_epoch = -1.0, 0
    train_started = time.time()
    try:
        with open(out_dir / "log.csv", "w", newline="") as log_file:
            log = csv.writer(log_file)
            log.writerow(["epoch", "train_loss", "val_loss", "val_dice", "keep_prob", "seconds"])
            for epoch in range(1, args.epochs + 1):
                start = time.time()
                keep_prob = scheduled_keep_prob(epoch, args.epochs, args.dropblock_keep_prob)
                model.set_keep_prob(keep_prob)
                model.train()
                running = 0.0
                for images, targets in loader:
                    if not args.no_augment:
                        images, targets = augment(images, targets)
                    images, targets = images.to(device), targets.to(device)
                    loss = criterion(model(images), targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running += loss.item()
                train_loss = running / len(loader)
                val_loss = mean_val_loss(model, val_cases, args.slices, device, criterion)
                val_dice = mean_dice(model, val_cases, args.slices, device)
                seconds = time.time() - start
                elapsed = time.time() - train_started
                log.writerow(
                    [
                        epoch,
                        f"{train_loss:.4f}",
                        f"{val_loss:.4f}",
                        f"{val_dice:.4f}",
                        f"{keep_prob:.4f}",
                        f"{seconds:.0f}",
                    ]
                )
                log_file.flush()
                print(
                    f"fold {fold} epoch {epoch:3d}  train/loss {train_loss:.4f}  "
                    f"val/loss {val_loss:.4f}  val/dice {val_dice:.4f}  "
                    f"keep {keep_prob:.3f}  {seconds:.0f}s",
                    flush=True,
                )
                if run is not None:
                    # Forge charts: epoch (x), time/*, train/*, val/*
                    run.log(
                        {
                            "epoch": epoch,
                            "train/loss": train_loss,
                            "val/loss": val_loss,
                            "val/dice": val_dice,
                            "train/keep_prob": keep_prob,
                            "time/epoch_seconds": seconds,
                            "time/elapsed_seconds": elapsed,
                        },
                        step=epoch,
                    )

                if val_dice > best_dice:
                    best_dice, best_epoch = val_dice, epoch
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "val_dice": val_dice,
                            "val_loss": val_loss,
                            "variant": args.model,
                        },
                        out_dir / "best.pt",
                    )
                elif epoch - best_epoch >= args.patience:
                    print(f"fold {fold}: no improvement for {args.patience} epochs, stopping")
                    break

        checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        results = test_metrics(model, test_cases, args.slices, device)
        results.update(
            fold=fold,
            model=args.model,
            n_slices=args.slices,
            exclude_irregular=args.exclude_irregular,
            dropblock_keep_prob=args.dropblock_keep_prob,
            dropblock_size=args.dropblock_size,
            skip_drop_prob=args.skip_drop_prob,
            dropout_p=args.dropout_p,
            best_epoch=best_epoch,
            best_val_dice=best_dice,
        )
        (out_dir / "test.json").write_text(json.dumps(results))
        test_dice = np.mean([patient["dice"] for patient in results["patients"]])
        print(f"fold {fold}: best epoch {best_epoch}, val dice {best_dice:.4f}, test dice {test_dice:.4f}", flush=True)
        if run is not None:
            run.summary["best_epoch"] = best_epoch
            run.summary["best_val_dice"] = best_dice
            run.summary["test_dice"] = test_dice
            run.save(str(out_dir / "test.json"))
    finally:
        if run is not None:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slices", type=int, required=True, help="adjacent slices fed as input channels (odd)")
    parser.add_argument("--model", choices=VARIANTS, default="standard", help="baseline variant")
    parser.add_argument("--folds", type=int, nargs="+", help="folds to run (default: all)")
    parser.add_argument("--cache", type=Path, default=Path("data/umd/cache256"))
    parser.add_argument("--out", type=Path, help="output directory (default: runs/<model>_s<slices>)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15, help="early-stopping patience in epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--exclude-irregular", action="store_true", help="drop the 33 patients whose masks lack geometry")
    parser.add_argument("--limit-patients", type=int, help="use only this many patients per split (smoke tests)")
    parser.add_argument("--dropblock-keep-prob", type=float, default=0.9, help="final DropBlock keep probability")
    parser.add_argument("--dropblock-size", type=int, default=7, help="DropBlock square size")
    parser.add_argument("--skip-drop-prob", type=float, default=0.25, help="skip-hardship drop probability per skip")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="Dropout2d probability for --model dropout")
    parser.add_argument("--wandb", action="store_true", help="log each fold to Weights & Biases (CoreWeave Forge)")
    parser.add_argument("--wandb-project", default="umd-fibroid")
    parser.add_argument(
        "--wandb-entity",
        default="elvayu-stanford-university",
        help="W&B entity/team on Forge",
    )
    parser.add_argument(
        "--wandb-base-url",
        default=FORGE_WANDB_BASE_URL,
        help="W&B API host (default: CoreWeave Forge)",
    )
    args = parser.parse_args()
    if args.slices % 2 != 1:
        parser.error("--slices must be odd")
    args.out = args.out or Path(f"runs/{args.model}_s{args.slices}")
    if args.wandb:
        configure_wandb(args)

    meta = load_meta(args.cache)
    device = pick_device()
    print(
        f"device: {device}  model: {args.model}  slices: {args.slices}  out: {args.out}"
        + (f"  wandb: {os.environ.get('WANDB_BASE_URL')}/{args.wandb_project}" if args.wandb else "")
    )
    for fold in args.folds if args.folds is not None else range(meta["n_folds"]):
        train_fold(args, fold, meta, device)


if __name__ == "__main__":
    main()
