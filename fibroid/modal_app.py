"""Modal app for UMD fibroid baselines with CoreWeave Forge W&B logging.

Setup (once):
    modal volume create umd-fibroid
    modal secret create wandb WANDB_API_KEY=...   # key from forge.coreweave.com/settings
    modal run -m fibroid.modal_app::upload_cache     # uploads local data/umd/cache256
    # or: modal run -m fibroid.modal_app::prepare_remote  # download+prepare on Modal

Validate Modal + Forge W&B:
    modal run -m fibroid.modal_app::validate

Train (examples; always logs to https://forge.coreweave.com/wandb):
    modal run -m fibroid.modal_app::train -- --slices 1 --model standard --folds 0
    modal run -m fibroid.modal_app::train -- --slices 1 --model dropout
    modal run -m fibroid.modal_app::train -- --slices 5 --model dropblock

Dropout / DropBlock baseline matrix (standard + dropout + dropblock × 1/3/5 slices):
    modal run -m fibroid.modal_app::baselines -- --folds 0
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

APP_NAME = "umd-fibroid"
VOLUME_NAME = "umd-fibroid"
MOUNT = Path("/data")
CACHE = MOUNT / "cache256"
RUNS = MOUNT / "runs"
WANDB_PROJECT = "umd-fibroid"
WANDB_ENTITY = "elvayu-stanford-university"
# CoreWeave Forge multi-tenant W&B. UI: https://forge.coreweave.com/wandb
WANDB_BASE_URL = "https://api.forge.coreweave.com"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.1.3",
        "scipy==1.14.1",
        "wandb==0.19.8",
        "tqdm==4.67.1",
    )
    .add_local_python_source("fibroid")
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _configure_forge_wandb(project: str = WANDB_PROJECT, entity: str = WANDB_ENTITY) -> None:
    import os

    os.environ.setdefault("WANDB_BASE_URL", WANDB_BASE_URL)
    os.environ.setdefault("WANDB_PROJECT", project)
    if entity:
        os.environ.setdefault("WANDB_ENTITY", entity)


@app.function(volumes={str(MOUNT): volume}, timeout=60 * 60)
def upload_cache(local_cache: str = "data/umd/cache256"):
    """Copy a local cache directory into the Modal volume."""
    import shutil

    src = Path(local_cache)
    if not src.exists():
        raise FileNotFoundError(f"Local cache not found: {src}")
    if CACHE.exists():
        shutil.rmtree(CACHE)
    shutil.copytree(src, CACHE)
    volume.commit()
    n = len(list(CACHE.glob("*.npz")))
    print(f"Uploaded {n} cached patients to {CACHE}")
    return {"patients": n, "cache": str(CACHE)}


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    timeout=60 * 60,
    cpu=4,
    memory=8192,
)
def prepare_remote():
    """Download UMD from Figshare on Modal and build the 256 cache."""
    import urllib.request
    import zipfile

    raw = MOUNT / "UMD"
    zip_path = MOUNT / "UMD.zip"
    if not (CACHE / "folds.json").exists():
        if not raw.exists():
            if not zip_path.exists():
                print("Downloading UMD.zip from Figshare...", flush=True)
                urllib.request.urlretrieve(
                    "https://ndownloader.figshare.com/files/44111183", zip_path
                )
            print("Extracting...", flush=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(MOUNT)
        _run(
            [
                "python",
                "-m",
                "fibroid.prepare",
                "--raw",
                str(raw if raw.exists() else MOUNT / "UMD"),
                "--cache",
                str(CACHE),
            ]
        )
        volume.commit()
    return {"cache": str(CACHE), "patients": len(list(CACHE.glob("*.npz")))}


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("wandb")],
    volumes={str(MOUNT): volume},
    timeout=10 * 60,
)
def validate():
    """Smoke-test CUDA, the dataset cache, one training step, and a Forge W&B run."""
    import os

    import torch
    import wandb

    from fibroid.data import SliceDataset, load_cases, load_meta, split_patients
    from fibroid.losses import DiceBCELoss
    from fibroid.model import UNet

    _configure_forge_wandb()
    report = {
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wandb_key": bool(os.environ.get("WANDB_API_KEY")),
        "wandb_base_url": os.environ.get("WANDB_BASE_URL"),
    }
    if not report["cuda"]:
        raise RuntimeError("CUDA not available on Modal worker")
    if not report["wandb_key"]:
        raise RuntimeError("WANDB_API_KEY missing from Modal secret 'wandb'")

    if not (CACHE / "folds.json").exists():
        raise FileNotFoundError(
            f"Cache missing at {CACHE}. Run upload_cache or prepare_remote first."
        )
    meta = load_meta(CACHE)
    train_ids, _, _ = split_patients(meta, fold=0)
    cases = load_cases(CACHE, train_ids[:4])
    loader_batch = next(iter(torch.utils.data.DataLoader(SliceDataset(cases, 1), batch_size=4)))
    images, targets = [t.cuda() for t in loader_batch]
    model = UNet(1, variant="dropout", dropout_p=0.1).cuda()
    loss = DiceBCELoss()(model(images), targets)
    loss.backward()
    report.update(
        patients=len(meta["patients"]),
        train_loss=float(loss.detach().cpu()),
        batch=list(images.shape),
        model="dropout",
    )

    init_kwargs = {
        "project": WANDB_PROJECT,
        "name": "modal-validate-dropout",
        "job_type": "validate",
        "config": report,
        "reinit": True,
    }
    try:
        init_kwargs["settings"] = wandb.Settings(base_url=os.environ["WANDB_BASE_URL"])
    except TypeError:
        pass
    run = wandb.init(**init_kwargs)
    wandb.log(
        {
            "epoch": 0,
            "train/loss": report["train_loss"],
            "val/loss": report["train_loss"],
            "val/dice": 0.0,
            "time/epoch_seconds": 0.0,
            "validate/cuda": 1,
        },
        step=0,
    )
    report["wandb_url"] = getattr(run, "url", None)
    run.finish()
    print(report, flush=True)
    return report


def _train_cmd(
    *,
    slices: int,
    model: str,
    folds: str,
    epochs: int,
    batch_size: int,
    limit_patients: int,
    dropout_p: float,
    dropblock_keep_prob: float,
    wandb_project: str,
    wandb_entity: str,
) -> tuple[list[str], Path]:
    import os

    _configure_forge_wandb(wandb_project, wandb_entity)
    out = RUNS / f"{model}_s{slices}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "fibroid.train",
        "--slices",
        str(slices),
        "--model",
        model,
        "--cache",
        str(CACHE),
        "--out",
        str(out),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--dropout-p",
        str(dropout_p),
        "--dropblock-keep-prob",
        str(dropblock_keep_prob),
        "--wandb",
        "--wandb-project",
        wandb_project,
        "--wandb-entity",
        wandb_entity or WANDB_ENTITY,
        "--wandb-base-url",
        os.environ["WANDB_BASE_URL"],
    ]
    if folds != "all":
        cmd += ["--folds", *[p for p in folds.replace(",", " ").split() if p]]
    if limit_patients:
        cmd += ["--limit-patients", str(limit_patients)]
    return cmd, out


@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb")],
    volumes={str(MOUNT): volume},
    timeout=60 * 60 * 12,
)
def train(
    slices: int = 1,
    model: str = "standard",
    folds: str = "all",
    epochs: int = 100,
    batch_size: int = 16,
    limit_patients: int = 0,
    dropout_p: float = 0.1,
    dropblock_keep_prob: float = 0.9,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
):
    """Launch fibroid.train on Modal and sync run artifacts back to the volume."""
    cmd, out = _train_cmd(
        slices=slices,
        model=model,
        folds=folds,
        epochs=epochs,
        batch_size=batch_size,
        limit_patients=limit_patients,
        dropout_p=dropout_p,
        dropblock_keep_prob=dropblock_keep_prob,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    try:
        _run(cmd)
    finally:
        volume.commit()
    entity = wandb_entity or WANDB_ENTITY
    return {
        "out": str(out),
        "wandb": f"https://forge.coreweave.com/wandb/{entity}/{wandb_project}",
    }


@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb")],
    volumes={str(MOUNT): volume},
    timeout=60 * 60 * 24,
)
def baselines(
    folds: str = "0",
    epochs: int = 100,
    batch_size: int = 16,
    slices: str = "1,3,5",
    models: str = "standard,dropout,dropblock",
    dropout_p: float = 0.1,
    dropblock_keep_prob: float = 0.9,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
    limit_patients: int = 0,
):
    """Run the standard vs dropout vs DropBlock baselines and log them to Forge."""
    slice_list = [int(s) for s in slices.replace(" ", "").split(",") if s]
    model_list = [m for m in models.replace(" ", "").split(",") if m]
    results = []
    try:
        for model_name in model_list:
            for n_slices in slice_list:
                print(f"=== baseline {model_name} slices={n_slices} ===", flush=True)
                cmd, out = _train_cmd(
                    slices=n_slices,
                    model=model_name,
                    folds=folds,
                    epochs=epochs,
                    batch_size=batch_size,
                    limit_patients=limit_patients,
                    dropout_p=dropout_p,
                    dropblock_keep_prob=dropblock_keep_prob,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                )
                _run(cmd)
                results.append({"model": model_name, "slices": n_slices, "out": str(out)})
    finally:
        volume.commit()
    entity = wandb_entity or WANDB_ENTITY
    return {
        "runs": results,
        "forge": f"https://forge.coreweave.com/wandb/{entity}/{wandb_project}",
    }


@app.local_entrypoint()
def main(action: str = "validate"):
    """modal run -m fibroid.modal_app -- --action validate|upload_cache|prepare_remote|baselines"""
    if action == "validate":
        print(validate.remote())
    elif action == "upload_cache":
        print(upload_cache.local())
    elif action == "prepare_remote":
        print(prepare_remote.remote())
    elif action == "baselines":
        print(baselines.remote())
    else:
        raise SystemExit(f"Unknown action {action}")
