"""Modal app for UMD fibroid baselines with optional Weights & Biases logging.

Setup (once):
    modal volume create umd-fibroid
    modal run -m fibroid.modal_app::upload_cache     # uploads local data/umd/cache256
    # or: modal run -m fibroid.modal_app::prepare_remote  # download+prepare on Modal

Validate Modal + W&B:
    modal run -m fibroid.modal_app::validate

Train (examples):
    modal run -m fibroid.modal_app::train -- --slices 1 --model standard --folds 0
    modal run -m fibroid.modal_app::train -- --slices 5 --model dropblock
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
    """Smoke-test CUDA, the dataset cache, one training step, and a W&B run."""
    import os

    import torch
    import wandb

    from fibroid.data import SliceDataset, load_cases, load_meta, split_patients
    from fibroid.losses import DiceBCELoss
    from fibroid.model import UNet

    report = {
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wandb_key": bool(os.environ.get("WANDB_API_KEY")),
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
    model = UNet(1, variant="standard").cuda()
    loss = DiceBCELoss()(model(images), targets)
    loss.backward()
    report.update(patients=len(meta["patients"]), train_loss=float(loss.detach().cpu()), batch=list(images.shape))

    run = wandb.init(
        project=WANDB_PROJECT,
        name="modal-validate",
        job_type="validate",
        config=report,
        reinit=True,
    )
    wandb.log({"validate/train_loss": report["train_loss"], "validate/cuda": 1})
    report["wandb_url"] = run.get_url()
    run.finish()
    print(report, flush=True)
    return report


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
    wandb_project: str = WANDB_PROJECT,
):
    """Launch fibroid.train on Modal and sync run artifacts back to the volume."""
    import os

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
        "--wandb",
        "--wandb-project",
        wandb_project,
    ]
    if folds != "all":
        cmd += ["--folds", *[p for p in folds.replace(",", " ").split() if p]]
    if limit_patients:
        cmd += ["--limit-patients", str(limit_patients)]
    os.environ.setdefault("WANDB_PROJECT", wandb_project)
    try:
        _run(cmd)
    finally:
        volume.commit()
    return {"out": str(out)}


@app.local_entrypoint()
def main(action: str = "validate"):
    """modal run -m fibroid.modal_app -- --action validate|upload_cache|prepare_remote"""
    if action == "validate":
        print(validate.remote())
    elif action == "upload_cache":
        print(upload_cache.local())
    elif action == "prepare_remote":
        print(prepare_remote.remote())
    else:
        raise SystemExit(f"Unknown action {action}")
