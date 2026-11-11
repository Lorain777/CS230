"""UMD loading, the resized preprocessing cache, patient folds and slice datasets."""

import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MYOMA = 3  # UMD labels: 1 uterine wall, 2 uterine cavity, 3 myoma, 4 nabothian cyst
N_LABELS = 5
META_FILE = "folds.json"

_NIFTI_DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}


def read_nifti(path):
    """Return (voxel array in file order, voxel spacing in mm).

    The affine is ignored on purpose: 33 UMD masks carry no geometry, so image
    and mask are paired by voxel index. 34 masks are also plain NIfTI despite
    their .gz suffix, which is why this does not rely on the file extension.
    """
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    order = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    dim = struct.unpack(order + "8h", raw[40:56])
    code = struct.unpack(order + "h", raw[70:72])[0]
    pixdim = struct.unpack(order + "8f", raw[76:108])
    offset = int(struct.unpack(order + "f", raw[108:112])[0])
    shape = dim[1 : 1 + dim[0]]
    data = np.frombuffer(raw, np.dtype(order + _NIFTI_DTYPES[code]), int(np.prod(shape)), offset)
    return data.reshape(shape, order="F"), pixdim[1:4]


def _only(case_dir, pattern):
    matches = sorted(case_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"{case_dir.name}: expected one {pattern}, found {len(matches)}")
    return matches[0]


def preprocess_case(case_dir, size):
    """Normalise one patient and resize every slice to size x size."""
    image, spacing = read_nifti(_only(case_dir, "*_t2.nii.gz"))
    mask_path = _only(case_dir, "*_se[gq].nii.gz")
    mask, _ = read_nifti(mask_path)
    if image.shape != mask.shape or image.ndim != 3 or image.shape[0] != image.shape[1]:
        raise ValueError(f"{case_dir.name}: unexpected shapes {image.shape} / {mask.shape}")
    if mask.min() < 0 or mask.max() >= N_LABELS:
        raise ValueError(f"{case_dir.name}: mask values outside 0..{N_LABELS - 1}")

    image = image.astype(np.float32)
    low, high = np.percentile(image, [0.5, 99.5])
    image = np.clip(image, low, high)
    image = (image - image.mean()) / (image.std() + 1e-6)

    # (X, Y, Z) -> (Z, Y, X): one sagittal slice per leading index.
    image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 1, 0)))
    mask = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 1, 0)).astype(np.int64))
    image = F.interpolate(image[:, None], size=(size, size), mode="bilinear", antialias=True)[:, 0]
    # Area-average each class and take the plurality, so thin structures are
    # not dropped the way nearest-neighbour resizing drops them.
    one_hot = F.one_hot(mask, N_LABELS).permute(0, 3, 1, 2).float()
    mask = F.interpolate(one_hot, size=(size, size), mode="area").argmax(1)

    # Voxel volume on the resized grid, from the image header (mask headers are unreliable).
    original = one_hot.shape[-1]
    voxel_ml = (spacing[0] * original / size) * (spacing[1] * original / size) * spacing[2] / 1000.0
    return {
        "image": image.numpy().astype(np.float16),
        "mask": mask.numpy().astype(np.uint8),
        "voxel_ml": float(voxel_ml),
        "irregular_mask": mask_path.name.endswith("_seq.nii.gz"),
    }


def assign_folds(volumes, n_folds, seed):
    """Deal patients into folds in blocks sorted by myoma volume, so every fold
    sees the same range of lesion burden."""
    rng = np.random.default_rng(seed)
    ordered = sorted(volumes, key=lambda pid: (volumes[pid], pid))
    folds = {}
    for start in range(0, len(ordered), n_folds):
        block = ordered[start : start + n_folds]
        for pid, fold in zip(block, rng.permutation(n_folds)):
            folds[pid] = int(fold)
    return folds


def load_meta(cache_dir):
    return json.loads((Path(cache_dir) / META_FILE).read_text())


def split_patients(meta, fold, val_fraction=0.15, exclude_irregular=False):
    """Return (train, val, test) patient ids. Test is the held-out fold; val is
    drawn from the other folds and is used only for early stopping."""
    patients = {
        pid: info
        for pid, info in meta["patients"].items()
        if not (exclude_irregular and info["irregular_mask"])
    }
    test = sorted(pid for pid, info in patients.items() if info["fold"] == fold)
    rest = sorted(pid for pid, info in patients.items() if info["fold"] != fold)
    rest = [rest[i] for i in np.random.default_rng(fold).permutation(len(rest))]
    n_val = round(val_fraction * len(rest))
    return sorted(rest[n_val:]), sorted(rest[:n_val]), test


def load_cases(cache_dir, patient_ids):
    cases = []
    for pid in patient_ids:
        with np.load(Path(cache_dir) / f"{pid}.npz") as data:
            cases.append(
                {
                    "id": pid,
                    "image": torch.from_numpy(data["image"]),
                    "myoma": torch.from_numpy(data["mask"] == MYOMA),
                    "voxel_ml": float(data["voxel_ml"]),
                }
            )
    return cases


def context_indices(depth, n_slices):
    """(depth, n_slices) slice indices centred on each slice. Neighbours past
    the first or last slice repeat the edge slice."""
    offsets = torch.arange(n_slices) - n_slices // 2
    return (torch.arange(depth)[:, None] + offsets[None]).clamp(0, depth - 1)


class SliceDataset(Dataset):
    """One item per slice: n_slices adjacent slices as channels, and the myoma
    mask of the centre slice."""

    def __init__(self, cases, n_slices):
        if n_slices % 2 != 1:
            raise ValueError("n_slices must be odd")
        self.cases = cases
        self.context = [context_indices(case["image"].shape[0], n_slices) for case in cases]
        self.index = [(c, z) for c, case in enumerate(cases) for z in range(case["image"].shape[0])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        c, z = self.index[item]
        case = self.cases[c]
        return case["image"][self.context[c][z]].float(), case["myoma"][z][None].float()


def augment(images, targets, max_rotation=10.0, max_scale=0.1, max_shift=0.05):
    """Random small rotation, scaling and translation applied identically to a
    batch's images and masks, plus contrast, brightness and noise on images."""
    batch = images.shape[0]
    angle = (torch.rand(batch) * 2 - 1) * math.radians(max_rotation)
    zoom = 1 + (torch.rand(batch) * 2 - 1) * max_scale
    shift = (torch.rand(batch, 2) * 2 - 1) * 2 * max_shift
    cos, sin = torch.cos(angle) * zoom, torch.sin(angle) * zoom
    theta = torch.stack(
        [torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1
    )
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    targets = F.grid_sample(targets, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    targets = (targets > 0.5).float()

    contrast = 1 + (torch.rand(batch, 1, 1, 1) * 2 - 1) * 0.15
    brightness = (torch.rand(batch, 1, 1, 1) * 2 - 1) * 0.15
    images = images * contrast + brightness + torch.randn_like(images) * 0.03
    return images, targets
