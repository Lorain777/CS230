"""Build the resized cache and the patient folds.

    python -m fibroid.prepare
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import META_FILE, MYOMA, assign_folds, preprocess_case
from .metrics import label_lesions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/umd/UMD"))
    parser.add_argument("--cache", type=Path, default=Path("data/umd/cache256"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    case_dirs = sorted(path for path in args.raw.glob("UMD_*") if path.is_dir())
    if not case_dirs:
        raise SystemExit(f"No patient folders found in {args.raw}")
    args.cache.mkdir(parents=True, exist_ok=True)

    patients = {}
    for number, case_dir in enumerate(case_dirs, 1):
        case = preprocess_case(case_dir, args.size)
        np.savez_compressed(args.cache / f"{case_dir.name}.npz", **case)
        labels, count = label_lesions(case["mask"] == MYOMA)
        voxels = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        patients[case_dir.name] = {
            "n_slices": int(case["mask"].shape[0]),
            "myoma_ml": float(voxels.sum() * case["voxel_ml"]),
            "lesion_ml": [float(v * case["voxel_ml"]) for v in voxels],
            "irregular_mask": bool(case["irregular_mask"]),
        }
        if number % 25 == 0 or number == len(case_dirs):
            print(f"{number}/{len(case_dirs)} patients", flush=True)

    folds = assign_folds({pid: info["myoma_ml"] for pid, info in patients.items()}, args.folds, args.seed)
    for pid, fold in folds.items():
        patients[pid]["fold"] = fold
    meta = {"size": args.size, "n_folds": args.folds, "seed": args.seed, "patients": patients}
    (args.cache / META_FILE).write_text(json.dumps(meta, indent=1))
    print(f"Wrote {len(patients)} patients and {META_FILE} to {args.cache}")


if __name__ == "__main__":
    main()
