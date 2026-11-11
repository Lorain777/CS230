"""Pool cross-validation test results and compare runs.

    python -m fibroid.summarize runs/unet_s1 runs/unet_s5

Small lesions are defined per fold from that fold's training patients, so the
definition never uses test data. Runs are compared at a matched false-positive
burden: each run's probability threshold is lowered from the top until its
pooled false positives per patient would exceed the budget.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_meta


def small_cutoff(meta, fold, exclude_irregular, min_lesion_ml, percentile):
    volumes = [
        volume
        for info in meta["patients"].values()
        if info["fold"] != fold and not (exclude_irregular and info["irregular_mask"])
        for volume in info["lesion_ml"]
        if volume >= min_lesion_ml
    ]
    return float(np.percentile(volumes, percentile))


def summarize(run_dir, meta, args):
    results = [json.loads(path.read_text()) for path in sorted(run_dir.glob("fold*/test.json"))]
    if not results:
        print(f"{run_dir}: no finished folds\n")
        return
    thresholds = np.array(results[0]["thresholds"])
    dice, cutoffs, all_scores, small_scores = [], [], [], []
    false_positives = np.zeros(len(thresholds))
    for result in results:
        cutoff = small_cutoff(meta, result["fold"], result["exclude_irregular"], args.min_lesion_ml, args.small_percentile)
        cutoffs.append(cutoff)
        for patient in result["patients"]:
            dice.append(patient["dice"])
            false_positives += np.array(patient["false_positives"])
        for lesion in result["lesions"]:
            if lesion["volume_ml"] < args.min_lesion_ml:
                continue
            all_scores.append(lesion["score"])
            if lesion["volume_ml"] <= cutoff:
                small_scores.append(lesion["score"])
    all_scores, small_scores = np.array(all_scores), np.array(small_scores)
    fp_per_patient = false_positives / len(dice)

    def recall(scores, threshold):
        return float((scores >= threshold).mean()) if scores.size else float("nan")

    print(f"{run_dir}  ({len(results)}/{meta['n_folds']} folds, {len(dice)} patients)")
    print(
        f"  lesions: {all_scores.size} of at least {args.min_lesion_ml} ml, {small_scores.size} small "
        f"(<= {min(cutoffs):.2f}-{max(cutoffs):.2f} ml, the training {args.small_percentile:.3g}th percentile per fold)"
    )
    print(f"  Dice per patient            {np.mean(dice):.3f} +/- {np.std(dice):.3f}")
    half = int(np.argmin(np.abs(thresholds - 0.5)))
    print(
        f"  at threshold 0.5            lesion recall {recall(all_scores, 0.5):.3f}, "
        f"small-lesion recall {recall(small_scores, 0.5):.3f}, "
        f"{fp_per_patient[half]:.2f} false positives per patient"
    )
    print("  small-lesion sensitivity at matched false-positive burden")
    for budget in args.fp_budgets:
        # Lower the threshold from the top until the budget would be exceeded.
        # Searching all thresholds instead would reward very low ones, where
        # predictions merge into one blob that touches a lesion and so counts
        # as zero false positives.
        over = np.flatnonzero(fp_per_patient > budget)
        first = over[-1] + 1 if over.size else 0
        if first < len(thresholds):
            threshold = thresholds[first]
            print(f"    <= {budget:g} FP/patient          {recall(small_scores, threshold):.3f}  (threshold {threshold:g})")
        else:
            print(f"    <= {budget:g} FP/patient          not reached at any threshold")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--cache", type=Path, default=Path("data/umd/cache256"))
    parser.add_argument("--min-lesion-ml", type=float, default=0.05, help="ignore reference components smaller than this")
    parser.add_argument("--small-percentile", type=float, default=100 / 3, help="training-set volume percentile that defines small")
    parser.add_argument("--fp-budgets", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    meta = load_meta(args.cache)
    for run_dir in args.runs:
        summarize(run_dir, meta, args)


if __name__ == "__main__":
    main()
