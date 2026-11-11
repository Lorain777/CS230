"""Volume Dice and lesion-level detection metrics.

A lesion is a 3D connected component of the mask (face connectivity, so
components join across adjacent slices only where they overlap in-plane).
"""

import math

import numpy as np
from scipy import ndimage

STRUCTURE = ndimage.generate_binary_structure(3, 1)
# Extra steps near 1 because a trained network's false positives are often confident.
THRESHOLDS = np.concatenate([np.round(np.arange(0.02, 1.0, 0.02), 2), [0.99, 0.995, 0.999]])


def dice_score(pred, truth):
    total = pred.sum() + truth.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(pred, truth).sum() / total)


def label_lesions(mask):
    labels, count = ndimage.label(mask, STRUCTURE)
    return labels, count


def lesion_scores(prob, truth, min_overlap=0.1):
    """One (voxel count, score) pair per reference lesion.

    A lesion counts as detected at threshold t when at least `min_overlap` of
    its voxels have probability >= t. That holds exactly when score >= t, so
    the score gives sensitivity at every threshold without re-thresholding.
    """
    labels, count = label_lesions(truth)
    records = []
    for index in range(1, count + 1):
        values = np.sort(prob[labels == index])[::-1]
        needed = max(1, math.ceil(min_overlap * values.size))
        records.append((int(values.size), float(values[needed - 1])))
    return records


def false_positive_counts(prob, truth, thresholds=THRESHOLDS):
    """Per threshold, the number of predicted components that touch no
    reference voxel. Components of any size count, including single voxels."""
    counts = []
    for threshold in thresholds:
        labels, count = label_lesions(prob >= threshold)
        touching = np.count_nonzero(np.unique(labels[truth]))
        counts.append(int(count - touching))
    return counts
