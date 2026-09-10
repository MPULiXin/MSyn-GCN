from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .data import Prescription


def ranked_indices(scores: np.ndarray, max_k: int) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got {scores.shape}")
    if max_k <= 0 or max_k > scores.shape[1]:
        raise ValueError(f"Invalid max_k={max_k} for {scores.shape[1]} herbs")
    return np.argsort(-scores, axis=1, kind="stable")[:, :max_k]


def standard_metrics(
    scores: np.ndarray,
    ground_truth: Iterable[Iterable[int]],
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """Precision/Recall plus the legacy hit-normalized NDCG used by the old model.

    Experimental compatibility note: unlike paper-standard NDCG, the IDCG below
    is built from the number of relevant herbs actually retrieved inside Top-K.
    This produces systematically higher NDCG values and is retained only for
    reproducing the historical evaluation convention.
    """
    cutoffs = tuple(sorted(int(value) for value in cutoffs))
    targets = [set(int(value) for value in row) for row in ground_truth]
    if len(targets) != len(scores):
        raise ValueError("scores and ground_truth must have the same number of rows")
    rankings = ranked_indices(scores, cutoffs[-1])
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
        precisions = []
        recalls = []
        ndcgs = []
        for ranking, target in zip(rankings[:, :cutoff], targets):
            hits = np.asarray([item in target for item in ranking], dtype=np.float64)
            hit_count = float(hits.sum())
            precisions.append(hit_count / cutoff)
            recalls.append(hit_count / max(1, len(target)))
            # Legacy convention: normalize only by relevant herbs retrieved in
            # Top-K, rather than by all relevant herbs available in the target.
            ideal_count = int(hit_count)
            ideal = float(discounts[:ideal_count].sum())
            ndcgs.append(float((hits * discounts).sum()) / ideal if ideal else 0.0)
        metrics[f"precision@{cutoff}"] = float(np.mean(precisions))
        metrics[f"recall@{cutoff}"] = float(np.mean(recalls))
        metrics[f"ndcg@{cutoff}"] = float(np.mean(ndcgs))
    return metrics


def syndrome_sparsity(
    eight: np.ndarray,
    zangfu: np.ndarray,
    threshold: float = 1e-4,
) -> dict[str, float]:
    def effective(values: np.ndarray) -> float:
        safe = np.clip(values, 1e-12, None)
        return float(np.exp(-(safe * np.log(safe)).sum(axis=1)).mean())

    return {
        "eight_active": float((eight > threshold).sum(axis=1).mean()),
        "eight_effective": effective(eight),
        "zangfu_active": float((zangfu > threshold).sum(axis=1).mean()),
        "zangfu_effective": effective(zangfu),
    }


def ambiguity_aware_metrics(
    scores: np.ndarray,
    prescriptions: list[Prescription],
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    if len(scores) != len(prescriptions):
        raise ValueError("scores and prescriptions must have the same number of rows")
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(prescriptions):
        groups[tuple(sorted(row.symptoms))].append(index)
    cutoffs = tuple(sorted(int(value) for value in cutoffs))
    rankings = ranked_indices(scores, cutoffs[-1])
    output: dict[str, float] = {}

    for cutoff in cutoffs:
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
        per_group = []
        weights = []
        for indices in groups.values():
            ranking = rankings[indices[0], :cutoff]
            targets = [set(prescriptions[index].herbs) for index in indices]
            best_precision = 0.0
            best_ndcg = 0.0
            for target in targets:
                hits = np.asarray([item in target for item in ranking], dtype=np.float64)
                best_precision = max(best_precision, float(hits.sum()) / cutoff)
                ideal_count = min(len(target), cutoff)
                ideal = float(discounts[:ideal_count].sum())
                ndcg = float((hits * discounts).sum()) / ideal if ideal else 0.0
                best_ndcg = max(best_ndcg, ndcg)
            union = set().union(*targets)
            union_recall = sum(item in union for item in ranking) / max(1, len(union))
            per_group.append((best_precision, best_ndcg, union_recall))
            weights.append(len(indices))
        values = np.asarray(per_group, dtype=np.float64)
        group_weights = np.asarray(weights, dtype=np.float64)
        labels = ("bm_precision", "bm_ndcg", "union_recall")
        for column, label in enumerate(labels):
            output[f"{label}@{cutoff}_unweighted"] = float(values[:, column].mean())
            output[f"{label}@{cutoff}_weighted"] = float(
                np.average(values[:, column], weights=group_weights)
            )
    return output
