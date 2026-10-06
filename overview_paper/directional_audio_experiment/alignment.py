"""Training-only frequency permutation alignment; no source labels are used.

Reference: Sawada, Araki & Makino (2011), Sec. IV-D, Eqs. 25--28, 31,
the alignment method cited by Ito, Araki & Nakatani (EUSIPCO 2016).
We implement the single-centroid global stage and adjacent/harmonic local
stage. Hungarian assignment exactly maximizes the paper's permutation score:
diagonal-minus-off-diagonal = 2 * trace - sum, with invariant total sum.

Explicit adaptations: missing low-amplitude bins use pairwise complete
correlations; neighbors are selected from the available sparse frequency
grid; multiple training-only initial anchors improve initialization. This is
not claimed to reproduce the authors' unpublished implementation bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Alignment:
    # aligned[f, k, t] = original[f, permutation[f, k], t]
    permutation: np.ndarray
    objective: float
    mean_matched_correlation: float
    iterations: int


def correlation_matrix(reference: np.ndarray, candidate: np.ndarray,
                       min_overlap: int = 20) -> np.ndarray:
    """Pearson correlations, rows=reference components, columns=candidate.

    NaNs denote unavailable observations, not zero activity. Constant traces
    or insufficient joint observations provide no permutation evidence (0).
    """
    a, b = np.asarray(reference, float), np.asarray(candidate, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Activity traces must have shapes (K,T) and (L,T).")
    out = np.zeros((len(a), len(b)), dtype=float)
    for i in range(len(a)):
        for j in range(len(b)):
            valid = np.isfinite(a[i]) & np.isfinite(b[j])
            if valid.sum() < min_overlap:
                continue
            x, y = a[i, valid], b[j, valid]
            x, y = x - x.mean(), y - y.mean()
            scale = np.linalg.norm(x) * np.linalg.norm(y)
            if scale > 1e-12:
                out[i, j] = np.clip(np.dot(x, y) / scale, -1, 1)
    return out


def best_assignment(score: np.ndarray, current: np.ndarray | None = None) -> np.ndarray:
    """Return candidate row indices in reference order, retaining ties."""
    score = np.asarray(score, float)
    if score.ndim != 2 or score.shape[0] != score.shape[1] or not np.isfinite(score).all():
        raise ValueError("Expected a finite square assignment matrix.")
    rows, cols = linear_sum_assignment(-score)
    permutation = cols[np.argsort(rows)]
    if current is not None:
        if score[np.arange(len(score)), permutation].sum() <= (
            score[np.arange(len(score)), current].sum() + 1e-12
        ):
            return np.asarray(current).copy()
    return permutation


def apply_alignment(posteriors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    p = np.asarray(posteriors)
    order = np.asarray(permutation)
    if p.ndim != 3 or order.shape != p.shape[:2]:
        raise ValueError("Expected posteriors (F,K,T) and permutation (F,K).")
    if not np.all(np.sort(order, axis=1) == np.arange(p.shape[1])):
        raise ValueError("Every frequency must have a one-to-one permutation.")
    return np.take_along_axis(p, order[:, :, None], axis=1)


def _standardize(p: np.ndarray) -> np.ndarray:
    valid = np.isfinite(p)
    n = valid.sum(axis=-1, keepdims=True)
    mean = np.divide(np.nansum(p, axis=-1, keepdims=True), n,
                     out=np.zeros_like(n, dtype=float), where=n > 0)
    centered = np.where(valid, p - mean, 0)
    variance = np.divide((centered ** 2).sum(axis=-1, keepdims=True), n,
                         out=np.zeros_like(mean), where=n > 0)
    result = centered / np.maximum(np.sqrt(variance), 1e-12)
    return np.where(valid, result, np.nan)


def _mean_available(p: np.ndarray, axis: int = 0) -> np.ndarray:
    count = np.isfinite(p).sum(axis=axis)
    return np.divide(np.nansum(p, axis=axis), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def _neighbor_sets(frequencies_hz: np.ndarray) -> list[list[int]]:
    """Sparse-grid version of adjacent + half/double-frequency neighbors."""
    hz = np.asarray(frequencies_hz, float)
    n = len(hz)
    spacing = float(np.median(np.diff(hz))) if n > 1 else 0.
    adjacency = [set() for _ in hz]
    for f in range(n):
        candidates = set(range(max(0, f - 3), min(n, f + 4))) - {f}
        for target in [hz[f] / 2, hz[f] * 2]:
            if hz[0] <= target <= hz[-1]:
                center = int(np.argmin(abs(hz - target)))
                for g in range(max(0, center - 1), min(n, center + 2)):
                    if g != f and abs(hz[g] - target) <= 1.5 * spacing:
                        candidates.add(g)
        for g in candidates:
            adjacency[f].add(g)
            adjacency[g].add(f)  # symmetric objective for local updates
    return [sorted(s) for s in adjacency]


def align_frequencies(train_posteriors: np.ndarray, frequencies_hz: np.ndarray,
                      *, min_overlap: int = 20, max_iterations: int = 30,
                      n_initializations: int = 3) -> Alignment:
    """Learn permutations ONLY on aligned-in-time training posterior traces.

    Input is (F,K,T); gated time-frequency observations must be NaN in every
    component. Save the result and apply it unchanged to held-out posteriors.
    Train/test component order must come from the same frozen fitted model.
    """
    p = np.asarray(train_posteriors, float)
    hz = np.asarray(frequencies_hz, float)
    if p.ndim != 3 or hz.shape != (p.shape[0],) or not np.all(np.diff(hz) > 0):
        raise ValueError("Need (F,K,T) posteriors and sorted, unique frequencies.")
    F, K, _ = p.shape
    if F < 1 or K < 1:
        raise ValueError("Empty frequency/component dimension.")
    valid = np.isfinite(p)
    if not np.all(valid == valid[:, :1, :]):
        raise ValueError("A missing observation must be missing in ALL components.")
    if np.any(p[valid] < -1e-7) or np.any(p[valid] > 1 + 1e-7):
        raise ValueError("Expected probabilities, not logits/log probabilities.")
    sums = np.nansum(p, axis=1)
    if not np.allclose(sums[valid[:, 0]], 1., atol=1e-4):
        raise ValueError("Posterior probabilities must sum to one over K.")
    identity = np.tile(np.arange(K), (F, 1))
    if K == 1 or F == 1:
        return Alignment(identity, 0., 0., 0)

    z = _standardize(p)
    # Q[f,g,i,j] compares original component i at f and j at g.
    Q = np.empty((F, F, K, K), dtype=float)
    for f in range(F):
        for g in range(f, F):
            Q[f, g] = correlation_matrix(p[f], p[g], min_overlap)
            Q[g, f] = Q[f, g].T

    def quality(order: np.ndarray, neighbors=None) -> float:
        values = []
        for f in range(F):
            targets = range(f + 1, F) if neighbors is None else [g for g in neighbors[f] if g > f]
            for g in targets:
                matrix = Q[f, g][np.ix_(order[f], order[g])]
                values.append(2 * np.trace(matrix) - matrix.sum())
        return float(np.mean(values)) if values else 0.

    strength = np.zeros(F)
    for f in range(F):
        for g in range(F):
            if g != f:
                order = best_assignment(Q[f, g])
                strength[f] += Q[f, g][np.arange(K), order].mean()
    anchors = np.argsort(-strength, kind="stable")[:max(1, min(n_initializations, F))]
    neighbors = _neighbor_sets(hz)
    candidates = []
    total_iterations = 0
    for anchor in anchors:
        order = identity.copy()
        for f in range(F):
            if f != anchor:
                order[f] = best_assignment(Q[anchor, f])
        # Sawada Eq. 27 (mean standardized traces) and Eq. 28 (matching).
        best_order, best_value = order.copy(), quality(order)
        seen = set()
        for _ in range(max_iterations):
            key = order.tobytes()
            if key in seen:
                break
            seen.add(key)
            centroid = _mean_available(apply_alignment(z, order), axis=0)
            new_order = np.stack([
                best_assignment(correlation_matrix(centroid, z[f], min_overlap), order[f])
                for f in range(F)
            ])
            total_iterations += 1
            value = quality(new_order)
            if value > best_value + 1e-12:
                best_order, best_value = new_order.copy(), value
            if np.array_equal(new_order, order):
                break
            order = new_order
        order = best_order
        # Sawada Eq. 31: improve agreement with neighboring/harmonic bins.
        for _ in range(max_iterations):
            changed = False
            for f in range(F):
                scores = np.zeros((K, K))  # common component x original row
                for g in neighbors[f]:
                    scores += Q[g, f][order[g], :]
                new = best_assignment(scores, order[f])
                changed |= not np.array_equal(new, order[f])
                order[f] = new
            total_iterations += 1
            if not changed:
                break
        candidates.append((quality(order, neighbors), order.copy()))
    _, order = max(candidates, key=lambda item: item[0])
    matches = [Q[f, g][order[f], order[g]].mean()
               for f in range(F) for g in neighbors[f] if g > f]
    mean_corr = float(np.mean(matches)) if matches else 0.
    if mean_corr < .1:
        warnings.warn("Weak cross-frequency activity correlation: inspect alignment before interpreting source identity.")
    return Alignment(order, quality(order, neighbors), mean_corr, total_iterations)


def frequency_average(posteriors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Average aligned probabilities across available frequencies: (F,K,T)->(K,T).

    Default equal weights match the requested posterior average. This is an
    activity summary, NOT a joint Bayesian posterior given all frequencies.
    """
    p = np.asarray(posteriors, float)
    valid = np.isfinite(p).all(axis=1)
    w = np.ones_like(valid, dtype=float) if weights is None else np.asarray(weights, float)
    if w.shape != valid.shape or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("Weights must be finite, nonnegative and shaped (F,T).")
    w = np.where(valid, w, 0.)
    denominator = w.sum(axis=0)
    result = np.divide(np.nansum(p * w[:, None, :], axis=0), denominator[None, :],
                       out=np.full(p.shape[1:], np.nan), where=denominator[None, :] > 0)
    return result
