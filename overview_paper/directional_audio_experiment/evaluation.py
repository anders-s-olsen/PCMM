"""Activity recovery and learned-component spatial summaries.

NMI uses hard dominant-source labels, not soft probabilities. With overlap it
does not measure recovery of every active speaker or audio separation quality.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from alignment import best_assignment, correlation_matrix, frequency_average


def oracle_activity(source_energy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Isolated-source energy fractions: (S,F,T), mask (F,T) -> (F,S,T).

    The sum uses isolated source powers, not the coherent mixture power.
    This is an evaluation target, never an input to fitting or alignment.
    """
    energy = np.asarray(source_energy, float)
    valid = np.asarray(valid, bool)
    if energy.ndim != 3 or energy.shape[1:] != valid.shape:
        raise ValueError("Need isolated source energy (S,F,T) and mask (F,T).")
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("Invalid isolated-source energies.")
    total = energy.sum(axis=0)
    mask = valid & (total > 0)
    fractions = np.divide(energy, total[None], out=np.full_like(energy, np.nan),
                          where=mask[None])
    return np.moveaxis(fractions, 0, 1)


def evaluate_activity(aligned_p: np.ndarray, split: dict,
                      min_frequencies_per_frame: int) -> dict:
    """Score the requested across-frequency mean using exactly the same bins.

    A single NMI is computed over held-out time frames. Component IDs have one
    common meaning across frequencies, up to ONE irrelevant global permutation.
    There is no per-frequency oracle matching and no average of per-bin NMIs.
    """
    p = np.asarray(aligned_p, float)
    valid = np.asarray(split["valid"], bool)
    if p.shape[::2] != valid.shape:
        raise ValueError("Posterior and observation grids disagree.")
    if not np.array_equal(np.isfinite(p).all(axis=1), valid):
        raise ValueError("Every method must predict the same selected observations.")
    oracle = oracle_activity(split["source_energy"], valid)
    # An unlabelled bin cannot enter either side of the comparison.
    common = valid & np.isfinite(oracle).all(axis=1)
    p = np.where(common[:, None], p, np.nan)
    mean_p = frequency_average(p)
    mean_true = frequency_average(oracle)
    frame_valid = (common.sum(axis=0) >= min_frequencies_per_frame)
    frame_valid &= np.isfinite(mean_p).all(axis=0) & np.isfinite(mean_true).all(axis=0)
    if not frame_valid.any():
        raise ValueError("No held-out frames meet the common frequency-coverage requirement.")
    truth = np.full(valid.shape[1], -1, dtype=int)
    predicted = np.full_like(truth, -1)
    truth[frame_valid] = mean_true[:, frame_valid].argmax(axis=0)
    predicted[frame_valid] = mean_p[:, frame_valid].argmax(axis=0)
    nmi = normalized_mutual_info_score(truth[frame_valid], predicted[frame_valid],
                                        average_method="arithmetic")
    source_active = np.asarray(
        split.get("source_active", np.ones((mean_true.shape[0], valid.shape[1]), bool)),
        bool,
    )
    if source_active.shape != (mean_true.shape[0], valid.shape[1]):
        raise ValueError("source_active must have shape (sources,time).")
    active_count = source_active.sum(axis=0)
    by_overlap = {}
    for count in range(1, source_active.shape[0] + 1):
        subset = frame_valid & (active_count == count)
        by_overlap[str(count)] = {
            "n_frames": int(subset.sum()),
            "nmi": (
                float(normalized_mutual_info_score(
                    truth[subset], predicted[subset], average_method="arithmetic"
                ))
                if subset.any() else None
            ),
        }
    return {"nmi": float(nmi), "mean_posterior": mean_p,
            "mean_oracle": mean_true, "truth": truth, "predicted": predicted,
            "valid_frames": frame_valid, "n_frames": int(frame_valid.sum()),
            "active_source_count": active_count, "nmi_by_active_source_count": by_overlap,
            "single_talker_nmi": by_overlap["1"]["nmi"]}


def evaluate_shared_activity(shared_p: np.ndarray, split: dict,
                             min_frequencies_per_frame: int) -> dict:
    """Score one joint cross-frequency posterior per time frame.

    Unlike :func:`evaluate_activity`, this does not average separately fitted
    posterior probabilities. ``shared_p[k,t]`` is already the posterior of a
    product mixture whose component label is common to every frequency.
    """
    p = np.asarray(shared_p, float)
    valid = np.asarray(split["valid"], bool)
    if p.ndim != 2 or p.shape[1] != valid.shape[1]:
        raise ValueError("Shared posterior must have shape (K,time).")
    if (not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=0), 1., atol=1e-4)):
        raise ValueError("Shared posterior must be finite probabilities summing to one.")
    oracle = oracle_activity(split["source_energy"], valid)
    common = valid & np.isfinite(oracle).all(axis=1)
    mean_true = frequency_average(np.where(common[:, None], oracle, np.nan))
    frame_valid = common.sum(axis=0) >= min_frequencies_per_frame
    frame_valid &= np.isfinite(mean_true).all(axis=0)
    if not frame_valid.any():
        raise ValueError("No held-out frames meet the common frequency-coverage requirement.")

    truth = np.full(valid.shape[1], -1, dtype=int)
    predicted = np.full_like(truth, -1)
    truth[frame_valid] = mean_true[:, frame_valid].argmax(axis=0)
    predicted[frame_valid] = p[:, frame_valid].argmax(axis=0)
    nmi = normalized_mutual_info_score(
        truth[frame_valid], predicted[frame_valid], average_method="arithmetic"
    )
    source_active = np.asarray(
        split.get("source_active", np.ones((mean_true.shape[0], valid.shape[1]), bool)),
        bool,
    )
    if source_active.shape != (mean_true.shape[0], valid.shape[1]):
        raise ValueError("source_active must have shape (sources,time).")
    active_count = source_active.sum(axis=0)
    by_overlap = {}
    for count in range(1, source_active.shape[0] + 1):
        subset = frame_valid & (active_count == count)
        by_overlap[str(count)] = {
            "n_frames": int(subset.sum()),
            "nmi": (
                float(normalized_mutual_info_score(
                    truth[subset], predicted[subset], average_method="arithmetic"
                ))
                if subset.any() else None
            ),
        }
    return {
        "nmi": float(nmi),
        "mean_posterior": p,
        "mean_oracle": mean_true,
        "truth": truth,
        "predicted": predicted,
        "valid_frames": frame_valid,
        "n_frames": int(frame_valid.sum()),
        "active_source_count": active_count,
        "nmi_by_active_source_count": by_overlap,
        "single_talker_nmi": by_overlap["1"]["nmi"],
    }


def display_order_from_seat_scores(seat_scores: np.ndarray,
                                   candidate_seats: list[int] | np.ndarray,
                                   occupied_seats: list[int] | np.ndarray) -> np.ndarray:
    """Order K=3 learned components by their scores at the occupied seats.

    This uses only fitted-parameter steering scores and the declared candidate
    geometry. It does not use source activity labels or test observations.
    """
    scores = np.asarray(seat_scores, float)
    candidates = list(map(int, candidate_seats))
    occupied = list(map(int, occupied_seats))
    if scores.shape != (len(occupied), len(candidates)):
        raise ValueError("Seat-based display matching is defined for K=true source count.")
    try:
        columns = [candidates.index(seat) for seat in occupied]
    except ValueError as error:
        raise ValueError("Every occupied seat must be a candidate seat.") from error
    # Assignment rows are occupied-seat hypotheses and columns are components.
    return best_assignment(scores[:, columns].T)


def spatial_fingerprints(x: np.ndarray, aligned_p: np.ndarray,
                         steering: np.ndarray) -> np.ndarray:
    """Posterior-weighted spatial similarity, NOT fitted probability density.

    Inputs: raw training x (F,T,p), aligned probabilities (F,K,T), measured
    seat steering vectors (8,F,p). Output: (K,8), bounded in [0,1].

    At each frequency average |h_seat^H u|^2 within a component, with its
    posterior as weight; u and h have unit norm. Then average frequencies.
    This common readout works even when a fitter only returns responsibilities.
    The eight candidate-seat IRs are used ONLY here, after fitting/alignment.
    """
    x, p, h = np.asarray(x), np.asarray(aligned_p), np.asarray(steering)
    F, T, channels = x.shape
    if p.shape[0] != F or p.shape[2] != T or h.shape[1:] != (F, channels):
        raise ValueError("Incompatible shapes for spatial fingerprints.")
    result = np.full((F, p.shape[1], len(h)), np.nan)
    for f in range(F):
        mask = np.isfinite(p[f]).all(axis=0)
        if not mask.any():
            continue
        u = x[f, mask].astype(np.complex128)
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-15)
        hf = h[:, f].astype(np.complex128)
        hf /= np.maximum(np.linalg.norm(hf, axis=1, keepdims=True), 1e-15)
        affinity = abs(u @ hf.conj().T) ** 2  # (n,8)
        for k in range(p.shape[1]):
            w = p[f, k, mask]
            mass = w.sum()
            if mass > 1e-6:
                result[f, k] = w @ affinity / mass
    count = np.isfinite(result).sum(axis=0)
    scores = np.divide(np.nansum(result, axis=0), count,
                        out=np.full(result.shape[1:], np.nan), where=count > 0)
    return np.clip(scores, 0., 1.)


def display_order(train_mean_posterior: np.ndarray,
                  train_mean_oracle: np.ndarray, min_overlap: int = 20) -> np.ndarray:
    """ONE global source-color ordering, for the K=3 figure only.

    No frequency-dependent label information is used. This does not change
    NMI, likelihood, model selection, or the saved frequency permutations.
    """
    if train_mean_posterior.shape[0] != train_mean_oracle.shape[0]:
        raise ValueError("Display matching is defined only at the true K=3.")
    return best_assignment(correlation_matrix(train_mean_oracle,
                                               train_mean_posterior, min_overlap))
