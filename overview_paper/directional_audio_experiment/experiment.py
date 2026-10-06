#!/usr/bin/env python3
"""Run the paired directional-audio model-order experiment.

    python experiment.py prepare --config config.json
    python experiment.py run     --config config.json
    python experiment.py plot    --config config.json

The fitted loop is deliberately K -> model family -> data condition ->
frequency -> restart.  The two conditions share one clean speech scene and
differ only by the addition of measured McVAMPIRE driving noise.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import random
import warnings

import numpy as np
from scipy.special import logsumexp

from alignment import align_frequencies, apply_alignment
from data import CONDITION_NAMES, Config, load_prepared, prepare_dataset
from evaluation import display_order_from_seat_scores, evaluate_shared_activity


def _numpy(value, name: str) -> np.ndarray:
    """Convert a posterior/score tensor to a finite real NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real values.")
    return value.astype(np.float64, copy=False)


def _posterior(model, x, K: int) -> np.ndarray:
    p = _numpy(model.posterior(x), "posterior")
    if p.shape != (K, len(x)):
        raise ValueError(f"Expected posterior (K,n)={(K, len(x))}; got {p.shape}.")
    if (np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=0), 1., atol=1e-4)):
        raise ValueError("Return probabilities that sum to one over K, not logits.")
    return p


def _scores(model, x) -> np.ndarray:
    if not hasattr(model, "score_samples"):
        raise TypeError(
            "The adapter needs score_samples(x)->(n,) for held-out likelihood; "
            "posteriors alone are insufficient."
        )
    scores = _numpy(model.score_samples(x), "score_samples")
    if scores.shape != (len(x),):
        raise ValueError(f"Expected score_samples shape {(len(x),)}, got {scores.shape}.")
    return scores


def _parameter_snapshot(model) -> dict[str, np.ndarray]:
    """Detach a fitted model's numeric parameters for durable persistence."""
    if not hasattr(model, "export_params"):
        raise TypeError(
            "The adapter needs export_params() so every displayed component "
            "can be traced to its actual fitted parameters."
        )
    raw = model.export_params()
    if not isinstance(raw, dict) or not raw:
        raise ValueError("export_params() must return a nonempty dictionary.")
    result = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Fitted parameter names must be nonempty strings.")
        array = np.asarray(value)
        if array.dtype == object or not (
            np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
        ):
            raise ValueError(f"Fitted parameter {key!r} must be a numeric array.")
        if not np.isfinite(array).all():
            raise ValueError(f"Fitted parameter {key!r} contains nonfinite values.")
        result[key] = np.array(array, copy=True)
    return result


def _stack_parameter_snapshots(models: list) -> dict[str, np.ndarray]:
    """Stack like-shaped parameter snapshots along the frequency axis."""
    snapshots = [_parameter_snapshot(model) for model in models]
    keys = set(snapshots[0])
    if any(set(snapshot) != keys for snapshot in snapshots[1:]):
        raise ValueError("A family's fitted parameter keys changed across frequencies.")
    stacked = {}
    for key in sorted(keys):
        shapes = {snapshot[key].shape for snapshot in snapshots}
        if len(shapes) != 1:
            raise ValueError(
                f"Fitted parameter {key!r} changed shape across frequencies: {shapes}."
            )
        stacked[f"fitted_parameter__{key}"] = np.stack(
            [snapshot[key] for snapshot in snapshots], axis=0
        )
    return stacked


def _check_preparation(cfg: Config, metadata: dict) -> None:
    """Reject changes that would make cached observations inconsistent."""
    mutable = {
        "models", "component_counts", "fit_restarts", "device",
        "fit_tolerance", "fit_max_iterations", "fit_patience",
        "fit_learning_rate", "fit_threads", "alignment_min_overlap",
        "alignment_iterations", "alignment_initializations",
        "consensus_max_iterations", "consensus_label_tolerance",
        "consensus_min_cluster_frames", "figure_seconds",
        "export_audio", "output_dir", "minilibrimix_root", "mcvampire_root",
    }
    prepared_config = metadata.get("config", {})
    changes = [
        key for key, value in asdict(cfg).items()
        if key not in mutable and value != prepared_config.get(key)
    ]
    if changes:
        raise ValueError(
            f"Preparation settings changed ({', '.join(changes)}). "
            "Run prepare --overwrite first."
        )


def _selected_components(summary: list[dict], cfg: Config,
                         *, completed_only: bool) -> dict[str, dict[str, int]]:
    """Select K independently by held-out likelihood within family/condition."""
    expected = set(cfg.component_counts)
    selected: dict[str, dict[str, int]] = {}
    for condition in CONDITION_NAMES:
        condition_selected = {}
        for family in cfg.models:
            rows = sorted(
                [row for row in summary
                 if row["condition"] == condition and row["model"] == family],
                key=lambda row: row["K"],
            )
            if not rows or (completed_only and {row["K"] for row in rows} != expected):
                continue
            # max retains the earlier/smaller K on an exact tie. Never select
            # model order with the ground-truth NMI.
            condition_selected[family] = int(max(
                rows, key=lambda row: row["heldout_log_likelihood"]
            )["K"])
        if condition_selected:
            selected[condition] = condition_selected
    return selected


def _write_metrics(path: Path, summary: list[dict],
                   selected: dict[str, dict[str, int]]) -> None:
    """Atomically publish all completed rows for mid-run inspection."""
    if not summary:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[*summary[0], "selected_by_heldout_ll"]
        )
        writer.writeheader()
        for row in summary:
            chosen = selected.get(row["condition"], {}).get(row["model"])
            writer.writerow({
                **row,
                "selected_by_heldout_ll": (
                    "" if chosen is None else int(row["K"] == chosen)
                ),
            })
    temporary.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _write_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _display_steering(reference: dict, cfg: Config,
                      n_frequencies: int, n_channels: int) -> np.ndarray:
    """Select the four declared measured seat templates from the IR bank."""
    steering = np.asarray(reference["steering"])
    if steering.ndim != 3 or steering.shape[1:] != (n_frequencies, n_channels):
        raise ValueError(
            "reference steering must have shape (candidate_seat,frequency,microphone)."
        )
    seat_ids = np.asarray(
        reference.get("seat_ids", np.arange(1, steering.shape[0] + 1)), dtype=int
    )
    if seat_ids.shape != (steering.shape[0],) or len(set(seat_ids.tolist())) != len(seat_ids):
        raise ValueError("reference seat_ids must uniquely identify every steering row.")
    try:
        indices = [int(np.flatnonzero(seat_ids == seat)[0])
                   for seat in cfg.display_seat_ids]
    except (IndexError, TypeError) as error:
        raise ValueError("Every displayed seat needs a measured steering template.") from error
    return steering[indices]


def _validate_prepared(cfg: Config, conditions: dict,
                       reference: dict) -> tuple[np.ndarray, np.ndarray]:
    """Enforce the paired, ungated, continuously scheduled experiment contract."""
    if set(conditions) != set(CONDITION_NAMES):
        raise ValueError(
            f"Prepared conditions must be exactly {CONDITION_NAMES}; got {tuple(conditions)}."
        )
    canonical_hz = None
    shapes = {}
    for condition in CONDITION_NAMES:
        if set(conditions[condition]) != {"train", "test"}:
            raise ValueError(f"{condition} must contain train and test splits.")
        for split_name in ("train", "test"):
            split = conditions[condition][split_name]
            x = np.asarray(split["x"])
            valid = np.asarray(split["valid"], bool)
            if x.ndim != 3 or x.shape[:2] != valid.shape:
                raise ValueError("Prepared x/valid grids disagree.")
            if x.shape[2] != len(cfg.microphone_ids):
                raise ValueError("Prepared microphone dimension differs from config.")
            if not np.iscomplexobj(x) or not np.isfinite(x).all():
                raise ValueError("Prepared observations must be finite complex vectors.")
            if not valid.all():
                raise ValueError(
                    "This redesign trains and evaluates without amplitude gating; "
                    "every prepared observation must be valid."
                )
            speech_active = np.asarray(split["speech_active"], bool)
            if speech_active.shape != (x.shape[1],) or not speech_active.all():
                raise ValueError(
                    "The paired scene must contain at least one scheduled speaker in every frame."
                )
            source_active = np.asarray(split["source_active"], bool)
            if source_active.shape != (3, x.shape[1]):
                raise ValueError("source_active must have shape (3,time).")
            if not source_active.any(axis=0).all():
                raise ValueError("Prepared source activity contains silent frames.")
            hz = np.asarray(split["frequencies_hz"], float)
            if hz.shape != (x.shape[0],):
                raise ValueError("Prepared frequency coordinates disagree with x.")
            canonical_hz = hz if canonical_hz is None else canonical_hz
            if not np.array_equal(hz, canonical_hz):
                raise ValueError("Every split and condition must use the same frequencies.")
            shapes[(condition, split_name)] = x.shape
    for split_name in ("train", "test"):
        clean = conditions["no_driving_noise"][split_name]
        noisy = conditions["car_noise"][split_name]
        for key in ("source_energy", "source_active", "speech_active", "times",
                    "frequencies_hz"):
            if not np.array_equal(clean[key], noisy[key]):
                raise ValueError(
                    f"Paired {split_name} conditions differ in clean-scene field {key!r}."
                )
        if shapes[("no_driving_noise", split_name)] != shapes[("car_noise", split_name)]:
            raise ValueError("Paired observation arrays must have identical shapes.")
    assert canonical_hz is not None
    first = conditions["no_driving_noise"]["train"]["x"]
    steering = _display_steering(reference, cfg, len(canonical_hz), first.shape[2])
    return canonical_hz, steering


def _component_seat_scores(models: list, steering: np.ndarray,
                           K: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate shared-label components at four seats and pool frequencies."""
    raw = []
    for f, model in enumerate(models):
        if not hasattr(model, "seat_scores"):
            raise TypeError(
                "The adapter needs seat_scores(steering_f) for parameter-derived car maps."
            )
        score = _numpy(model.seat_scores(steering[:, f, :]), "seat_scores")
        if score.shape != (K, steering.shape[0]):
            raise ValueError(
                f"Expected component-by-seat scores {(K, steering.shape[0])}; "
                f"got {score.shape}."
            )
        if np.any(score < 0) or np.any(score.sum(axis=1) <= 0):
            raise ValueError("Each component needs nonnegative, nonzero seat scores.")
        raw.append(score / score.sum(axis=1, keepdims=True))
    raw = np.stack(raw, axis=0)  # (F,K,seat), already in shared label order
    # Equal-frequency geometric pooling corresponds to multiplying discrete
    # uniform-prior seat evidence without letting the number of bins change
    # its scale. Scores have already been normalized within component/bin.
    tiny = np.finfo(np.float64).tiny
    pooled_log = np.mean(np.log(np.maximum(raw, tiny)), axis=0)
    pooled_log -= pooled_log.max(axis=1, keepdims=True)
    pooled = np.exp(pooled_log)
    pooled /= pooled.sum(axis=1, keepdims=True)
    return raw, pooled


def _mixture_weights(model, K: int) -> np.ndarray:
    snapshot = _parameter_snapshot(model)
    if "pi" not in snapshot:
        raise ValueError("Every fitted mixture must export its mixing weights as 'pi'.")
    weights = np.asarray(snapshot["pi"], float)
    if (weights.shape != (K,) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., atol=1e-4)):
        raise ValueError("Exported mixture weights must be a positive normalized length-K vector.")
    return weights / weights.sum()


def _joint_product_posterior(
    posterior_per_frequency: np.ndarray,
    mixture_scores_per_frequency: np.ndarray,
    priors_per_frequency: np.ndarray,
    shared_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form the posterior of a shared-label product mixture.

    A frequency mixture supplies both its posterior and its normalized mixture
    score.  Their combination recovers each component log density without
    reaching into a model-family-specific implementation:
    ``log f_k(x) = log r_k(x) + log p(x) - log pi_k``.
    """
    p = np.asarray(posterior_per_frequency, float)
    scores = np.asarray(mixture_scores_per_frequency, float)
    priors = np.asarray(priors_per_frequency, float)
    if p.ndim != 3:
        raise ValueError("posterior_per_frequency must have shape (F,K,T).")
    F, K, T = p.shape
    if scores.shape != (F, T) or priors.shape != (F, K):
        raise ValueError("Mixture scores/priors disagree with posterior dimensions.")
    if (not np.isfinite(p).all() or not np.isfinite(scores).all()
            or not np.isfinite(priors).all() or np.any(priors <= 0)):
        raise ValueError("Product-mixture inputs must be finite with positive priors.")
    if not np.allclose(p.sum(axis=1), 1., atol=1e-4):
        raise ValueError("Per-frequency posteriors must sum to one.")
    priors = priors / priors.sum(axis=1, keepdims=True)
    if shared_weights is None:
        weights = priors.mean(axis=0)
    else:
        weights = np.asarray(shared_weights, float)
    if (weights.shape != (K,) or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError("Shared product-mixture weights must be positive and length K.")
    weights = weights / weights.sum()
    tiny = np.finfo(np.float64).tiny
    conditional_log_density = (
        np.log(np.maximum(p, tiny)) + scores[:, None, :]
        - np.log(priors[:, :, None])
    )
    component_log_density = (
        np.log(weights[:, None]) + conditional_log_density.sum(axis=0)
    )
    joint_score = logsumexp(component_log_density, axis=0)
    joint_posterior = np.exp(component_log_density - joint_score[None])
    return joint_posterior, joint_score, weights


def _enforce_minimum_cluster_size(
    labels: np.ndarray, posterior: np.ndarray, minimum: int,
) -> tuple[np.ndarray, int]:
    """Repair empty/tiny hard clusters using the least-cost train assignments."""
    labels = np.asarray(labels, int).copy()
    p = np.asarray(posterior, float)
    K, T = p.shape
    if labels.shape != (T,) or minimum < 1 or T < K * minimum:
        raise ValueError("Training set is too small for the requested consensus clusters.")
    counts = np.bincount(labels, minlength=K)
    log_p = np.log(np.maximum(p, np.finfo(np.float64).tiny))
    moved = 0
    for target in np.flatnonzero(counts < minimum):
        while counts[target] < minimum:
            candidates = np.flatnonzero(counts[labels] > minimum)
            if candidates.size == 0:
                raise ValueError("Cannot repair undersized consensus clusters.")
            current = labels[candidates]
            gain = log_p[target, candidates] - log_p[current, candidates]
            chosen = int(candidates[np.argmax(gain)])
            donor = int(labels[chosen])
            labels[chosen] = target
            counts[donor] -= 1
            counts[target] += 1
            moved += 1
    return labels, moved


def run_experiment(cfg: Config, *, fitter=None, partition_fitter=None,
                   overwrite: bool = False) -> Path:
    """Initialize per frequency, learn shared labels, and refit each density."""
    import torch
    from plotting import (ACTIVITY_FILENAME, ATLAS_FILENAMES, refresh_activity_figure,
                          refresh_component_atlas, refresh_performance_figures)

    if fitter is None:
        from fitter_adapter import fit_model
        fitter = partial(
            fit_model,
            fit_tolerance=cfg.fit_tolerance,
            fit_max_iterations=cfg.fit_max_iterations,
            fit_patience=cfg.fit_patience,
            fit_learning_rate=cfg.fit_learning_rate,
            fit_threads=cfg.fit_threads,
        )
    if partition_fitter is None:
        from fitter_adapter import fit_partitioned_model
        partition_fitter = partial(
            fit_partitioned_model,
            fit_tolerance=cfg.fit_tolerance,
            fit_max_iterations=cfg.fit_max_iterations,
            fit_patience=cfg.fit_patience,
            fit_learning_rate=cfg.fit_learning_rate,
            fit_threads=cfg.fit_threads,
        )
    cfg.validate()
    metadata, conditions, reference = load_prepared(cfg.output_dir)
    _check_preparation(cfg, metadata)
    hz, steering = _validate_prepared(cfg, conditions, reference)

    out = Path(cfg.output_dir) / "results"
    completion_marker = out / "results.json"
    if completion_marker.exists() and not overwrite:
        raise FileExistsError(
            "Completed results already exist; use run --overwrite to replace them deliberately."
        )
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    if completion_marker.exists():
        completion_marker.unlink()
    refresh_activity_figure(
        conditions, figures / ACTIVITY_FILENAME, source_positions=cfg.source_positions,
        train_seconds=cfg.train_seconds, test_seconds=cfg.test_seconds,
        figure_seconds=cfg.figure_seconds,
    )

    F = len(hz)
    _, train_frames, channels = conditions["no_driving_noise"]["train"]["x"].shape
    test_frames = conditions["no_driving_noise"]["test"]["x"].shape[1]
    summary: list[dict] = []
    records: list[dict] = []
    atlas: dict[str, dict[str, dict]] = {condition: {} for condition in CONDITION_NAMES}
    total_fits = len(set(cfg.component_counts)) * len(cfg.models) * len(CONDITION_NAMES)

    print(
        f"{len(metadata['speaker_ids'])} distinct fixed speakers; {channels} measured "
        f"microphones; {F} frequencies; {train_frames}/{test_frames} train/test frames.",
        flush=True,
    )
    print(
        "No amplitude gate: every train and held-out frame is retained, and every "
        "frame contains scheduled speech. K is the total unconstrained mixture order.",
        flush=True,
    )
    print(
        "Fit order: K -> model -> condition -> frequency -> restart. "
        f"Conditions: {', '.join(CONDITION_NAMES)}; families: {', '.join(cfg.models)}.",
        flush=True,
    )
    print(
        "Independent frequency mixtures initialize a train-only shared partition; "
        "frequency-specific one-component densities are then refit to that partition. "
        "Held-out prediction uses one product-mixture posterior per time frame.",
        flush=True,
    )

    for K in sorted(set(cfg.component_counts)):
        for family_index, family in enumerate(cfg.models):
            for condition in CONDITION_NAMES:
                train = conditions[condition]["train"]
                test = conditions[condition]["test"]
                T_train, T_test = train["x"].shape[1], test["x"].shape[1]
                counts_train = np.asarray(train["valid"], bool).sum(axis=1)
                counts_test = np.asarray(test["valid"], bool).sum(axis=1)
                initial_p_train = np.full((F, K, T_train), np.nan, dtype=np.float32)
                initial_p_test = np.full((F, K, T_test), np.nan, dtype=np.float32)
                initial_sample_ll_train = np.full((F, T_train), np.nan)
                initial_sample_ll_test = np.full((F, T_test), np.nan)
                restart_train_ll = np.full((F, cfg.fit_restarts), np.nan)
                chosen_restart = np.full(F, -1, dtype=int)
                initial_models = []
                train_tensors = []
                test_tensors = []

                for f, frequency_hz in enumerate(hz):
                    train_mask = np.asarray(train["valid"][f], bool)
                    test_mask = np.asarray(test["valid"][f], bool)
                    x_train = torch.as_tensor(
                        np.ascontiguousarray(train["x"][f, train_mask]),
                        dtype=torch.complex64, device=cfg.device,
                    )
                    x_test = torch.as_tensor(
                        np.ascontiguousarray(test["x"][f, test_mask]),
                        dtype=torch.complex64, device=cfg.device,
                    )
                    train_tensors.append(x_train)
                    test_tensors.append(x_test)

                    best_model, best_train_ll, best_restart = None, -np.inf, -1
                    restart_errors = []
                    for restart in range(cfg.fit_restarts):
                        # Intentionally omit condition: paired fits receive the
                        # same initialization seed, while their observations differ.
                        seed = int(np.random.SeedSequence(
                            [cfg.seed, family_index, K, f, restart]
                        ).generate_state(1)[0])
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        try:
                            model = fitter(
                                x_train=x_train, model_name=family,
                                n_components=K, frequency_hz=float(frequency_hz),
                                seed=seed,
                            )
                            with torch.no_grad():
                                score = float(_scores(model, x_train).mean())
                        except NotImplementedError:
                            raise
                        except Exception as error:
                            restart_errors.append(error)
                            warnings.warn(
                                f"Skipping failed initialization: condition={condition}, "
                                f"family={family}, K={K}, f={frequency_hz:.2f} Hz, "
                                f"restart={restart}: {error}"
                            )
                            continue
                        restart_train_ll[f, restart] = score
                        if score > best_train_ll:
                            best_model, best_train_ll, best_restart = model, score, restart
                    if best_model is None:
                        raise RuntimeError(
                            f"All {cfg.fit_restarts} initializations failed: "
                            f"condition={condition}, family={family}, K={K}, "
                            f"f={frequency_hz:.2f} Hz."
                        ) from restart_errors[-1]
                    initial_models.append(best_model)
                    chosen_restart[f] = best_restart
                    with torch.no_grad():
                        train_posterior = _posterior(best_model, x_train, K)
                        train_scores = _scores(best_model, x_train)
                        test_posterior = _posterior(best_model, x_test, K)
                        test_scores = _scores(best_model, x_test)
                    initial_p_train[f][:, train_mask] = train_posterior
                    initial_p_test[f][:, test_mask] = test_posterior
                    initial_sample_ll_train[f, train_mask] = train_scores
                    initial_sample_ll_test[f, test_mask] = test_scores
                    print(
                        f"  {family:17s} K={K} {condition:18s} "
                        f"{f + 1:02d}/{F} {frequency_hz:7.2f} Hz "
                        f"train/test={len(x_train)}/{len(x_test)}",
                        flush=True,
                    )

                # Initial matching and the shared hard partition use training
                # observations only. Held-out audio and source labels are absent.
                alignment = align_frequencies(
                    initial_p_train, hz, min_overlap=cfg.alignment_min_overlap,
                    max_iterations=cfg.alignment_iterations,
                    n_initializations=cfg.alignment_initializations,
                )
                initial_aligned_train = apply_alignment(
                    initial_p_train, alignment.permutation
                )
                initial_aligned_test = apply_alignment(
                    initial_p_test, alignment.permutation
                )
                initial_native_priors = np.stack(
                    [_mixture_weights(model, K) for model in initial_models], axis=0
                )
                initial_aligned_priors = np.take_along_axis(
                    initial_native_priors, alignment.permutation, axis=1
                )
                initial_shared_train, _, initial_shared_weights = (
                    _joint_product_posterior(
                        initial_aligned_train, initial_sample_ll_train,
                        initial_aligned_priors,
                    )
                )
                initial_shared_test, _, _ = _joint_product_posterior(
                    initial_aligned_test, initial_sample_ll_test,
                    initial_aligned_priors, initial_shared_weights,
                )
                minimum_cluster_frames = max(
                    cfg.consensus_min_cluster_frames, channels + 1
                )
                labels, rescued = _enforce_minimum_cluster_size(
                    initial_shared_train.argmax(axis=0), initial_shared_train,
                    minimum_cluster_frames,
                )
                initial_consensus_labels = labels.copy()
                consensus_changed_fraction = []
                consensus_rescued_frames = rescued
                consensus_converged = K == 1

                # K=1 already is a shared partition. For K>1, alternate hard
                # joint classification and per-frequency component refitting.
                final_models = initial_models
                final_p_train = initial_aligned_train
                final_p_test = initial_aligned_test
                final_sample_ll_train = initial_sample_ll_train
                final_sample_ll_test = initial_sample_ll_test
                final_priors = initial_aligned_priors
                shared_train = initial_shared_train
                shared_test = initial_shared_test
                train_joint_ll = _joint_product_posterior(
                    final_p_train, final_sample_ll_train, final_priors,
                    initial_shared_weights,
                )[1]
                test_joint_ll = _joint_product_posterior(
                    final_p_test, final_sample_ll_test, final_priors,
                    initial_shared_weights,
                )[1]
                labels_used_for_final_fit = labels.copy()

                if K > 1:
                    for iteration in range(cfg.consensus_max_iterations):
                        labels_used_for_final_fit = labels.copy()
                        shared_weights = np.bincount(
                            labels_used_for_final_fit, minlength=K
                        ).astype(float)
                        shared_weights /= shared_weights.sum()
                        refitted_models = []
                        for f, frequency_hz in enumerate(hz):
                            seed = int(np.random.SeedSequence([
                                cfg.seed, family_index, K, f, 10_000 + iteration,
                            ]).generate_state(1)[0])
                            try:
                                model = partition_fitter(
                                    x_train=train_tensors[f],
                                    assignments=labels_used_for_final_fit,
                                    mixture_weights=shared_weights,
                                    model_name=family,
                                    n_components=K,
                                    frequency_hz=float(frequency_hz),
                                    seed=seed,
                                )
                            except Exception as error:
                                raise RuntimeError(
                                    f"Consensus refit failed: condition={condition}, "
                                    f"family={family}, K={K}, f={frequency_hz:.2f} Hz, "
                                    f"iteration={iteration + 1}."
                                ) from error
                            refitted_models.append(model)

                        refit_p_train = np.empty((F, K, T_train), dtype=np.float32)
                        refit_p_test = np.empty((F, K, T_test), dtype=np.float32)
                        refit_ll_train = np.empty((F, T_train), dtype=np.float64)
                        refit_ll_test = np.empty((F, T_test), dtype=np.float64)
                        with torch.no_grad():
                            for f, model in enumerate(refitted_models):
                                refit_p_train[f] = _posterior(
                                    model, train_tensors[f], K
                                )
                                refit_p_test[f] = _posterior(
                                    model, test_tensors[f], K
                                )
                                refit_ll_train[f] = _scores(
                                    model, train_tensors[f]
                                )
                                refit_ll_test[f] = _scores(
                                    model, test_tensors[f]
                                )
                        refit_priors = np.stack(
                            [_mixture_weights(model, K)
                             for model in refitted_models], axis=0
                        )
                        shared_train, train_joint_ll, _ = _joint_product_posterior(
                            refit_p_train, refit_ll_train, refit_priors,
                            shared_weights,
                        )
                        shared_test, test_joint_ll, _ = _joint_product_posterior(
                            refit_p_test, refit_ll_test, refit_priors,
                            shared_weights,
                        )
                        new_labels, moved = _enforce_minimum_cluster_size(
                            shared_train.argmax(axis=0), shared_train,
                            minimum_cluster_frames,
                        )
                        changed = float(np.mean(new_labels != labels_used_for_final_fit))
                        consensus_changed_fraction.append(changed)
                        consensus_rescued_frames += moved
                        final_models = refitted_models
                        final_p_train, final_p_test = refit_p_train, refit_p_test
                        final_sample_ll_train = refit_ll_train
                        final_sample_ll_test = refit_ll_test
                        final_priors = refit_priors
                        print(
                            f"    consensus {iteration + 1}/{cfg.consensus_max_iterations}: "
                            f"changed={changed:.3%}; cluster sizes="
                            f"{np.bincount(new_labels, minlength=K).tolist()}",
                            flush=True,
                        )
                        if changed <= cfg.consensus_label_tolerance:
                            consensus_converged = True
                            break
                        labels = new_labels

                train_eval = evaluate_shared_activity(
                    shared_train, train, cfg.min_frequencies_per_frame
                )
                test_eval = evaluate_shared_activity(
                    shared_test, test, cfg.min_frequencies_per_frame
                )
                scored_truth = test_eval["truth"][test_eval["valid_frames"]]
                if np.unique(scored_truth).size < 3:
                    warnings.warn(
                        "Fewer than three dominant-source labels appear in scored held-out "
                        "frames; NMI cannot assess recovery of every physical source."
                    )

                train_mean_ll = float(np.mean(train_joint_ll))
                test_mean_ll = float(np.mean(test_joint_ll))
                seat_scores_per_frequency, component_seat_scores = _component_seat_scores(
                    final_models, steering, K
                )
                row = {
                    "condition": condition,
                    "model": family,
                    "K": K,
                    "train_log_likelihood": train_mean_ll,
                    "heldout_log_likelihood": test_mean_ll,
                    "heldout_nmi": test_eval["nmi"],
                    "heldout_single_talker_nmi": test_eval["single_talker_nmi"],
                    "heldout_frames_active_1": test_eval["nmi_by_active_source_count"]["1"]["n_frames"],
                    "heldout_frames_active_2": test_eval["nmi_by_active_source_count"]["2"]["n_frames"],
                    "heldout_frames_active_3": test_eval["nmi_by_active_source_count"]["3"]["n_frames"],
                    "evaluated_frames": test_eval["n_frames"],
                    "heldout_observations": int(test_eval["n_frames"]),
                    "joint_frequency_count": F,
                    "alignment_mean_correlation": alignment.mean_matched_correlation,
                    "consensus_iterations": len(consensus_changed_fraction),
                    "consensus_converged": consensus_converged,
                    "failed_initial_restarts": int(np.isnan(restart_train_ll).sum()),
                }
                summary.append(row)
                values: dict[str, np.ndarray] = {
                    "frequencies_hz": hz,
                    "initial_permutation": alignment.permutation,
                    "permutation": alignment.permutation,
                    "initial_train_posterior": initial_aligned_train,
                    "initial_test_posterior": initial_aligned_test,
                    "initial_train_shared_posterior": initial_shared_train,
                    "initial_test_shared_posterior": initial_shared_test,
                    "train_posterior_per_frequency": final_p_train,
                    "test_posterior_per_frequency": final_p_test,
                    "train_posterior": final_p_train,
                    "test_posterior": final_p_test,
                    "train_shared_posterior": shared_train,
                    "test_shared_posterior": shared_test,
                    "train_mean_posterior": train_eval["mean_posterior"],
                    "test_mean_posterior": test_eval["mean_posterior"],
                    "train_mean_oracle": train_eval["mean_oracle"],
                    "test_mean_oracle": test_eval["mean_oracle"],
                    "test_truth": test_eval["truth"],
                    "test_predicted": test_eval["predicted"],
                    "test_valid_frames": test_eval["valid_frames"],
                    "test_active_source_count": test_eval["active_source_count"],
                    "train_log_likelihood_per_frame": train_joint_ll,
                    "test_log_likelihood_per_frame": test_joint_ll,
                    "train_frequency_mixture_log_likelihood": final_sample_ll_train,
                    "test_frequency_mixture_log_likelihood": final_sample_ll_test,
                    "initial_train_frequency_mixture_log_likelihood": initial_sample_ll_train,
                    "initial_test_frequency_mixture_log_likelihood": initial_sample_ll_test,
                    "final_priors_per_frequency": final_priors,
                    "initial_priors_per_frequency": initial_aligned_priors,
                    "train_n_per_frequency": counts_train,
                    "test_n_per_frequency": counts_test,
                    "restart_train_ll": restart_train_ll,
                    "chosen_restart": chosen_restart,
                    "initial_consensus_labels": initial_consensus_labels,
                    "labels_used_for_final_fit": labels_used_for_final_fit,
                    "final_train_predicted_labels": shared_train.argmax(axis=0),
                    "consensus_changed_fraction": np.asarray(consensus_changed_fraction),
                    "consensus_converged": np.asarray(consensus_converged),
                    "consensus_rescued_frames": np.asarray(consensus_rescued_frames),
                    "component_seat_scores_per_frequency": seat_scores_per_frequency,
                    "component_seat_scores": component_seat_scores,
                    "display_seat_ids": np.asarray(cfg.display_seat_ids, dtype=int),
                    **_stack_parameter_snapshots(final_models),
                }
                if K == 3:
                    order = display_order_from_seat_scores(
                        component_seat_scores, cfg.display_seat_ids,
                        cfg.source_positions,
                    )
                    values["display_order"] = order
                    atlas[condition][family] = {
                        "seat_scores": component_seat_scores,
                        "display_order": order,
                    }

                filename = f"{condition}_{family}_K{K}.npz"
                _write_npz(out / filename, values)
                record = {**row, "file": filename}
                records.append(record)
                selected_so_far = _selected_components(
                    summary, cfg, completed_only=True
                )
                _write_metrics(out / "metrics.csv", summary, selected_so_far)
                _write_json(out / "progress.json", {
                    "schema_version": 3,
                    "status": "in_progress",
                    "preparation_id": metadata["preparation_id"],
                    "loop_order": ["K", "model", "condition", "frequency", "restart"],
                    "completed": len(records),
                    "total": total_fits,
                    "records": records,
                    "selected_K_for_completed_family_conditions": selected_so_far,
                })
                # Publish both standalone 1x2 metric figures after every
                # completed condition fit. Missing K/model points are expected.
                refresh_performance_figures(
                    records, figures, model_order=cfg.models,
                    conditions=CONDITION_NAMES, true_speakers=3,
                )
                if K == 3:
                    refresh_component_atlas(
                        atlas[condition], figures / ATLAS_FILENAMES[condition],
                        condition=condition, geometry=metadata["geometry"],
                        occupied_seats=cfg.source_positions,
                        candidate_seats=cfg.display_seat_ids,
                        microphone_ids=cfg.microphone_ids,
                        model_order=cfg.models,
                    )
                print(
                    f"  -> {condition}: held-out NMI={test_eval['nmi']:.3f}; "
                    f"joint mean LL/frame={test_mean_ll:.4f}; consensus "
                    f"iterations={len(consensus_changed_fraction)}; "
                    f"saved {len(records)}/{total_fits}",
                    flush=True,
                )

    selected = _selected_components(summary, cfg, completed_only=False)
    _write_metrics(out / "metrics.csv", summary, selected)
    payload = {
        "schema_version": 3,
        "preparation_id": metadata["preparation_id"],
        "config": asdict(cfg),
        "loop_order": ["K", "model", "condition", "frequency", "restart"],
        "records": records,
        "selected_K": selected,
        "selection": (
            "Maximum held-out mean log density separately within each family "
            "and noise condition; exact ties choose smaller K."
        ),
        "likelihood_figure": (
            "Within-family held-out mean joint log density per multfrequency frame, "
            "changed from K=1; absolute "
            "densities on Euclidean, projective, and toroidal spaces are not comparable."
        ),
        "nmi": (
            "One hard-label NMI from the held-out shared product-mixture posterior, "
            "against dominant isolated-source power. Initial frequency matching, "
            "consensus partitioning, and component refitting use training data only."
        ),
        "component_maps": (
            "Final consensus-refit component steering scores at four measured seats, "
            "pooled geometrically over frequency; contours are explicitly schematic "
            "interpolation."
        ),
        "scope": (
            "The held-out split is also used for K selection, so the selected-model "
            "likelihood is not an untouched final-test estimate."
        ),
    }
    _write_json(out / "results.json", payload)
    _write_json(out / "progress.json", {
        "schema_version": 3,
        "status": "complete",
        "preparation_id": metadata["preparation_id"],
        "loop_order": ["K", "model", "condition", "frequency", "restart"],
        "completed": len(records),
        "total": total_fits,
        "records": records,
        "selected_K_for_completed_family_conditions": selected,
    })
    return out


def regenerate_figures(cfg: Config) -> dict[str, Path]:
    """Recreate current performance figures and any available K=3 atlases."""
    from plotting import (ACTIVITY_FILENAME, ATLAS_FILENAMES, refresh_activity_figure,
                          refresh_component_atlas, refresh_performance_figures)

    out = Path(cfg.output_dir) / "results"
    source = out / "results.json"
    if not source.exists():
        source = out / "progress.json"
    if not source.exists():
        raise FileNotFoundError("No progress.json or results.json is available to plot.")
    payload = json.loads(source.read_text())
    records = payload.get("records", [])
    figures = out / "figures"
    paths = refresh_performance_figures(
        records, figures, model_order=cfg.models,
        conditions=CONDITION_NAMES, true_speakers=3,
    )
    metadata, conditions, _ = load_prepared(cfg.output_dir)
    paths["activity"] = refresh_activity_figure(
        conditions, figures / ACTIVITY_FILENAME, source_positions=cfg.source_positions,
        train_seconds=cfg.train_seconds, test_seconds=cfg.test_seconds,
        figure_seconds=cfg.figure_seconds,
    )
    scores_by_condition: dict[str, dict[str, dict]] = {
        condition: {} for condition in CONDITION_NAMES
    }
    for record in records:
        if int(record["K"]) != 3:
            continue
        with np.load(out / record["file"], allow_pickle=False) as values:
            if "component_seat_scores" not in values or "display_order" not in values:
                continue
            scores_by_condition[record["condition"]][record["model"]] = {
                "seat_scores": values["component_seat_scores"],
                "display_order": values["display_order"],
            }
    for condition, scores in scores_by_condition.items():
        if not scores:
            continue
        paths[f"components_{condition}"] = refresh_component_atlas(
            scores, figures / ATLAS_FILENAMES[condition], condition=condition,
            geometry=metadata["geometry"], occupied_seats=cfg.source_positions,
            candidate_seats=cfg.display_seat_ids,
            microphone_ids=cfg.microphone_ids, model_order=cfg.models,
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "plot"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name != "plot":
            command.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = Config.from_json(args.config)
    if args.command == "prepare":
        print(f"Prepared: {prepare_dataset(cfg, overwrite=args.overwrite)}")
    elif args.command == "run":
        print(f"Results: {run_experiment(cfg, overwrite=args.overwrite)}")
    else:
        paths = regenerate_figures(cfg)
        print("Figures: " + ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
