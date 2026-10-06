"""Fit phase-mixture models and matched K-means methods over a noise sweep."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PCMM.PCMMtorch import ACG, MACG, Normal, SingularWishart, Watson, WrappedNormal, Bingham
from PCMM.VMVM_PCMMtorch import VMVM
from PCMM.mixture_torch_loop import mixture_torch_loop
from PCMM.phase_coherence_kmeans import diametrical_clustering, grassmann_clustering, least_squares_sign_flip, quotient_torus_clustering, torus_clustering, weighted_grassmann_clustering, projective_hyperplane_clustering
from synthetic_phase_mixture_benchmark import (
    construct_representations,
    nmi_table,
    # sample_fixed,
    # sample_oscillatory,
)
# from rank_factor_torus_sampler import sample_fixed, sample_oscillatory
from fmri_phase_state_sampler import sample_fixed, sample_oscillatory

DEFAULT_RANK, N_CLUSTERS, SEED = 1, 2, 1
NOISE_LEVELS = np.linspace(0.1, 2*np.pi, 9)
# NOISE_LEVELS = np.linspace(0, 1, 9)
MAX_ITER, NUM_REPL, TOL, LEARNING_RATE = 10000, 3, 1e-7, 0.1
QUOTIENT_INITIALIZER = "qtc"

# These offsets keep a model's initialization unchanged when other model lines
# in fit_models() are commented out.
MODEL_SEED_OFFSETS = {
    "VMVM (quotient torus)": 0,
    "VMVM (ordinary torus)": 1,
    "VMVM (ordinary torus, uniform marginals)": 19,
    "Wrapped normal (ordinary torus)": 2,
    "Wrapped normal (quotient torus)": 3,
    "Complex Bingham": 4,
    "Complex Watson": 5,
    "Complex ACG": 6,
    "Real ACG": 7,
    "MACG": 8,
    "Singular Wishart": 9,
    "Complex Normal": 10,
    "Torus K-means (ordinary torus)": 11,
    "Torus K-means (quotient torus)": 12,
    "Complex diametrical K-means": 13,
    "Diametrical K-means": 14,
    "Grassmann K-means": 15,
    "Weighted Grassmann K-means": 16,
    "Least-squares K-means": 17,
    "Complex projective hyperplane K-means": 18,
}

def _raw_phase_initial_labels(theta, method, seed):
    """Cluster non-quotiented phases for a quotient-space mixture model."""
    np.random.seed(seed)
    kwargs = dict(
        K=N_CLUSTERS,
        max_iter=MAX_ITER,
        num_repl=NUM_REPL,
        tol=TOL,
        init="++",
        suppress_output=True,
    )
    if method == "qtc":
        clustering_data = theta
        clustering = quotient_torus_clustering
    elif method == "dc":
        clustering_data = np.exp(1j * theta) / np.sqrt(theta.shape[1])
        clustering = diametrical_clustering
    else:
        raise ValueError("quotient_initializer must be 'qtc' or 'dc'")
    return np.asarray(clustering(clustering_data, **kwargs)[1], dtype=int)


def _fit_labels(model, data, init, seed, noise, initial_labels=None):
    print(f"Fitting {model.__class__.__name__}")
    if "Watson" in model.__class__.__name__ and noise < 0.5:
        print("Skipping Watson fit due to low noise")
        return None
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_repl = NUM_REPL
    if initial_labels is not None:
        model.initialize(
            torch.from_numpy(np.ascontiguousarray(data)),
            posterior=torch.from_numpy(initial_labels),
        )
        init = "no"
        # The external clustering already used NUM_REPL. Reusing one initialized
        # model for several optimizer replications would not produce independent
        # fits.
        num_repl = 1
    _, posterior, _ = mixture_torch_loop(model, data=np.ascontiguousarray(data), tol=TOL, max_iter=MAX_ITER, num_repl=num_repl, init=init, LR=LEARNING_RATE, suppress_output=False, threads=1, decrease_lr_on_plateau=False, num_comparison=10)
    return np.asarray(posterior).argmax(axis=0)


def _score_labels(true_labels, estimated_labels):
    skipped = [name for name, labels in estimated_labels.items() if labels is None]
    fitted = {name: labels for name, labels in estimated_labels.items() if labels is not None}
    scores = nmi_table(true_labels, fitted)
    if skipped:
        scores = pd.concat(
            [
                scores,
                pd.DataFrame({"model": skipped, "NMI": np.nan}),
            ],
            ignore_index=True,
        )
    return scores.sort_values("NMI", ascending=False, na_position="last").reset_index(drop=True)


def _kmeans_labels(method, data, seed):
    print(f"Fitting {method.__name__}")
    np.random.seed(seed)
    kwargs = dict(K=N_CLUSTERS, max_iter=MAX_ITER, num_repl=NUM_REPL, tol=TOL, init="++")
    if method is least_squares_sign_flip: kwargs["init"] = "++"
    if method is least_squares_sign_flip: kwargs.pop("suppress_output", None)
    else: kwargs["suppress_output"] = True
    return np.asarray(method(data.copy(), **kwargs)[1])


def fit_models(theta, rank, seed, noise, quotient_initializer=QUOTIENT_INITIALIZER):
    if rank==6:
        rank_quotient_wn = 5
    else:
        rank_quotient_wn = rank
    X, p = construct_representations(theta), theta.shape[1]
    weighted_eigenspace = X["cosine_eigenvectors_2"]*np.sqrt(np.clip(X["cosine_eigenvalues_2"], 0, None))[:, None, :]
    wrapped = lambda dimension: WrappedNormal(p=dimension, rank=rank, K=N_CLUSTERS, winding_radius=1)
    wrapped_quotient = lambda dimension: WrappedNormal(p=dimension, rank=rank_quotient_wn, K=N_CLUSTERS, winding_radius=1)
    quotient_labels = _raw_phase_initial_labels(
        X["torus_raw"], quotient_initializer, seed
    )
    # Comment out any models that you do not want to rerun. Their rows in an
    # existing output CSV will be retained by _update_results().
    mixture_specs = (
        # ("VMVM (quotient torus)", VMVM(p=p-1, K=N_CLUSTERS), X["torus_quotient"], "no", quotient_labels),
        ("VMVM (ordinary torus)", VMVM(p=p, K=N_CLUSTERS), X["torus_raw"], "dc"),
        ("VMVM (ordinary torus, uniform marginals)", VMVM(p=p, K=N_CLUSTERS, oscillatory_data=True), X["torus_raw"], "dc"),
        ("Wrapped normal (ordinary torus)", wrapped(p), X["torus_raw"], "tc"), 
        ("Wrapped normal (quotient torus)", wrapped_quotient(p-1), X["torus_quotient"], "no", quotient_labels),
        ("Complex Bingham", Bingham(p=p, rank=rank, K=N_CLUSTERS, complex=True), X["complex_projective"], "dc"),
        ("Complex Watson", Watson(p=p, K=N_CLUSTERS, complex=True), X["complex_projective"], "dc"), 
        ("Complex ACG", ACG(p=p, rank=rank, K=N_CLUSTERS, complex=True), X["complex_projective"], "dc"),
        # ("Real Watson", Watson(p=p, K=N_CLUSTERS), X["cosine_leading_vector"], "dc"), 
        ("Real ACG", ACG(p=p, rank=rank, K=N_CLUSTERS), X["cosine_leading_vector"], "dc"),
        ("MACG", MACG(p=p, q=2, rank=rank, K=N_CLUSTERS), X["cosine_eigenvectors_2"], "gc"), 
        ("Singular Wishart", SingularWishart(p=p, q=2, rank=rank, K=N_CLUSTERS), weighted_eigenspace, "wgc"),
        # ("Complex Normal", Normal(p=p, rank=rank, K=N_CLUSTERS, complex=True), X["complex_projective"], "dc"),
    )
    kmeans_specs = (
        ("Torus K-means (ordinary torus)", torus_clustering, X["torus_raw"]),
        # ("Torus K-means (quotient torus)", torus_clustering, X["torus_quotient"]),
        ("Torus K-means (quotient torus)", quotient_torus_clustering, X["torus_raw"]),
        ("Complex diametrical K-means", diametrical_clustering, X["complex_projective"]), 
        ("Complex projective hyperplane K-means", projective_hyperplane_clustering, X["complex_projective"]), 
        ("Diametrical K-means", diametrical_clustering, X["cosine_leading_vector"]),
        ("Grassmann K-means", grassmann_clustering, X["cosine_eigenvectors_2"]), 
        ("Weighted Grassmann K-means", weighted_grassmann_clustering, weighted_eigenspace),
        ("Least-squares K-means", least_squares_sign_flip, X["cosine_leading_vector"]),
    )
    labels = {
        name: _fit_labels(
            model, data, init, seed + MODEL_SEED_OFFSETS[name], noise,
            initial_labels,
        )
        for name, model, data, init, *labels_arg in mixture_specs
        for initial_labels in [labels_arg[0] if labels_arg else None]
    }
    # labels = {}
    labels.update(
        {
            name: _kmeans_labels(
                method, data, seed + MODEL_SEED_OFFSETS[name]
            )
            for name, method, data in kmeans_specs
        }
    )
    return labels


def run_noise_sweep(
    dataset,
    noise_levels=NOISE_LEVELS,
    data_rank=DEFAULT_RANK,
    model_rank=6,
    quotient_initializer=QUOTIENT_INITIALIZER,
):
    sampler = {"oscillatory": sample_oscillatory, "fixed": sample_fixed}[dataset]
    tables = []
    for i, noise in enumerate(noise_levels):
        print(f"{dataset.capitalize()} data, noise {noise:.3f} ({i+1}/{len(noise_levels)})")
        theta, true_labels = sampler(noise=float(noise), rank=data_rank, seed=SEED)
        scores = _score_labels(
            true_labels,
            fit_models(
                theta, model_rank, SEED, noise=noise,
                quotient_initializer=quotient_initializer,
            ),
        )
        scores.insert(0, "dataset", dataset)
        scores.insert(0, "noise", noise) 
        scores.insert(0, "rank", data_rank)
        tables.append(scores)
    return pd.concat(tables, ignore_index=True)


def _update_results(output_path, new_results):
    """Replace rerun model/noise rows while retaining all other saved rows."""
    if not output_path.exists():
        return new_results

    existing = pd.read_csv(output_path)
    key_columns = ["rank", "noise", "dataset", "model"]
    missing = [column for column in key_columns if column not in existing.columns]
    if missing:
        raise ValueError(
            f"Cannot update {output_path}: missing key columns {missing}"
        )

    # CSV round trips can slightly alter floating-point noise values, so use a
    # rounded comparison key rather than comparing raw floats.
    def keys(frame):
        result = frame[key_columns].copy()
        result["noise"] = pd.to_numeric(result["noise"]).round(12)
        return pd.MultiIndex.from_frame(result)

    keep = ~keys(existing).isin(keys(new_results))
    combined = pd.concat([existing.loc[keep], new_results], ignore_index=True)
    return combined.sort_values(
        ["rank", "noise", "dataset", "NMI"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quotient-initializer",
        choices=("qtc", "dc"),
        default=QUOTIENT_INITIALIZER,
        help=(
            "Cluster non-quotiented phases with quotient-torus ('qtc') or "
            "complex diametrical ('dc') clustering before fitting quotient models."
        ),
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    datasets = ["oscillatory","fixed",]
    ranks = [1,3]
    for dataset in datasets:
        for rank in ranks:
            output_stem = (
                f"synthetic_phase_mixture_nmi_over_noise_{dataset}_rank={rank}"
            )
            output_path = Path("overview_paper/synthetic_results") / (
                output_stem + ".csv"
            )

            new_results = run_noise_sweep(
                dataset,
                data_rank=rank,
                model_rank=5,
                quotient_initializer=args.quotient_initializer,
            )
            results = _update_results(output_path, new_results)
            print("\n", results.to_string(index=False))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
