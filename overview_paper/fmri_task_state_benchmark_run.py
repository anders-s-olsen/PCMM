"""Fit K=7 PCMM models to HCP task fMRI and score state/task overlap.

This is deliberately self-contained: it reads the existing HDF5 dataset,
constructs task labels, fits models through PCMM, and calculates subject-level
NMI without importing any of the helpers in ``paper/``.

Rank-dependent models are fitted in ascending rank order.  The parameters from
one rank initialize the next rank, using PCMM's low-rank SVD expansion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd
import torch

from PCMM.PCMMtorch import ACG, Bingham, MACG, Normal, SingularWishart, Watson, WrappedNormal
from PCMM.VMVM_PCMMtorch import VMVM
from PCMM.mixture_torch_loop import mixture_torch_loop
from PCMM.phase_coherence_kmeans import (
    diametrical_clustering,
    grassmann_clustering,
    least_squares_sign_flip,
    quotient_torus_clustering,
    weighted_grassmann_clustering,
    projective_hyperplane_clustering,
)


K = 7
P = 116
TASKS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
TASK_LENGTHS = np.array([176, 253, 316, 284, 232, 274, 405])
PRESTIM_LENGTHS = np.array([15, 11, 0, 11, 11, 11, 11])
FIT_TASK_LENGTHS = TASK_LENGTHS - PRESTIM_LENGTHS
POINTS_PER_SUBJECT = int(FIT_TASK_LENGTHS.sum())
TASK_VECTOR_DIR = Path("paper/data/task_indices")
DEFAULT_RANKS = (1, 5, 10, 25, 50)
DEFAULT_DATA = Path("paper/data/processed/concatenated_datasets/all_tasksfMRI_SchaeferTian116_GSR.h5")
DEFAULT_OUTPUT = Path("overview_paper/fmri_task_results/fmri_task_state_nmi.csv")
RESULT_COLUMNS = ("model", "rank", "set", "subject", "NMI")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dataset: str
    init: str
    ranked: bool
    constructor: Callable[[int, dict | None], object] | None = None
    clustering: Callable | None = None


MODEL_SPECS = (
    # ModelSpec("Complex Watson", "U_complex", "dc", False, lambda rank, params: Watson(p=P, K=K, complex=True, params=params)),
    # ModelSpec("Complex ACG", "U_complex", "dc", True, lambda rank, params: ACG(p=P, rank=rank, K=K, complex=True, params=params)),
    # ModelSpec("Complex Bingham", "U_complex", "dc", True, lambda rank, params: Bingham(p=P, rank=rank, K=K, complex=True, params=params)),
    # ModelSpec("Complex diametrical K-means", "U_complex", "++", False, clustering=diametrical_clustering),
    # ModelSpec("Complex projective hyperplane K-means", "U_complex", "++", False, clustering=projective_hyperplane_clustering),
    # ModelSpec("Wrapped normal","raw_phases","tc",True,lambda rank, params: WrappedNormal(p=P, rank=rank, K=K, params=params, winding_radius=0)),
    # ModelSpec("VMVM", "raw_phases","dc",False,lambda rank, params: VMVM(p=P,K=K,params=params,oscillatory_data=True)),
    ModelSpec("Quotient torus K-means", "raw_phases", "++", False, clustering=quotient_torus_clustering),
    # ModelSpec("MACG", "U_cos", "gc", True, lambda rank, params: MACG(p=P, q=2, rank=rank, K=K, params=params)),
    # ModelSpec("Singular Wishart", "weighted_U_cos", "wgc", True, lambda rank, params: SingularWishart(p=P, q=2, rank=rank, K=K, params=params)),
    # ModelSpec("Grassmann K-means", "U_cos", "++", False, clustering=grassmann_clustering),
    # ModelSpec("Weighted Grassmann K-means", "weighted_U_cos", "++", False, clustering=weighted_grassmann_clustering),
    # ModelSpec("Watson", "leading_U_cos", "dc", False, lambda rank, params: Watson(p=P, K=K, params=params)),
    # ModelSpec("ACG", "leading_U_cos", "dc", True, lambda rank, params: ACG(p=P, rank=rank, K=K, params=params)),
    # ModelSpec("Diametrical K-means", "leading_U_cos", "++", False, clustering=diametrical_clustering),
    # ModelSpec("Least-squares K-means", "leading_U_cos", "++", False, clustering=least_squares_sign_flip),
    # ModelSpec("Normal", "timeseries", "ls", True, lambda rank, params: Normal(p=P, rank=rank, K=K, params=params)),
    # ModelSpec("Complex Normal", "analytic_signal", "ls", True, lambda rank, params: Normal(p=P, rank=rank, K=K, complex=True, params=params)),
)


def load_representation(h5_path: Path, representation: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one model representation while avoiding duplicate 2.6 GB copies."""
    with h5py.File(h5_path, "r") as handle:
        if representation == "U_complex":
            train, test = handle["U_complex_train"][:], handle["U_complex_test"][:]
        elif representation == "leading_U_cos":
            train = handle["U_cos_train"][:, :, 0]
            test = handle["U_cos_test"][:, :, 0]
        elif representation == "U_cos":
            train, test = handle["U_cos_train"][:], handle["U_cos_test"][:]
        elif representation == "weighted_U_cos":
            train = handle["U_cos_train"][:] * np.sqrt(
                handle["L_cos_train"][:][:, None, :]
            )
            test = handle["U_cos_test"][:] * np.sqrt(
                handle["L_cos_test"][:][:, None, :]
            )
        elif representation == "timeseries":
            train, test = handle["timeseries_train"][:], handle["timeseries_test"][:]
        elif representation == "analytic_signal":
            # Analytic signal: Hilbert amplitude multiplied by exp(i theta).
            train = handle["U_complex_train"][:] * handle["A_train"][:]
            test = handle["U_complex_test"][:] * handle["A_test"][:]
        elif representation == "quotient_phases":
            train = quotient_phases(handle["U_complex_train"][:])
            test = quotient_phases(handle["U_complex_test"][:])
        elif representation == "raw_phases":
            train = np.angle(handle["U_complex_train"][:])
            test = np.angle(handle["U_complex_test"][:])
        else:
            raise ValueError(f"Unknown representation: {representation}")
    train = np.ascontiguousarray(train)
    test = np.ascontiguousarray(test)
    if train.shape[0] % POINTS_PER_SUBJECT or test.shape[0] % POINTS_PER_SUBJECT:
        raise ValueError(f"The HDF5 datasets must contain {POINTS_PER_SUBJECT} samples per subject.")
    return train, test


def quotient_phases(unit_complex: np.ndarray) -> np.ndarray:
    """Take theta from exp(i theta), quotient by the last phase, and drop it."""
    theta = np.angle(unit_complex)
    relative = theta - theta[:, [-1]]
    return np.arctan2(np.sin(relative[:, :-1]), np.cos(relative[:, :-1]))


def active_task_partition() -> tuple[np.ndarray, np.ndarray]:
    """Return the within-task mask and labels after pre-stimulus removal."""
    masks = []
    labels = []
    for task, (name, prestim) in enumerate(zip(TASKS, PRESTIM_LENGTHS)):
        task_vector = np.loadtxt(TASK_VECTOR_DIR / f"100206_{name}_task_vector.txt", dtype=bool)
        task_vector = task_vector[prestim:]
        if task_vector.size != FIT_TASK_LENGTHS[task]:
            raise ValueError(f"Unexpected task-vector length for {name}: {task_vector.size}.")
        masks.append(task_vector)
        labels.append(np.full(task_vector.sum(), task, dtype=int))
    active = np.concatenate(masks)
    return active, np.eye(K, dtype=float)[np.concatenate(labels)].T


def normalized_mutual_information(z_states: np.ndarray, z_tasks: np.ndarray) -> float:
    """NMI between hard/soft partitions, matching the earlier paper analysis."""

    def mutual_information(z1: np.ndarray, z2: np.ndarray) -> float:
        joint = z1 @ z2.T
        total = joint.sum()
        if total <= 0:
            return np.nan
        joint = joint / total
        independent = np.outer(joint.sum(axis=1), joint.sum(axis=0))
        occupied = joint > 0
        return float(
            np.sum(joint[occupied] * np.log(joint[occupied] / independent[occupied]))
        )

    overlap = mutual_information(z_states, z_tasks)
    denominator = mutual_information(z_states, z_states) + mutual_information(
        z_tasks, z_tasks
    )
    return float(2 * overlap / denominator) if denominator > 0 else np.nan


def subject_nmi_rows(
    posterior: np.ndarray, set_name: str, model: str, rank: int | None
) -> list[dict]:
    """Calculate one task/state NMI for every subject in a concatenated set."""
    posterior = np.asarray(posterior)
    if posterior.shape[0] != K and posterior.shape[1] == K:
        posterior = posterior.T
    if posterior.shape[0] != K or posterior.shape[1] % POINTS_PER_SUBJECT:
        raise ValueError(
            f"Posterior shape {posterior.shape} is incompatible with K={K} and "
            f"{POINTS_PER_SUBJECT} points per subject."
        )
    active, truth = active_task_partition()
    return [
        {
            "model": model,
            "rank": rank,
            "set": set_name,
            "subject": subject,
            "NMI": normalized_mutual_information(
                posterior[
                    :,
                    subject * POINTS_PER_SUBJECT : (subject + 1) * POINTS_PER_SUBJECT,
                ][:, active],
                truth,
            ),
        }
        for subject in range(posterior.shape[1] // POINTS_PER_SUBJECT)
    ]


def posterior_from_model(model, data: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(data)
    model.to(device=tensor.device)
    with torch.no_grad():
        return as_numpy(model.posterior(tensor))


def as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def fit_mixture(
    spec: ModelSpec,
    train: np.ndarray,
    rank: int,
    previous_params: dict | None,
    quotient_labels: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray]:
    if spec.name == "Wrapped normal" and previous_params is not None:
        previous_rank = int(previous_params["M"].shape[-1])
        model = spec.constructor(previous_rank, previous_params)
        model.r = rank
    else:
        model = spec.constructor(rank, previous_params)
    init = "no" if previous_params is not None else spec.init
    if previous_params is None and spec.init == "raw_phases":
        model.initialize(
            torch.from_numpy(train),
            posterior=torch.from_numpy(quotient_labels),
        )
        init = "no"
    params, posterior, _ = mixture_torch_loop(
        model,
        train,
        tol=args.tol,
        max_iter=args.max_iter,
        num_repl=(
            1
            if previous_params is not None or spec.init == "raw_phases"
            else args.num_repl
        ),
        init=init,
        LR=args.learning_rate,
        suppress_output=args.quiet,
        threads=args.threads,
        decrease_lr_on_plateau=False,
        num_comparison=args.num_comparison,
    )
    return params, as_numpy(posterior)


def load_raw_phase_labels(h5_path: Path, args: argparse.Namespace) -> np.ndarray:
    """Cluster all 116 non-quotiented training phases for quotient models."""
    with h5py.File(h5_path, "r") as handle:
        phases = np.angle(handle["U_complex_train"][:])
    if args.quotient_initializer == "qtc":
        clustering = quotient_torus_clustering
        clustering_data = phases
    else:
        clustering = diametrical_clustering
        clustering_data = np.exp(1j * phases) / np.sqrt(phases.shape[1])
    _, labels, _ = clustering(
        clustering_data,
        K=K,
        max_iter=args.max_iter,
        num_repl=args.num_repl,
        init="++",
        tol=args.tol,
        suppress_output=args.quiet,
    )
    return np.asarray(labels, dtype=int)


def fit_kmeans(
    spec: ModelSpec, train: np.ndarray, args: argparse.Namespace
) -> tuple[dict, np.ndarray]:
    kwargs = dict(
        K=K,
        max_iter=args.max_iter,
        num_repl=args.num_repl,
        init=spec.init,
        tol=args.tol,
    )
    if spec.clustering is not least_squares_sign_flip:
        kwargs["suppress_output"] = args.quiet
    centroids, labels, _ = spec.clustering(train.copy(), **kwargs)
    return {"C": centroids}, np.eye(K)[np.asarray(labels, dtype=int)].T


def kmeans_posterior(spec: ModelSpec, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign held-out samples using the objective used during PCMM clustering."""
    if spec.clustering is least_squares_sign_flip:
        x = data.copy()
        x[(x > 0).sum(axis=1) > x.shape[1] / 2] *= -1
        similarity = -np.sum((x[:, None] - centroids[None]) ** 2, axis=-1)
    elif spec.clustering is diametrical_clustering or spec.clustering is projective_hyperplane_clustering:
        similarity = np.abs(data @ centroids.conj().T) ** 2
    elif spec.clustering is grassmann_clustering:
        q = data.shape[2]
        similarity = -(
            2 * q
            - 2
            * np.linalg.norm(
                np.swapaxes(data[:, None], -2, -1) @ centroids[None],
                axis=(-2, -1),
            )
            ** 2
        ) / np.sqrt(2)
    elif spec.clustering is weighted_grassmann_clustering:
        data_weights = np.linalg.norm(data, axis=1) ** 2
        center_weights = np.linalg.norm(centroids, axis=1) ** 2
        cross = np.swapaxes(data, -2, -1)[:, None] @ centroids[None]
        similarity = -(
            np.sum(data_weights**2, axis=1)[:, None]
            + np.sum(center_weights**2, axis=1)[None]
            - 2 * np.linalg.norm(cross, axis=(-2, -1)) ** 2
        ) / np.sqrt(2)
    elif spec.clustering is quotient_torus_clustering:
        centers = np.asarray(centroids)
        observations = data if np.iscomplexobj(data) else np.exp(1j * data)
        similarity = np.abs(observations @ centers.conj().T)
    else:
        raise ValueError(f"No assignment rule for {spec.name}")
    return np.eye(K)[np.argmax(similarity, axis=1)].T


def save_results(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(output, index=False)


def retained_results(output: Path, models_being_run: set[str]) -> list[dict]:
    """Keep results for models that are not part of the current invocation."""
    if not output.exists():
        return []

    existing = pd.read_csv(output)
    missing = set(RESULT_COLUMNS) - set(existing.columns)
    if missing:
        raise ValueError(
            f"Existing results file {output} is missing columns: {sorted(missing)}"
        )
    retained = existing.loc[
        ~existing["model"].isin(models_being_run), list(RESULT_COLUMNS)
    ]
    return retained.to_dict("records")


def run(args: argparse.Namespace) -> pd.DataFrame:
    selected = {name.strip() for name in args.models.split(",")} if args.models else None
    specs = [spec for spec in MODEL_SPECS if selected is None or spec.name in selected]
    if selected is not None:
        unknown = selected - {spec.name for spec in specs}
        if unknown:
            raise ValueError(f"Unknown model names: {sorted(unknown)}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    # A run replaces results only for its active models. This permits MODEL_SPECS
    # entries to be commented out and fitted in separate sequential invocations
    # without erasing results produced by earlier invocations.
    rows = retained_results(args.output, {spec.name for spec in specs})
    parameter_dir = args.output.parent / "params"
    parameter_dir.mkdir(parents=True, exist_ok=True)
    quotient_labels = None

    for spec in specs:
        print(f"\nLoading {spec.dataset} for {spec.name}")
        train, test = load_representation(args.data, spec.dataset)
        if spec.init == "raw_phases" and quotient_labels is None:
            quotient_labels = load_raw_phase_labels(args.data, args)
        ranks = args.ranks if spec.ranked else (None,)
        previous_params = None

        for rank_value in ranks:
            rank = int(rank_value) if rank_value is not None else 1
            label = f"{spec.name}, rank={rank}" if spec.ranked else spec.name
            print(f"Fitting {label}")
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            if spec.clustering is None:
                params, train_posterior = fit_mixture(
                    spec, train, rank, previous_params, quotient_labels, args
                )
                fitted_model = spec.constructor(rank, params)
                test_posterior = posterior_from_model(fitted_model, test)
                parameter_path = parameter_dir / (
                    spec.name.lower().replace(" ", "_")
                    + (f"_rank={rank}" if spec.ranked else "")
                    + ".pt"
                )
                torch.save(params, parameter_path)
                if spec.ranked:
                    previous_params = params
            else:
                params, train_posterior = fit_kmeans(spec, train, args)
                test_posterior = kmeans_posterior(spec, test, params["C"])
                np.save(
                    parameter_dir / (spec.name.lower().replace(" ", "_") + ".npy"),
                    params,
                    allow_pickle=True,
                )

            rows.extend(
                subject_nmi_rows(
                    train_posterior, "train", spec.name,
                    rank if spec.ranked else None,
                )
            )
            rows.extend(
                subject_nmi_rows(
                    test_posterior, "test", spec.name,
                    rank if spec.ranked else None,
                )
            )
            save_results(args.output, rows)
            current_rank = rank if spec.ranked else None
            current_nmi = [
                row["NMI"]
                for row in rows
                if row["model"] == spec.name and row["set"] == "test" and row["rank"] == current_rank
            ]
            print(f"{label}: mean test NMI = {np.mean(current_nmi):.3f}")

        del train, test

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--models", help="Comma-separated model names; by default all models are fitted.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=100_000)
    parser.add_argument("--num-repl", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-comparison", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--quotient-initializer",
        choices=("qtc", "dc"),
        default="qtc",
        help=(
            "Cluster non-quotiented phases with quotient-torus ('qtc') or "
            "complex diametrical ('dc') clustering before fitting quotient models."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.ranks = tuple(args.ranks)
    if tuple(sorted(set(args.ranks))) != args.ranks:
        parser.error("--ranks must be unique and in ascending order for sequential initialization")
    if any(rank < 1 or rank > P for rank in args.ranks):
        parser.error(f"Every rank must lie in [1, {P}]")
    return args


if __name__ == "__main__":
    run(parse_args())
