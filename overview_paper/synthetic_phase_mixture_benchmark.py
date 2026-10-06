"""Compact two-state phase-mixture benchmark.

Dependencies: numpy, pandas, scikit-learn
Fitting code is intentionally omitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

TAU = 2.0 * np.pi


def wrap(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (x + np.pi) % TAU - np.pi


def _check_noise(noise: float) -> None:
    if not 0.0 <= noise <= TAU:
        raise ValueError("noise must lie in [0, 2*pi].")


def _regional_noise(rng: np.random.Generator, shape, noise: float) -> np.ndarray:
    """
    Independent circular noise with arc width `noise`.

    noise = 0: no perturbation
    noise = 2*pi: uniform phase noise on the full circle
    """
    _check_noise(noise)
    return rng.uniform(-noise / 2.0, noise / 2.0, size=shape)


def _state_groupings(p: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Two distinct assignments of p regions to `rank` latent phase factors."""
    if not 1 <= rank <= p:
        raise ValueError("rank must satisfy 1 <= rank <= p.")

    # Contiguous groups, e.g. p=6, rank=3 -> [0,0,1,1,2,2].
    g1 = np.floor(np.arange(p) * rank / p).astype(int)

    # Fixed permutation gives [0,1,0,2,1,2] for p=6, rank=3.
    if p == 6 and rank == 3:
        g2 = np.array([0, 1, 0, 2, 1, 2])
    else:
        rng = np.random.default_rng(9173 + 31 * p + rank)
        g2 = rng.permutation(g1)
        if np.array_equal(g1, g2):
            g2 = np.roll(g1, 1)
    return g1, g2


def sample_oscillatory(
    noise: float,
    *,
    p: int = 6,
    n_per_state: int = 500,
    rank: int = 3,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two consecutive oscillatory states.

    Each state contains `rank` latent oscillators. Regions assigned to the same
    oscillator have a fixed phase difference. Integer Fourier frequencies make
    the noiseless block coherence signal rank `rank`.
    """
    rng = np.random.default_rng(seed)
    groups = _state_groupings(p, rank)
    cycles = np.array([7, 13, 29, 43, 61, 79])[:rank]
    t = np.arange(n_per_state)

    blocks = []
    for state, group in enumerate(groups):
        latent = TAU * t[:, None] * cycles[None, :] / n_per_state
        latent += rng.uniform(-np.pi, np.pi, size=(1, rank))

        # Common phase nuisance; it cancels in phase differences.
        gamma = TAU * 3 * t[:, None] / n_per_state

        # Large state-specific regional offsets. With one latent oscillator,
        # use two non-collinear phase templates so that both real cosine
        # representations have rank two. For the default p=6 design, each
        # template has cosine eigenvalues 4 and 2, avoiding a non-identifiable
        # leading eigenvector. Higher-rank designs retain the original offsets.
        if rank == 1 and p == 6:
            rank_one_offsets = (
                np.array([0, 0, 0, 0, np.pi / 2, np.pi / 2]),
                np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0]),
            )
            alpha = rank_one_offsets[state]
        else:
            alpha = np.zeros(p) if state == 0 else TAU * np.arange(p) / p

        theta = gamma + latent[:, group] + alpha
        theta += _regional_noise(rng, theta.shape, noise)
        blocks.append(wrap(theta))

    theta = np.vstack(blocks)
    labels = np.repeat([0, 1], n_per_state)
    return theta, labels


def _contrast_basis(groups: np.ndarray, rank: int) -> np.ndarray:
    """One unit contrast per group."""
    p = len(groups)
    B = np.zeros((p, rank))
    for k in range(rank):
        idx = np.flatnonzero(groups == k)
        if len(idx) < 2:
            raise ValueError("For the fixed-state design, each group needs >=2 regions.")
        B[idx[0], k] = 1.0 / np.sqrt(2.0)
        B[idx[1], k] = -1.0 / np.sqrt(2.0)
    return B


def sample_fixed(
    noise: float,
    *,
    p: int = 6,
    n_per_state: int = 500,
    rank: int = 3,
    seed: int = 2,
    latent_scale: float = 0.25,
    random_global_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Independent stimulus-locked snapshots around two fixed phase patterns.

    Structured trial variation has rank `rank`; `noise` adds independent
    regional perturbations. At noise=2*pi, regional phases are maximally noisy.
    """
    if p != 6 or not 1 <= rank <= 3:
        raise ValueError("For p=6, the fixed design supports ranks 1, 2, or 3.")

    rng = np.random.default_rng(seed)
    g1, g2 = _state_groupings(p, rank)
    bases = (_contrast_basis(g1, rank), _contrast_basis(g2, rank))

    means = (
        np.array([0, 0, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3, 4*np.pi/3]),
        np.array([0, np.pi, 0, np.pi, 0, np.pi]),
    )

    blocks = []
    for mu, B in zip(means, bases):
        scores = rng.normal(0.0, latent_scale, size=(n_per_state, rank))
        theta = mu[None, :] + scores @ B.T

        if random_global_phase:
            theta += rng.uniform(0.0, TAU, size=(n_per_state, 1))

        theta += _regional_noise(rng, theta.shape, noise)
        blocks.append(wrap(theta))

    theta = np.vstack(blocks)
    labels = np.repeat([0, 1], n_per_state)
    return theta, labels


def construct_representations(theta: np.ndarray, reference: int = -1) -> dict[str, np.ndarray]:
    """Construct torus, projective, phase-difference, and eigen representations."""
    theta = np.asarray(theta)
    _, p = theta.shape

    # Quotient torus: subtract one reference phase and remove that coordinate.
    relative = wrap(theta - theta[:, [reference]])
    keep = np.ones(p, dtype=bool)
    keep[reference] = False
    relative = relative[:, keep]

    # Unit complex vector and a gauge-fixed copy.
    z = np.exp(1j * theta) / np.sqrt(p)
    z_gauge = z * np.exp(-1j * np.angle(z[:, [reference]]))

    # H contains both cosine and sine phase differences.
    H = np.einsum("ni,nj->nij", z, z.conj())
    C = p * H.real
    S = p * H.imag

    upper = np.triu_indices(p, k=1)
    cosine_features = C[:, upper[0], upper[1]]
    sine_features = S[:, upper[0], upper[1]]

    # Each C has rank at most two.
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = eigenvalues[:, ::-1]
    eigenvectors = eigenvectors[:, :, ::-1]
    U2 = eigenvectors[:, :, :2]
    lambda2 = eigenvalues[:, :2]
    u1 = U2[:, :, 0]

    P2 = np.einsum("nik,njk->nij", U2, U2)
    P1 = np.einsum("ni,nj->nij", u1, u1)

    return {
        "torus_raw": theta,
        "torus_quotient": relative,
        "torus_kmeans_embedding": np.c_[np.cos(relative), np.sin(relative)],
        "complex_projective": z,
        "complex_projective_gauge": z_gauge,
        "hermitian_H": H,
        "cosine_matrix": C,
        "sine_matrix": S,
        "cosine_features": cosine_features,
        "sine_features": sine_features,
        "cosine_sine_features": np.c_[cosine_features, sine_features],
        "cosine_eigenvalues_2": lambda2,
        "cosine_eigenvectors_2": U2,
        "cosine_subspace_projector": P2,
        "cosine_leading_vector": u1,
        "cosine_leading_projector": P1,
    }


def empirical_coherence(theta: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """K_s[i,j] = mean exp(i(theta_i-theta_j)) within each state."""
    v = np.exp(1j * theta)
    return {
        int(s): v[labels == s].T @ v[labels == s].conj() / np.sum(labels == s)
        for s in np.unique(labels)
    }


def coherence_spectrum(theta: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Eigenvalues of each empirical coherence matrix."""
    rows = []
    for state, K in empirical_coherence(theta, labels).items():
        vals = np.linalg.eigvalsh(K)[::-1].real
        rows.append({"state": state, **{f"eig_{j+1}": x for j, x in enumerate(vals)}})
    return pd.DataFrame(rows)


def nmi_table(true_labels: np.ndarray, estimated_labels: dict[str, np.ndarray]) -> pd.DataFrame:
    """Score labels returned by external fitting code."""
    rows = []
    for name, labels in estimated_labels.items():
        labels = np.asarray(labels)
        if labels.shape != true_labels.shape:
            raise ValueError(f"{name}: estimated labels have the wrong shape.")
        rows.append({
            "model": name,
            "NMI": normalized_mutual_info_score(
                true_labels, labels, average_method="arithmetic"
            ),
        })
    return pd.DataFrame(rows).sort_values("NMI", ascending=False).reset_index(drop=True)


def reflection_diagnostic(theta: np.ndarray) -> dict[str, float]:
    """Confirm what is lost by retaining cosine differences alone."""
    original = construct_representations(theta)
    reflected = construct_representations(wrap(-theta))
    return {
        "max_abs_C_minus_C_reflected": float(np.max(
            np.abs(original["cosine_matrix"] - reflected["cosine_matrix"])
        )),
        "max_abs_S_plus_S_reflected": float(np.max(
            np.abs(original["sine_matrix"] + reflected["sine_matrix"])
        )),
    }


if __name__ == "__main__":
    NOISE = 0.6
    RANK = 3

    theta_osc, y_osc = sample_oscillatory(NOISE, rank=RANK)
    theta_fix, y_fix = sample_fixed(NOISE, rank=RANK)

    X_osc = construct_representations(theta_osc)
    X_fix = construct_representations(theta_fix)

    print("Oscillatory shapes:")
    print({k: v.shape for k, v in X_osc.items()})
    print("\nOscillatory coherence spectrum:")
    print(coherence_spectrum(theta_osc, y_osc).round(3))
    print("\nReflection diagnostic:")
    print(reflection_diagnostic(theta_fix))

    # Insert labels from your existing model fits:
    #
    # estimated = {
    #     "wrapped normal, quotient torus": labels_wn,
    #     "complex Watson": labels_watson,
    #     "singular Wishart": labels_wishart,
    #     "MACG rank 2": labels_macg,
    #     "real ACG leading vector": labels_acg,
    # }
    # print(nmi_table(y_fix, estimated))
