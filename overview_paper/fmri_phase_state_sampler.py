"""Compact two-state phase-mixture benchmark for fMRI-style data.

The public calling structure matches the original script:

    theta_osc, y_osc = sample_oscillatory(NOISE, rank=RANK)
    theta_fix, y_fix = sample_fixed(NOISE, rank=RANK)

Conventions
-----------
* p=6 by default and there are two equally sized states.
* ``rank=1`` is the requested *isotropic benchmark* convention.
  It does not mean that the p x p identity covariance has algebraic rank one.
* ``rank>1`` gives a state-specific rank-``rank`` structured correlation
  component plus a small isotropic residual.
* ``noise`` is an angular cloud-width control in [0, 2*pi].
  At zero there is no perturbation.  As noise increases, the wrapped-normal
  cloud widens.  At exactly 2*pi, every regional phase is sampled independently
  and uniformly, so the two states are indistinguishable.
* The two states always have maximally separated *relative-phase templates*:
  state 0 is synchronized and state 1 is evenly spread around the circle.

The oscillatory sampler adds a common rotating carrier.  It therefore has
uniform regional phase marginals over complete cycles while preserving the
state-specific relative-phase pattern.  The fixed sampler omits the carrier
and gives stimulus-locked, centrally concentrated phase snapshots.
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
    """Validate the angular cloud-width parameter."""
    if not 0.0 <= noise <= TAU:
        raise ValueError("noise must lie in [0, 2*pi].")


def _check_design(p: int, rank: int) -> None:
    if p < 2:
        raise ValueError("p must be at least 2.")
    if not 1 <= rank <= p:
        raise ValueError("rank must satisfy 1 <= rank <= p.")


def _state_groupings(p: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Two different assignments of regions to rank latent covariance groups."""
    if rank <= 1:
        raise ValueError("groupings are only used for rank > 1.")

    # Contiguous groups, e.g. p=6, rank=3 -> [0,0,1,1,2,2].
    g1 = np.floor(np.arange(p) * rank / p).astype(int)

    # Preserve the grouping used in the initial benchmark for p=6, rank=3.
    if p == 6 and rank == 3:
        g2 = np.array([0, 1, 0, 2, 1, 2])
    else:
        rng = np.random.default_rng(9173 + 31 * p + rank)
        g2 = rng.permutation(g1)
        if np.array_equal(g1, g2):
            g2 = np.roll(g1, 1)

    return g1, g2


def _maximally_separated_templates(p: int) -> tuple[np.ndarray, np.ndarray]:
    """Two strongly separated relative-phase mean patterns.

    State 0 is synchronized.  State 1 is a regular p-gon on the circle.
    The difference is not merely a common/global phase shift.
    """
    mu_1 = np.zeros(p)
    mu_2 = TAU * np.arange(p) / p
    return wrap(mu_1), wrap(mu_2)


def _rank_structured_correlation(groups: np.ndarray, rank: int) -> np.ndarray:
    """Rank-r block correlation matrix with unit diagonal."""
    p = len(groups)
    loading = np.zeros((p, rank))
    loading[np.arange(p), groups] = 1.0
    return loading @ loading.T


def _state_correlation_matrices(
    noise: float,
    *,
    p: int,
    rank: int,
    anisotropy_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation shapes for the two states before angular scaling.

    For rank=1 both matrices are identity.  For rank>1, each matrix contains a
    rank-r grouped component.  Independent noise progressively makes the cloud
    more spherical as noise approaches 2*pi.
    """
    if not 0.0 <= anisotropy_strength < 1.0:
        raise ValueError("anisotropy_strength must lie in [0, 1).")

    eye = np.eye(p)
    if rank == 1:
        return eye.copy(), eye.copy()

    g1, g2 = _state_groupings(p, rank)
    structured_1 = _rank_structured_correlation(g1, rank)
    structured_2 = _rank_structured_correlation(g2, rank)

    fraction = noise / TAU
    # Keep anisotropy strong at medium noise, but let it vanish smoothly near
    # maximal noise so that the two state distributions converge.
    weight = anisotropy_strength * (1.0 - fraction**4)

    corr_1 = (1.0 - weight) * eye + weight * structured_1
    corr_2 = (1.0 - weight) * eye + weight * structured_2
    return corr_1, corr_2


def _angular_scale(noise: float) -> float:
    """Map [0, 2*pi) to a wrapped-normal scale in [0, infinity).

    For small noise, sigma is approximately noise/4, so ``noise`` behaves like
    an approximate 95% cloud width.  The scale diverges near 2*pi, reflecting
    convergence to circular uniformity.  The endpoint itself is handled by
    exact uniform sampling.
    """
    if noise == 0.0:
        return 0.0
    return float(np.tan(noise / 4.0))


def latent_angular_covariance_matrices(
    noise: float,
    *,
    p: int = 6,
    rank: int = 3,
    anisotropy_strength: float = 0.85,
) -> dict[int, np.ndarray]:
    """Covariances of the unwrapped Gaussian angular perturbations.

    At noise=2*pi the actual sampler uses independent circular uniforms;
    their conventional wrapped-angle covariance is (pi^2/3) I.
    """
    _check_noise(noise)
    _check_design(p, rank)

    if np.isclose(noise, TAU):
        uniform_covariance = (np.pi**2 / 3.0) * np.eye(p)
        return {0: uniform_covariance.copy(), 1: uniform_covariance.copy()}

    corr_1, corr_2 = _state_correlation_matrices(
        noise,
        p=p,
        rank=rank,
        anisotropy_strength=anisotropy_strength,
    )
    variance = _angular_scale(noise) ** 2
    return {0: variance * corr_1, 1: variance * corr_2}


def _draw_gaussian_perturbations(
    rng: np.random.Generator,
    *,
    n: int,
    covariance: np.ndarray,
) -> np.ndarray:
    """Draw independent multivariate Gaussian angular perturbations."""
    p = covariance.shape[0]
    if np.allclose(covariance, 0.0):
        return np.zeros((n, p))
    return rng.multivariate_normal(
        mean=np.zeros(p),
        cov=covariance,
        size=n,
        check_valid="raise",
    )


def _draw_ar1_perturbations(
    rng: np.random.Generator,
    *,
    n: int,
    covariance: np.ndarray,
    temporal_ar: float,
) -> np.ndarray:
    """Stationary vector AR(1) with the requested marginal covariance."""
    if not 0.0 <= temporal_ar < 1.0:
        raise ValueError("temporal_ar must lie in [0, 1).")

    p = covariance.shape[0]
    if np.allclose(covariance, 0.0):
        return np.zeros((n, p))

    innovations = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=covariance,
        size=n,
        check_valid="raise",
    )
    out = np.empty((n, p))
    out[0] = innovations[0]
    innovation_scale = np.sqrt(1.0 - temporal_ar**2)

    for t in range(1, n):
        out[t] = temporal_ar * out[t - 1] + innovation_scale * innovations[t]

    return out


def sample_oscillatory(
    noise: float,
    *,
    p: int = 6,
    n_per_state: int = 500,
    rank: int = 3,
    seed: int = 1,
    temporal_ar: float = 0.90,
    carrier_cycles: int = 7,
    anisotropy_strength: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample two consecutive oscillatory phase states.

    A common rotating carrier makes every regional phase marginally uniform
    over complete cycles.  Relative phases are centered on two maximally
    separated templates.  ``rank=1`` is isotropic; ``rank>1`` adds a
    state-specific rank-r covariance structure.

    ``noise`` lies in [0, 2*pi].  At 2*pi, all regional phases are independent
    circular uniforms and state labels carry no distributional information.
    """
    _check_noise(noise)
    _check_design(p, rank)

    if n_per_state < 2:
        raise ValueError("n_per_state must be at least 2.")
    if carrier_cycles < 1 or int(carrier_cycles) != carrier_cycles:
        raise ValueError("carrier_cycles must be a positive integer.")

    rng = np.random.default_rng(seed)
    means = _maximally_separated_templates(p)
    labels = np.repeat([0, 1], n_per_state)

    if np.isclose(noise, TAU):
        theta = rng.uniform(-np.pi, np.pi, size=(2 * n_per_state, p))
        return theta, labels

    covariances = latent_angular_covariance_matrices(
        noise,
        p=p,
        rank=rank,
        anisotropy_strength=anisotropy_strength,
    )

    t = np.arange(n_per_state)
    blocks = []
    for state, mean in enumerate(means):
        perturbation = _draw_ar1_perturbations(
            rng,
            n=n_per_state,
            covariance=covariances[state],
            temporal_ar=temporal_ar,
        )

        initial_phase = rng.uniform(-np.pi, np.pi)
        carrier = (
            TAU * carrier_cycles * t / n_per_state + initial_phase
        )[:, None]

        blocks.append(wrap(carrier + mean[None, :] + perturbation))

    return np.vstack(blocks), labels


def sample_fixed(
    noise: float,
    *,
    p: int = 6,
    n_per_state: int = 500,
    rank: int = 3,
    seed: int = 2,
    anisotropy_strength: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample independent stimulus-locked phase snapshots from two states.

    Marginal phases are centered around state-specific regional means for
    noise < 2*pi.  ``rank=1`` produces isotropic clouds; ``rank>1`` produces
    anisotropic clouds with a rank-r structured covariance component.

    At noise=2*pi, every regional phase is independently uniform and the two
    states are exactly indistinguishable.
    """
    _check_noise(noise)
    _check_design(p, rank)

    if n_per_state < 1:
        raise ValueError("n_per_state must be positive.")

    rng = np.random.default_rng(seed)
    means = _maximally_separated_templates(p)
    labels = np.repeat([0, 1], n_per_state)

    if np.isclose(noise, TAU):
        theta = rng.uniform(-np.pi, np.pi, size=(2 * n_per_state, p))
        return theta, labels

    covariances = latent_angular_covariance_matrices(
        noise,
        p=p,
        rank=rank,
        anisotropy_strength=anisotropy_strength,
    )

    blocks = []
    for state, mean in enumerate(means):
        perturbation = _draw_gaussian_perturbations(
            rng,
            n=n_per_state,
            covariance=covariances[state],
        )
        blocks.append(wrap(mean[None, :] + perturbation))

    return np.vstack(blocks), labels


def construct_representations(
    theta: np.ndarray,
    reference: int = -1,
) -> dict[str, np.ndarray]:
    """Construct torus, projective, phase-difference, and eigen representations."""
    theta = np.asarray(theta)
    if theta.ndim != 2:
        raise ValueError("theta must have shape (n_samples, p).")
    _, p = theta.shape

    relative = wrap(theta - theta[:, [reference]])
    keep = np.ones(p, dtype=bool)
    keep[reference] = False
    relative = relative[:, keep]

    z = np.exp(1j * theta) / np.sqrt(p)
    z_gauge = z * np.exp(-1j * np.angle(z[:, [reference]]))

    H = np.einsum("ni,nj->nij", z, z.conj())
    C = p * H.real
    S = p * H.imag

    upper = np.triu_indices(p, k=1)
    cosine_features = C[:, upper[0], upper[1]]
    sine_features = S[:, upper[0], upper[1]]

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


def empirical_coherence(
    theta: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """K_s[i,j] = mean exp(i(theta_i-theta_j)) within each state."""
    v = np.exp(1j * theta)
    return {
        int(state): (
            v[labels == state].T
            @ v[labels == state].conj()
            / np.sum(labels == state)
        )
        for state in np.unique(labels)
    }


def coherence_spectrum(
    theta: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Eigenvalues of each empirical phase-coherence matrix."""
    rows = []
    for state, matrix in empirical_coherence(theta, labels).items():
        values = np.linalg.eigvalsh(matrix)[::-1].real
        rows.append(
            {
                "state": state,
                **{
                    f"eig_{index + 1}": value
                    for index, value in enumerate(values)
                },
            }
        )
    return pd.DataFrame(rows)


def nmi_table(
    true_labels: np.ndarray,
    estimated_labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Score labels returned by external fitting code."""
    rows = []
    for name, labels in estimated_labels.items():
        labels = np.asarray(labels)
        if labels.shape != true_labels.shape:
            raise ValueError(f"{name}: estimated labels have the wrong shape.")
        rows.append(
            {
                "model": name,
                "NMI": normalized_mutual_info_score(
                    true_labels,
                    labels,
                    average_method="arithmetic",
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("NMI", ascending=False)
        .reset_index(drop=True)
    )


def reflection_diagnostic(theta: np.ndarray) -> dict[str, float]:
    """Confirm what is lost by retaining cosine phase differences alone."""
    original = construct_representations(theta)
    reflected = construct_representations(wrap(-theta))
    return {
        "max_abs_C_minus_C_reflected": float(
            np.max(
                np.abs(
                    original["cosine_matrix"]
                    - reflected["cosine_matrix"]
                )
            )
        ),
        "max_abs_S_plus_S_reflected": float(
            np.max(
                np.abs(
                    original["sine_matrix"]
                    + reflected["sine_matrix"]
                )
            )
        ),
    }


if __name__ == "__main__":
    NOISE = 0.8
    RANK = 3

    theta_osc, y_osc = sample_oscillatory(NOISE, rank=RANK)
    theta_fix, y_fix = sample_fixed(NOISE, rank=RANK)

    X_osc = construct_representations(theta_osc)
    X_fix = construct_representations(theta_fix)

    print("Oscillatory shapes:")
    print({name: value.shape for name, value in X_osc.items()})

    print("\nOscillatory coherence spectrum:")
    print(coherence_spectrum(theta_osc, y_osc).round(3))

    print("\nFixed coherence spectrum:")
    print(coherence_spectrum(theta_fix, y_fix).round(3))

    print("\nLatent angular covariance eigenvalues:")
    for state, covariance in latent_angular_covariance_matrices(
        NOISE,
        rank=RANK,
    ).items():
        print(state, np.linalg.eigvalsh(covariance)[::-1].round(3))

    print("\nReflection diagnostic:")
    print(reflection_diagnostic(theta_fix))

    # Insert labels from your existing model fits:
    #
    # estimated = {
    #     "toroidal k-means": labels_kmeans,
    #     "complex ACG mixture": labels_cacg,
    #     "wrapped normal mixture": labels_wn,
    # }
    # print(nmi_table(y_fix, estimated))
