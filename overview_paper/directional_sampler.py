#!/usr/bin/env python3
"""Generate matched directional-distribution experiments as CSV files.

The experiment spans vector/torus distributions and matrix distributions on
the Stiefel manifold V_q(R^p).  It deliberately uses one shared orthogonal
basis per ambient dimension so that preferred subspaces correspond across
families.

Implemented distributions and kernels
-------------------------------------
wrapped_normal
    theta = (mu + N(0, Sigma)) mod 2*pi (direct, IID).
vmf
    exp(kappa * mu.T @ x), Wood/Ulrich rejection sampler (IID).
watson
    exp(kappa * (mu.T @ x)**2), sampled as a Bingham law (IID).
bingham
    exp(-x.T @ A @ x), BACG rejection sampler (IID).
fisher_bingham
    exp(b.T @ x - x.T @ A @ x), diagnosed geodesic-slice MCMC.
acg
    normalize(N(0, Sigma)) (direct, IID).
matrix_fisher
    exp(tr(F.T @ X)) on X.T @ X = I (diagnosed geodesic-slice MCMC).
matrix_bingham
    exp(-tr(X.T @ A @ X)), matrix-BACG rejection sampler (IID).
matrix_fisher_bingham
    exp(tr(F.T @ X) - tr(X.T @ A @ X)) (diagnosed geodesic-slice MCMC).
macg
    polar factor of a matrix-normal draw (direct, IID).

Rank convention
---------------
"full" means maximal identifiable active rank, not necessarily literal full
matrix rank.  For a vector Bingham parameter this is p-1, since A+cI gives the
same distribution.  For a matrix Bingham parameter on V_q(R^p), it is p-q,
leaving a q-dimensional modal subspace.  Matrix Fisher--Bingham instead uses
the maximal gauge-fixed rank p-1: its linear term identifies the q-frame while
its quadratic term adds trace-free transverse anisotropy.  Reduced/half-rank
cells are not part of this experiment.
vMF and Watson are emitted only once with rank_label="rank1".

Concentration convention
------------------------
Low and high mean normalized modal alignments of .20 and .80.  Each family's
scalar natural-parameter multiplier is calibrated to that common population
quantity.  Signed directional/frame models use mean alignment with their modal
direction or frame; axial/subspace models use an affine rescaling of squared
overlap that is zero under the uniform law and one at the mode.  Wrapped-normal,
vMF, Watson, and ACG calibration is deterministic; the remaining families use
a bounded pilot calibration whose diagnostics are recorded in the manifest.
In combined Fisher--Bingham laws, the quadratic term remains a fixed trace-free
anisotropy fraction of the Fisher scale.

References
----------
Wood (1994), Simulation of the von Mises Fisher distribution.
Kent, Ganeiber & Mardia (2018), A new unified approach for the simulation of a
wide class of directional distributions, Statistics and Computing 28:509-521.
Hoff (2009), Simulation of the matrix Bingham-von Mises-Fisher distribution,
JCGS 18:438-456.
Chikuse (1990), The matrix angular central Gaussian distribution, JMVA 33.
Habeck et al. (2025), Geodesic slice sampling on the sphere, JMLR 26.

Example
-------
python overview_paper/directional_sampler.py --output-dir samples --n-samples 400
python overview_paper/directional_sampler.py --output-dir samples_budgeted \
    --sample-size-mode dimension_power --sample-exponent -1 --n-samples 4000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, ive, roots_jacobi
from scipy.stats.sampling import TransformedDensityRejection


Array = NDArray[np.float64]
TWO_PI = 2.0 * math.pi
SAMPLER_SCHEMA_VERSION = 2

ALL_DISTRIBUTIONS = (
    "wrapped_normal",
    "vmf",
    "watson",
    "bingham",
    "fisher_bingham",
    "acg",
    "matrix_fisher",
    "matrix_bingham",
    "matrix_fisher_bingham",
    "macg",
)


@dataclass
class ManifestRow:
    sampler_schema_version: int
    distribution: str
    ambient_dim: int
    matrix_cols: int
    rank_label: str
    active_rank: int
    model_factor_rank: int
    linear_rank: int
    modal_dim: int
    modal_frame_dim: int
    quadratic_nullity: int
    concentration_label: str
    concentration_strength: float
    target_alignment: float
    observed_alignment: float
    alignment_mcse: float
    alignment_qc_status: str
    calibration_method: str
    calibration_status: str
    calibration_iterations: int
    calibration_draws: int
    calibration_estimated_alignment: float
    calibration_mcse: float
    calibration_seconds: float
    fisher_bingham_anisotropy: float
    natural_concentration: float
    linear_concentration: float
    quadratic_concentration: float
    quadratic_spectrum: str
    acg_precision_ratio: float
    n_samples: int
    n_train: int
    n_test: int
    split_index: int
    iid: bool
    sampler: str
    burnin_sweeps: int
    thin_sweeps: int
    n_chains: int
    effective_sample_size: float
    rhat_max: float
    diagnostic_rhat: float
    slice_evaluations: int
    decorrelation_status: str
    mcmc_qc_status: str
    proposal_count: int
    acceptance_rate: float
    sampler_qc_status: str
    seed: int
    basis_seed: int
    basis_filename: str
    orientation: str
    support: str
    density_kernel: str
    elapsed_seconds: float
    filename: str


class SamplingEfficiencyError(RuntimeError):
    """A bounded rejection sampler exhausted its proposal budget."""


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(x) for x in parts)]).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _proposal_budget(n: int, max_proposals: int | None, per_sample: int = 10_000) -> int:
    if max_proposals is not None:
        if max_proposals < n:
            raise ValueError("max_proposals must be at least n")
        return int(max_proposals)
    return max(10_000, per_sample * n)


def _sample_result(
    samples: Array,
    proposals: int,
    accepted: int,
    return_diagnostics: bool,
) -> Array | tuple[Array, dict]:
    diagnostics = {
        "proposal_count": int(proposals),
        "acceptance_rate": float(accepted / proposals) if proposals else 1.0,
        "sampler_qc_status": "pass",
    }
    return (samples, diagnostics) if return_diagnostics else samples


def orthogonal_basis(p: int, seed: int, orientation: str) -> Array:
    if orientation == "canonical":
        return np.eye(p)
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(p, p)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def unit_rows(x: Array) -> Array:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def uniform_sphere(n: int, p: int, rng: np.random.Generator) -> Array:
    return unit_rows(rng.normal(size=(n, p)))


def rotate_from_first_axis(y: Array, mu: Array) -> Array:
    """Householder map taking e_1 to mu, applied to row vectors y."""
    e = np.zeros_like(mu)
    e[0] = 1.0
    u = e - mu
    uu = float(u @ u)
    if uu < 1e-28:
        return y
    return y - 2.0 * np.outer((y @ u) / uu, u)


def sample_vmf(
    n: int,
    mu: Array,
    kappa: float,
    rng: np.random.Generator,
    *,
    max_proposals: int | None = None,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, dict]:
    """IID vMF draws on S^(p-1), using the Wood/Ulrich rejection method."""
    p = mu.size
    if p < 2:
        raise ValueError("vMF requires ambient dimension >= 2")
    if kappa < 1e-10:
        samples = uniform_sphere(n, p, rng)
        return _sample_result(samples, n, n, return_diagnostics)

    m = p - 1.0
    # Stable form of (sqrt(4*kappa^2+m^2)-2*kappa)/m.
    b = m / (math.sqrt(4.0 * kappa * kappa + m * m) + 2.0 * kappa)
    x0 = (1.0 - b) / (1.0 + b)
    log_c = kappa * x0 + m * math.log1p(-(x0 * x0))
    alpha = 0.5 * m

    w_parts: list[Array] = []
    remaining = n
    proposals = 0
    accepted_total = 0
    proposal_budget = _proposal_budget(n, max_proposals, per_sample=1_000)
    while remaining:
        available = proposal_budget - proposals
        if available <= 0:
            raise SamplingEfficiencyError(
                f"vMF exhausted {proposal_budget:,} proposals for {n:,} draws"
            )
        batch = min(max(64, 2 * remaining), available)
        z = rng.beta(alpha, alpha, size=batch)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        lhs = kappa * w + m * np.log1p(-x0 * w) - log_c
        accepted = w[np.log(rng.random(batch)) <= lhs]
        proposals += batch
        accepted_total += accepted.size
        take = min(remaining, accepted.size)
        if take:
            w_parts.append(accepted[:take])
            remaining -= take
    w_all = np.concatenate(w_parts)
    tangent = uniform_sphere(n, p - 1, rng)
    y = np.column_stack((w_all, np.sqrt(np.maximum(0.0, 1.0 - w_all**2))[:, None] * tangent))
    samples = rotate_from_first_axis(y, mu)
    return _sample_result(samples, proposals, accepted_total, return_diagnostics)


def _standardize_precision(a: Array) -> tuple[Array, Array, Array]:
    a = 0.5 * (a + a.T)
    values, vectors = np.linalg.eigh(a)
    values = np.maximum(values - values.min(), 0.0)
    return values, vectors, (vectors * values) @ vectors.T


def _bacg_b(eigenvalues: Array) -> float:
    p = eigenvalues.size
    if eigenvalues.max(initial=0.0) < 1e-14:
        return float(p)

    def objective(b: float) -> float:
        return float(np.sum(1.0 / (b + 2.0 * eigenvalues)) - 1.0)

    return float(brentq(objective, 1e-12, float(p), xtol=1e-13, rtol=1e-13))


def sample_bingham(
    n: int,
    precision: Array,
    rng: np.random.Generator,
    *,
    max_proposals: int | None = None,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, dict]:
    """IID draws with density proportional to exp(-x' precision x)."""
    lam, q, _ = _standardize_precision(precision)
    p = lam.size
    if lam.max(initial=0.0) < 1e-14:
        samples = uniform_sphere(n, p, rng)
        return _sample_result(samples, n, n, return_diagnostics)
    b = _bacg_b(lam)
    omega = 1.0 + 2.0 * lam / b
    log_m = -0.5 * (p - b) + 0.5 * p * math.log(p / b)

    accepted_parts: list[Array] = []
    remaining = n
    proposals = 0
    accepted_total = 0
    proposal_budget = _proposal_budget(n, max_proposals)
    while remaining:
        available = proposal_budget - proposals
        if available <= 0:
            raise SamplingEfficiencyError(
                f"Bingham BACG exhausted {proposal_budget:,} proposals for {n:,} draws"
            )
        batch = min(max(64, 2 * remaining), available)
        z = rng.normal(size=(batch, p)) / np.sqrt(omega)[None, :]
        z = unit_rows(z)
        u = (z * z) @ lam
        log_ratio = -u + 0.5 * p * np.log1p(2.0 * u / b) - log_m
        keep = z[np.log(rng.random(batch)) <= np.minimum(0.0, log_ratio)]
        proposals += batch
        accepted_total += keep.shape[0]
        take = min(remaining, keep.shape[0])
        if take:
            accepted_parts.append(keep[:take])
            remaining -= take
    samples = np.concatenate(accepted_parts, axis=0) @ q.T
    return _sample_result(samples, proposals, accepted_total, return_diagnostics)


def sample_fisher_bingham(
    n: int,
    linear: Array,
    precision: Array,
    rng: np.random.Generator,
    *,
    max_proposals: int | None = None,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, dict]:
    """IID FBACG draws with density exp(linear'x - x'precision*x)."""
    kappa = float(np.linalg.norm(linear))
    _, _, a = _standardize_precision(precision)
    if kappa < 1e-12:
        return sample_bingham(
            n,
            a,
            rng,
            max_proposals=max_proposals,
            return_diagnostics=return_diagnostics,
        )
    mu = linear / kappa
    p = mu.size
    envelope_a = a + 0.5 * kappa * (np.eye(p) - np.outer(mu, mu))

    parts: list[Array] = []
    remaining = n
    proposals_total = 0
    accepted_total = 0
    proposal_budget = _proposal_budget(n, max_proposals)
    while remaining:
        available = proposal_budget - proposals_total
        if available <= 0:
            raise SamplingEfficiencyError(
                f"Fisher--Bingham rejection exhausted {proposal_budget:,} outer proposals "
                f"for {n:,} draws"
            )
        batch = min(max(32, 2 * remaining), available)
        proposals = sample_bingham(batch, envelope_a, rng)
        t = proposals @ mu
        log_accept = -0.5 * kappa * (1.0 - t) ** 2
        keep = proposals[np.log(rng.random(proposals.shape[0])) <= log_accept]
        proposals_total += proposals.shape[0]
        accepted_total += keep.shape[0]
        take = min(remaining, keep.shape[0])
        if take:
            parts.append(keep[:take])
            remaining -= take
    samples = np.concatenate(parts, axis=0)
    return _sample_result(samples, proposals_total, accepted_total, return_diagnostics)


def _log_uniform_sphere_mgf(dimension: int, z: float) -> float:
    """log E[exp(z*Y_1)] for Y uniform on S^(dimension-1)."""
    if z < 1e-8:
        return z * z / (2.0 * dimension)
    nu = 0.5 * dimension - 1.0
    return float(
        gammaln(0.5 * dimension)
        + nu * math.log(2.0 / z)
        + math.log(float(ive(nu, z)))
        + z
    )


def _vmf_mean_ratio(dimension: int, z: float) -> float:
    if z < 1e-8:
        return z / dimension
    nu = 0.5 * dimension - 1.0
    return float(ive(nu + 1.0, z) / ive(nu, z))


def sample_structured_fisher_bingham(
    n: int,
    mu: Array,
    active_basis: Array,
    null_basis: Array,
    kappa: float,
    quadratic_coefficient: float,
    rng: np.random.Generator,
) -> tuple[Array, str]:
    """IID aligned FB draws using a one-dimensional TDR decomposition.

    The target is exp(kappa*mu'x-a*||P_active*x||^2), with mu in the
    orthogonal null subspace.  Conditional on u=||P_active*x||^2, the active
    direction is uniform and the null-space direction is vMF.  TDR samples
    the exact one-dimensional marginal without a concentration-dependent
    rejection collapse.  If a low-dimensional marginal is not T-concave,
    the exact general FBACG method is used instead.
    """
    r = active_basis.shape[1]
    m = null_basis.shape[1]
    p = mu.size
    if r + m != p or r < 1 or m < 1:
        raise ValueError("active and null bases must form a complete orthogonal basis")
    if np.linalg.norm(null_basis[:, 0] - mu) > 1e-8:
        raise ValueError("the first null-space direction must be mu")

    alpha, beta = 0.5 * r, 0.5 * m

    def log_pdf(u: float) -> float:
        if not 0.0 < u < 1.0:
            return -math.inf
        z = kappa * math.sqrt(1.0 - u)
        return (
            (alpha - 1.0) * math.log(u)
            + (beta - 1.0) * math.log1p(-u)
            - quadratic_coefficient * u
            + _log_uniform_sphere_mgf(m, z)
        )

    result = minimize_scalar(
        lambda u: -log_pdf(float(u)),
        bounds=(1e-11, 1.0 - 1e-11),
        method="bounded",
        options={"xatol": 1e-13},
    )
    mode = float(result.x)
    log_at_mode = log_pdf(mode)

    class Marginal:
        def pdf(self, u: float) -> float:
            return math.exp(log_pdf(u) - log_at_mode) if 0.0 < u < 1.0 else 0.0

        def dpdf(self, u: float) -> float:
            if not 0.0 < u < 1.0:
                return 0.0
            one_minus = 1.0 - u
            z = kappa * math.sqrt(one_minus)
            derivative = (
                (alpha - 1.0) / u
                - (beta - 1.0) / one_minus
                - quadratic_coefficient
            )
            if z > 1e-12:
                derivative -= _vmf_mean_ratio(m, z) * kappa / (2.0 * math.sqrt(one_minus))
            return self.pdf(u) * derivative

    try:
        generator = TransformedDensityRejection(
            Marginal(),
            mode=mode,
            domain=(0.0, 1.0),
            c=-0.5,
            random_state=rng,
        )
        u = np.asarray(generator.rvs(size=n), dtype=float)
    except Exception:
        # This occurs for a few small-p, low-concentration shapes that are not
        # T-concave.  FBACG is fast in exactly those cases.
        precision = quadratic_coefficient * projector(active_basis)
        return sample_fisher_bingham(n, kappa * mu, precision, rng), "fbacg_rejection"

    active_direction = uniform_sphere(n, r, rng)
    null_direction = np.empty((n, m))
    if m == 1:
        z = kappa * np.sqrt(1.0 - u)
        # Stable Bernoulli probability exp(z)/(exp(z)+exp(-z)).
        prob_plus = np.where(z > 20.0, 1.0, 1.0 / (1.0 + np.exp(-2.0 * z)))
        null_direction[:, 0] = np.where(rng.random(n) < prob_plus, 1.0, -1.0)
    else:
        e0 = np.zeros(m)
        e0[0] = 1.0
        for i, concentration in enumerate(kappa * np.sqrt(1.0 - u)):
            null_direction[i] = sample_vmf(1, e0, float(concentration), rng)[0]

    samples = (
        np.sqrt(u)[:, None] * (active_direction @ active_basis.T)
        + np.sqrt(1.0 - u)[:, None] * (null_direction @ null_basis.T)
    )
    return samples, "structured_fb_tdr"


def sample_wrapped_normal(
    n: int,
    basis: Array,
    active_rank: int,
    strength: float,
    rng: np.random.Generator,
    inactive_variance: float,
    interval: str,
) -> Array:
    p = basis.shape[0]
    tau = max(strength * active_rank, 1e-12)
    variances = np.full(p, inactive_variance)
    variances[:active_rank] = 1.0 / tau
    z = rng.normal(size=(n, p)) * np.sqrt(variances)[None, :]
    unwrapped = z @ basis.T
    wrapped = np.mod(unwrapped, TWO_PI)
    if interval == "minus_pi_pi":
        wrapped = np.mod(wrapped + math.pi, TWO_PI) - math.pi
    return wrapped


def sample_acg(n: int, precision: Array, rng: np.random.Generator) -> Array:
    """IID ACG draws parameterized by precision Omega=Sigma^{-1}."""
    values, vectors, _ = _standardize_positive_definite(precision)
    z = rng.normal(size=(n, values.size)) / np.sqrt(values)[None, :]
    return unit_rows(z) @ vectors.T


def _standardize_positive_definite(a: Array) -> tuple[Array, Array, Array]:
    a = 0.5 * (a + a.T)
    values, vectors = np.linalg.eigh(a)
    if values.min(initial=math.inf) <= 0.0:
        raise ValueError("matrix must be positive definite")
    return values, vectors, (vectors * values) @ vectors.T


def polar_factor(y: Array) -> Array:
    gram = 0.5 * (y.T @ y + (y.T @ y).T)
    values, vectors = np.linalg.eigh(gram)
    if values.min(initial=math.inf) <= 1e-14:
        raise FloatingPointError("rank-deficient Gaussian matrix in polar factor")
    inv_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    return y @ inv_sqrt


def sample_macg(n: int, qcols: int, precision: Array, rng: np.random.Generator) -> Array:
    """IID MACG draws on V_q(R^p), parameterized by row precision."""
    values, vectors, _ = _standardize_positive_definite(precision)
    p = values.size
    out = np.empty((n, p, qcols))
    scale = 1.0 / np.sqrt(values)
    for i in range(n):
        y = (rng.normal(size=(p, qcols)) * scale[:, None])
        out[i] = polar_factor(vectors @ y)
    return out


def sample_matrix_bingham(
    n: int,
    qcols: int,
    precision: Array,
    rng: np.random.Generator,
    *,
    max_proposals: int | None = None,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, dict]:
    """IID matrix-Bingham draws via the MACG envelope."""
    lam, vectors, _ = _standardize_precision(precision)
    p = lam.size
    if qcols >= p:
        raise ValueError("matrix Bingham requires matrix_cols < ambient_dim")
    if lam.max(initial=0.0) < 1e-14:
        samples = sample_macg(n, qcols, np.eye(p), rng)
        return _sample_result(samples, n, n, return_diagnostics)
    b = _bacg_b(lam)
    omega = 1.0 + 2.0 * lam / b
    log_m_one = -0.5 * (p - b) + 0.5 * p * math.log(p / b)

    out: list[Array] = []
    proposals = 0
    proposal_budget = _proposal_budget(n, max_proposals)
    while len(out) < n:
        if proposals >= proposal_budget:
            raise SamplingEfficiencyError(
                f"matrix BACG exhausted {proposal_budget:,} proposals for {n:,} draws"
            )
        proposals += 1
        y = rng.normal(size=(p, qcols)) / np.sqrt(omega)[:, None]
        z = polar_factor(y)  # eigen-coordinates
        g = (z.T * lam[None, :]) @ z
        sign, logdet = np.linalg.slogdet(np.eye(qcols) + (2.0 / b) * g)
        if sign <= 0:
            raise FloatingPointError("non-positive MACG envelope determinant")
        log_ratio = -float(np.trace(g)) + 0.5 * p * logdet - qcols * log_m_one
        if math.log(rng.random()) <= min(0.0, log_ratio):
            out.append(vectors @ z)
    samples = np.stack(out, axis=0)
    return _sample_result(samples, proposals, n, return_diagnostics)


def _column_log_kernel_factory(x: Array, v: Array, f: Array, a: Array) -> Callable[[float], float]:
    fx = float(f @ x)
    fv = float(f @ v)
    axv = a @ v
    ax = float(x @ (a @ x))
    av = float(v @ axv)
    cross = float(x @ axv)

    def log_kernel(theta: float) -> float:
        c, s = math.cos(theta), math.sin(theta)
        return fx * c + fv * s - (ax * c * c + 2.0 * cross * c * s + av * s * s)

    return log_kernel


def _shrinkage_slice_angle(
    log_kernel: Callable[[float], float],
    rng: np.random.Generator,
    max_steps: int = 10_000,
    *,
    return_evaluations: bool = False,
) -> float | tuple[float, int]:
    log_y = log_kernel(0.0) + math.log(rng.random())
    theta = rng.uniform(0.0, TWO_PI)
    lower, upper = theta - TWO_PI, theta
    for step in range(1, max_steps + 1):
        if log_kernel(theta) >= log_y:
            return (theta, step + 1) if return_evaluations else theta
        if theta < 0.0:
            lower = theta
        else:
            upper = theta
        theta = rng.uniform(lower, upper)
    raise RuntimeError("geodesic slice bracket failed to contract")


def _run_stiefel_sweeps(
    state: Array,
    linear: Array,
    precision: Array,
    rng: np.random.Generator,
    sweeps: int,
    *,
    save_every: int = 0,
) -> tuple[Array, Array, int]:
    """Advance one geodesic-slice chain and optionally retain regular sweeps."""
    xmat = np.array(state, dtype=float, copy=True)
    p, qcols = xmat.shape
    a = 0.5 * (precision + precision.T)
    saved: list[Array] = []
    evaluations = 0
    for sweep in range(sweeps):
        for j in rng.permutation(qcols):
            v = rng.normal(size=p)
            v -= xmat @ (xmat.T @ v)
            norm_v = float(np.linalg.norm(v))
            while norm_v < 1e-12:
                v = rng.normal(size=p)
                v -= xmat @ (xmat.T @ v)
                norm_v = float(np.linalg.norm(v))
            v /= norm_v
            old = xmat[:, j].copy()
            log_kernel = _column_log_kernel_factory(old, v, linear[:, j], a)
            theta, used = _shrinkage_slice_angle(
                log_kernel, rng, return_evaluations=True
            )
            evaluations += used
            xmat[:, j] = old * math.cos(theta) + v * math.sin(theta)
        xmat = polar_factor(xmat)
        if save_every and (sweep + 1) % save_every == 0:
            saved.append(xmat.copy())
    draws = (
        np.stack(saved, axis=0)
        if saved
        else np.empty((0, p, qcols), dtype=float)
    )
    return xmat, draws, evaluations


def sample_matrix_exponential_family(
    n: int,
    linear: Array,
    precision: Array,
    initial: Array,
    rng: np.random.Generator,
    burnin_sweeps: int,
    thin_sweeps: int,
) -> Array:
    """Geodesic-slice MCMC for exp(tr(F'X)-tr(X'AX)) on V_q(R^p)."""
    state, _, _ = _run_stiefel_sweeps(
        initial, linear, precision, rng, burnin_sweeps
    )
    _, out, _ = _run_stiefel_sweeps(
        state,
        linear,
        precision,
        rng,
        n * thin_sweeps,
        save_every=thin_sweeps,
    )
    if out.shape[0] != n:
        raise AssertionError(f"saved {out.shape[0]} matrix draws; expected {n}")
    return out


def _split_rhat(chains: Array) -> float:
    """Conventional split-Rhat for a scalar diagnostic."""
    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or min(values.shape) < 4:
        return math.nan
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    if within <= np.finfo(float).tiny:
        return 1.0
    between = half * float(np.var(np.mean(split, axis=1), ddof=1))
    variance = (half - 1.0) * within / half + between / half
    return float(math.sqrt(max(variance / within, 0.0)))


def _autocorrelation(values: Array) -> Array:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    n = centered.size
    if n < 2 or float(centered @ centered) <= np.finfo(float).tiny:
        return np.ones(n)
    transformed = np.fft.rfft(centered, n=2 * n)
    covariance = np.fft.irfft(transformed * np.conj(transformed))[:n]
    covariance /= np.arange(n, 0, -1)
    return covariance / covariance[0]


def _multi_chain_ess(chains: Array) -> float:
    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[1] < 4:
        return float(values.size)
    correlations = np.stack([_autocorrelation(chain) for chain in values])
    mean_correlation = correlations.mean(axis=0)
    paired_sum = 0.0
    for lag in range(1, mean_correlation.size - 1, 2):
        pair = float(mean_correlation[lag] + mean_correlation[lag + 1])
        if pair <= 0.0:
            break
        paired_sum += pair
    return float(min(values.size, values.size / max(1.0 + 2.0 * paired_sum, 1.0)))


def _decorrelation_lag(
    diagnostic_chains: Sequence[Array], threshold: float, max_lag: int
) -> tuple[int, bool]:
    autocorrelations = [
        np.stack([_autocorrelation(chain) for chain in diagnostic]).mean(axis=0)
        for diagnostic in diagnostic_chains
    ]
    available = min(max_lag, min(ac.size for ac in autocorrelations) - 1)
    for lag in range(1, max(available - 2, 1)):
        if all(
            np.all(np.abs(ac[lag : lag + 3]) <= threshold)
            for ac in autocorrelations
        ):
            return lag, True
    return max(1, available), False


def _matrix_log_kernel_values(samples: Array, linear: Array, precision: Array) -> Array:
    linear_part = np.einsum("pq,npq->n", linear, samples)
    quadratic = np.einsum("npi,pr,nri->n", samples, precision, samples)
    return linear_part - quadratic


def sample_diagnosed_matrix_exponential_family(
    n: int,
    linear: Array,
    precision: Array,
    initial: Array,
    rng: np.random.Generator,
    alignment_function: Callable[[Array], Array],
    args: argparse.Namespace,
) -> tuple[Array, dict]:
    """Five-chain, bounded and diagnosed geodesic-slice sampler.

    Chains 0--3 are written first for training; chain 4 is written last as an
    independently initialized held-out block.
    """
    n_chains = 5
    if n % n_chains:
        raise ValueError("MCMC sample counts must be divisible by five")
    p, qcols = initial.shape
    chain_seeds = rng.integers(0, (1 << 63) - 1, size=n_chains, dtype=np.int64)
    chain_rngs = [np.random.default_rng(int(seed)) for seed in chain_seeds]
    states = [np.array(initial, copy=True)]
    states.extend(sample_macg(n_chains - 1, qcols, np.eye(p), rng))
    evaluations = 0

    for chain in range(n_chains):
        states[chain], _, used = _run_stiefel_sweeps(
            states[chain],
            linear,
            precision,
            chain_rngs[chain],
            args.burnin_sweeps,
        )
        evaluations += used
    discarded_sweeps = args.burnin_sweeps
    diagnostic_alignment: Array | None = None
    diagnostic_kernel: Array | None = None
    diagnostic_rhat = math.inf
    converged = False

    while discarded_sweeps < args.max_burnin_sweeps:
        diagnostic_batch = min(
            args.diagnostic_sweeps, args.max_burnin_sweeps - discarded_sweeps
        )
        if diagnostic_batch < 4:
            break
        alignment_rows = []
        kernel_rows = []
        for chain in range(n_chains):
            states[chain], trace, used = _run_stiefel_sweeps(
                states[chain],
                linear,
                precision,
                chain_rngs[chain],
                diagnostic_batch,
                save_every=1,
            )
            evaluations += used
            alignment_rows.append(alignment_function(trace))
            kernel_rows.append(_matrix_log_kernel_values(trace, linear, precision))
        discarded_sweeps += diagnostic_batch
        diagnostic_alignment = np.stack(alignment_rows)
        diagnostic_kernel = np.stack(kernel_rows)
        diagnostic_rhat = max(
            _split_rhat(diagnostic_alignment), _split_rhat(diagnostic_kernel)
        )
        if math.isfinite(diagnostic_rhat) and diagnostic_rhat <= args.rhat_threshold:
            converged = True
            break

    if diagnostic_alignment is None or diagnostic_kernel is None:
        raise AssertionError("MCMC diagnostics were not evaluated")
    thin, decorrelated = _decorrelation_lag(
        (diagnostic_alignment, diagnostic_kernel),
        args.max_saved_autocorrelation,
        min(args.max_thin_sweeps, args.diagnostic_sweeps // 2),
    )
    thin = max(args.thin_sweeps, thin)

    per_chain = n // n_chains
    retained = []
    retained_alignment = []
    retained_kernel = []
    for chain in range(n_chains):
        states[chain], draws, used = _run_stiefel_sweeps(
            states[chain],
            linear,
            precision,
            chain_rngs[chain],
            per_chain * thin,
            save_every=thin,
        )
        evaluations += used
        retained.append(draws)
        retained_alignment.append(alignment_function(draws))
        retained_kernel.append(_matrix_log_kernel_values(draws, linear, precision))

    alignment_chains = np.stack(retained_alignment)
    kernel_chains = np.stack(retained_kernel)
    ess = min(_multi_chain_ess(alignment_chains), _multi_chain_ess(kernel_chains))
    retained_rhat = max(_split_rhat(alignment_chains), _split_rhat(kernel_chains))
    qc_pass = (
        converged
        and decorrelated
        and math.isfinite(retained_rhat)
        and retained_rhat <= max(args.rhat_threshold, 1.05)
        and ess >= args.minimum_ess_fraction * n
    )
    diagnostics = {
        "proposal_count": 0,
        "acceptance_rate": math.nan,
        "sampler_qc_status": "pass" if qc_pass else "warning",
        "burnin_sweeps": discarded_sweeps,
        "thin_sweeps": thin,
        "n_chains": n_chains,
        "effective_sample_size": ess,
        "rhat_max": retained_rhat,
        "diagnostic_rhat": diagnostic_rhat,
        "decorrelation_status": "pass" if decorrelated else "warning",
        "mcmc_qc_status": "pass" if qc_pass else "warning",
        "slice_evaluations": evaluations,
    }
    if args.strict_sampler_qc and not qc_pass:
        raise RuntimeError(
            "MCMC sampler QC failed: "
            f"diagnostic Rhat={diagnostic_rhat:.4g}, retained Rhat={retained_rhat:.4g}, "
            f"ESS={ess:.1f}, thin={thin}"
        )
    return np.concatenate(retained, axis=0), diagnostics


def projector(columns: Array) -> Array:
    return columns @ columns.T if columns.shape[1] else np.zeros((columns.shape[0], columns.shape[0]))


def rank_for(label: str, maximum: int) -> int:
    if maximum < 1:
        raise ValueError("maximum active rank must be positive")
    if label != "full":
        raise ValueError("Only the maximal identifiable rank is supported.")
    return maximum


def distinct_penalties(rank: int, mean_penalty: float) -> Array:
    """Deterministic positive, distinct penalties with a fixed arithmetic mean."""
    if rank < 1 or mean_penalty <= 0:
        raise ValueError("rank and mean_penalty must be positive")
    if rank == 1:
        return np.array([mean_penalty])
    weights = np.linspace(0.5, 1.5, rank)
    return mean_penalty * weights / weights.mean()


def trace_free_fisher_bingham_precision(
    basis: Array,
    modal_dim: int,
    linear_concentration: float,
    anisotropy_fraction: float,
) -> tuple[Array, Array, Array]:
    """Return a gauge-fixed quadratic term orthogonal to Fisher curvature.

    The first ``modal_dim`` basis vectors span the Fisher modal frame.  The
    quadratic deviations in the remaining directions are distinct and have
    zero arithmetic mean.  The Fisher term therefore controls mean local
    curvature, while the quadratic term controls anisotropy.  A common scalar
    gauge shift makes the precision positive semidefinite without changing a
    spherical or Stiefel density.
    """
    p = basis.shape[0]
    transverse_dim = p - modal_dim
    if basis.shape != (p, p) or modal_dim < 1 or transverse_dim < 2:
        raise ValueError(
            "trace-free Fisher--Bingham anisotropy requires a square basis "
            "and at least two directions transverse to the Fisher modal frame"
        )
    if linear_concentration <= 0 or not 0.0 < anisotropy_fraction < 1.0:
        raise ValueError(
            "linear concentration must be positive and anisotropy_fraction must lie in (0, 1)"
        )

    # The small even perturbation avoids a zero weight when transverse_dim is
    # odd while retaining an approximately symmetric deterministic spectrum.
    positions = np.linspace(-1.0, 1.0, transverse_dim)
    weights = positions + 0.125 * positions**2
    weights -= weights.mean()
    weights /= np.max(np.abs(weights))
    deviations = 0.5 * anisotropy_fraction * linear_concentration * weights

    raw_eigenvalues = np.concatenate((np.zeros(modal_dim), deviations))
    gauge_eigenvalues = raw_eigenvalues - raw_eigenvalues.min()
    precision = (basis * gauge_eigenvalues[None, :]) @ basis.T
    active_eigenvalues = gauge_eigenvalues[gauge_eigenvalues > 1e-12]
    return precision, active_eigenvalues, deviations


def spectrum_string(values: Array) -> str:
    return ";".join(f"{value:.12g}" for value in values)


def alignment_values(
    samples: Array,
    distribution: str,
    basis: Array,
    qcols: int,
) -> Array:
    """Return a 0-at-uniform, 1-at-mode concentration statistic per draw."""
    p = basis.shape[0]
    if distribution == "wrapped_normal":
        return np.cos(samples).mean(axis=1)
    if distribution in {"vmf", "fisher_bingham"}:
        return samples @ basis[:, 0]
    if distribution in {"watson", "bingham", "acg"}:
        squared = (samples @ basis[:, 0]) ** 2
        return (p * squared - 1.0) / (p - 1.0)
    target = basis[:, :qcols]
    if distribution in {"matrix_fisher", "matrix_fisher_bingham"}:
        return np.einsum("pq,npq->n", target, samples) / qcols
    if distribution in {"matrix_bingham", "macg"}:
        overlap = np.square(
            np.einsum("pi,npj->nij", target, samples)
        ).sum(axis=(1, 2)) / qcols
        return (p * overlap - qcols) / (p - qcols)
    raise ValueError(f"no alignment statistic for {distribution}")


@lru_cache(maxsize=None)
def _beta_quadrature(alpha: float, beta: float, order: int = 160) -> tuple[Array, Array]:
    """Gauss--Jacobi nodes and normalized weights for Beta(alpha,beta)."""
    nodes, weights = roots_jacobi(order, beta - 1.0, alpha - 1.0)
    u = 0.5 * (nodes + 1.0)
    weights = weights / weights.sum()
    return np.asarray(u), np.asarray(weights)


def _watson_alignment(p: int, kappa: float) -> float:
    u, weights = _beta_quadrature(0.5, 0.5 * (p - 1))
    log_weights = np.log(weights) + kappa * u
    log_weights -= log_weights.max()
    tilted = np.exp(log_weights)
    mean_u = float((tilted * u).sum() / tilted.sum())
    return (p * mean_u - 1.0) / (p - 1.0)


def _acg_alignment(p: int, ratio: float) -> float:
    u, weights = _beta_quadrature(0.5, 0.5 * (p - 1))
    modal_fraction = ratio * u / (1.0 + (ratio - 1.0) * u)
    mean_u = float(weights @ modal_fraction)
    return (p * mean_u - 1.0) / (p - 1.0)


def _bracketed_positive_root(
    function: Callable[[float], float],
    target: float,
    initial_upper: float = 1.0,
) -> float:
    upper = initial_upper
    for _ in range(40):
        if function(upper) >= target:
            return float(brentq(lambda value: function(value) - target, 0.0, upper))
        upper *= 2.0
    raise RuntimeError(f"failed to bracket target alignment {target}")


def n_for_dim(args: argparse.Namespace, p: int) -> int:
    if args.sample_size_mode == "fixed":
        return args.n_samples
    value = args.n_samples * (p / args.reference_dim) ** args.sample_exponent
    return max(args.min_samples, int(round(value)))


def write_sample_csv(path: Path, samples: Array, distribution: str) -> None:
    if samples.ndim == 2:
        prefix = "theta" if distribution == "wrapped_normal" else "x"
        header = [f"{prefix}_{j}" for j in range(samples.shape[1])]
        flat = samples
    elif samples.ndim == 3:
        header = [
            f"x_{i}_{j}" for i in range(samples.shape[1]) for j in range(samples.shape[2])
        ]
        flat = samples.reshape(samples.shape[0], -1)
    else:
        raise ValueError("samples must be a 2D or 3D array")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(flat)


def validate_samples(samples: Array, distribution: str, atol: float = 2e-10) -> None:
    if not np.all(np.isfinite(samples)):
        raise FloatingPointError(f"{distribution}: non-finite sample")
    if distribution == "wrapped_normal":
        return
    if samples.ndim == 2:
        err = float(np.max(np.abs(np.linalg.norm(samples, axis=1) - 1.0)))
    else:
        gram = np.einsum("npi,npj->nij", samples, samples)
        eye = np.eye(samples.shape[2])[None, :, :]
        err = float(np.max(np.abs(gram - eye)))
    if err > atol:
        raise AssertionError(f"{distribution}: manifold error {err:.3g} exceeds {atol:.3g}")


def parse_csv_values(text: str, cast: Callable[[str], object]) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def cell_eligibility(distribution: str, p: int, matrix_cols: int) -> tuple[bool, str]:
    is_matrix = distribution.startswith("matrix_") or distribution == "macg"
    if is_matrix and p <= matrix_cols:
        return False, f"requires ambient_dim > matrix_cols ({p} <= {matrix_cols})"
    if distribution == "fisher_bingham" and p < 3:
        return False, "trace-free vector anisotropy requires p >= 3"
    if distribution == "matrix_fisher_bingham" and p - matrix_cols < 2:
        return False, "trace-free matrix anisotropy requires p - q >= 2"
    return True, "eligible"


def experiment_rows(args: argparse.Namespace) -> Iterable[tuple[str, int, str, str, float]]:
    targets = {"low": args.low_alignment, "high": args.high_alignment}
    for p in args.dimensions:
        for distribution in args.distributions:
            eligible, _ = cell_eligibility(distribution, p, args.matrix_cols)
            if not eligible:
                continue
            rank_labels = ("rank1",) if distribution in {"vmf", "watson"} else ("full",)
            for rank_label in rank_labels:
                for concentration_label in args.concentrations:
                    yield distribution, p, rank_label, concentration_label, targets[concentration_label]


def _direct_sampling_diagnostics(n: int) -> dict:
    return {
        "proposal_count": n,
        "acceptance_rate": 1.0,
        "sampler_qc_status": "pass",
        "burnin_sweeps": 0,
        "thin_sweeps": 0,
        "n_chains": 0,
        "effective_sample_size": float(n),
        "rhat_max": math.nan,
        "diagnostic_rhat": math.nan,
        "slice_evaluations": 0,
        "decorrelation_status": "not_applicable",
        "mcmc_qc_status": "not_applicable",
    }


def _sample_calibration_chains(
    n: int,
    linear: Array,
    precision: Array,
    initial: Array,
    rng: np.random.Generator,
    alignment_function: Callable[[Array], Array],
    args: argparse.Namespace,
) -> tuple[Array, dict]:
    """Small fixed-work panel used only to calibrate an MCMC family."""
    n_chains = 3
    per_chain = max(8, int(math.ceil(n / n_chains)))
    p, qcols = initial.shape
    initials = [np.array(initial, copy=True)]
    initials.extend(sample_macg(n_chains - 1, qcols, np.eye(p), rng))
    retained = []
    alignment_rows = []
    kernel_rows = []
    evaluations = 0
    for chain in range(n_chains):
        chain_rng = np.random.default_rng(int(rng.integers(0, (1 << 63) - 1)))
        state, _, used = _run_stiefel_sweeps(
            initials[chain],
            linear,
            precision,
            chain_rng,
            args.calibration_burnin_sweeps,
        )
        evaluations += used
        _, draws, used = _run_stiefel_sweeps(
            state,
            linear,
            precision,
            chain_rng,
            per_chain * args.calibration_thin_sweeps,
            save_every=args.calibration_thin_sweeps,
        )
        evaluations += used
        retained.append(draws)
        alignment_rows.append(alignment_function(draws))
        kernel_rows.append(_matrix_log_kernel_values(draws, linear, precision))
    alignment_chains = np.stack(alignment_rows)
    kernel_chains = np.stack(kernel_rows)
    diagnostics = {
        "proposal_count": 0,
        "acceptance_rate": math.nan,
        "sampler_qc_status": "calibration_pilot",
        "burnin_sweeps": args.calibration_burnin_sweeps,
        "thin_sweeps": args.calibration_thin_sweeps,
        "n_chains": n_chains,
        "effective_sample_size": min(
            _multi_chain_ess(alignment_chains), _multi_chain_ess(kernel_chains)
        ),
        "rhat_max": max(_split_rhat(alignment_chains), _split_rhat(kernel_chains)),
        "diagnostic_rhat": math.nan,
        "decorrelation_status": "calibration_pilot",
        "mcmc_qc_status": "calibration_pilot",
        "slice_evaluations": evaluations,
    }
    return np.concatenate(retained, axis=0)[:n], diagnostics


def _generate_at_strength(
    args: argparse.Namespace,
    distribution: str,
    p: int,
    rank_label: str,
    concentration_label: str,
    strength: float,
    basis: Array,
    rng: np.random.Generator,
    n: int,
    *,
    sampling_mode: str = "final",
) -> tuple[Array, dict]:
    qcols = min(args.matrix_cols, p - 1)
    mu = basis[:, 0]
    vector_max_rank = p - 1
    matrix_max_rank = p - qcols
    is_matrix = distribution.startswith("matrix_") or distribution == "macg"

    if rank_label == "rank1":
        active_rank = 1
    elif distribution == "wrapped_normal":
        active_rank = rank_for(rank_label, p)
    elif is_matrix:
        active_rank = rank_for(rank_label, matrix_max_rank)
    else:
        active_rank = rank_for(rank_label, vector_max_rank)

    linear_rank = 0
    model_factor_rank = active_rank
    modal_dim = p - active_rank
    modal_frame_dim = modal_dim
    quadratic_nullity = p
    ratio = 1.0
    natural = 0.0
    linear_concentration = 0.0
    quadratic_concentration = 0.0
    quadratic_spectrum = ""
    iid = True
    sampling_diagnostics = _direct_sampling_diagnostics(n)
    max_proposals = args.max_proposals_per_sample * n

    def mcmc_samples(linear: Array, precision: Array, initial: Array) -> tuple[Array, dict]:
        def score(draws: Array) -> Array:
            values = draws[:, :, 0] if distribution == "fisher_bingham" else draws
            return alignment_values(values, distribution, basis, qcols)

        if sampling_mode == "calibration":
            return _sample_calibration_chains(
                n, linear, precision, initial, rng, score, args
            )
        return sample_diagnosed_matrix_exponential_family(
            n, linear, precision, initial, rng, score, args
        )

    if distribution == "wrapped_normal":
        natural = strength * active_rank
        quadratic_concentration = natural
        samples = sample_wrapped_normal(
            n, basis, active_rank, strength, rng, args.inactive_wrapped_variance, args.angle_interval
        )
        sampling_diagnostics = _direct_sampling_diagnostics(n)
        sampler = "direct_wrapped_gaussian"
        support = f"torus_T^{p}"
        density = "wrap(N(0,Sigma)); active precision=concentration_strength*active_rank"

    elif distribution == "vmf":
        natural = strength * (p - 1)
        linear_concentration = natural
        linear_rank = 1
        modal_dim = 1
        modal_frame_dim = 1
        samples, sampling_diagnostics = sample_vmf(
            n,
            mu,
            natural,
            rng,
            max_proposals=max_proposals,
            return_diagnostics=True,
        )
        sampler = "wood_ulrich_rejection"
        support = f"sphere_S^{p-1}"
        density = "exp(kappa*mu.T@x)"

    elif distribution == "watson":
        natural = 0.5 * strength * (p - 1)  # local axial curvature is 2*kappa
        quadratic_concentration = natural
        quadratic_spectrum = spectrum_string(np.full(p - 1, natural))
        modal_dim = 1
        modal_frame_dim = 1
        quadratic_nullity = 1
        a = natural * (np.eye(p) - np.outer(mu, mu))
        samples, sampling_diagnostics = sample_bingham(
            n,
            a,
            rng,
            max_proposals=max_proposals,
            return_diagnostics=True,
        )
        sampler = "bacg_rejection"
        support = f"sphere_S^{p-1}"
        density = "exp(kappa*(mu.T@x)^2)"

    elif distribution in {"bingham", "fisher_bingham", "acg"}:
        active = basis[:, 1 : 1 + active_rank]
        p_active = projector(active)
        natural = strength * active_rank
        if distribution == "bingham":
            penalties = distinct_penalties(active_rank, 0.5 * natural)
            a = (active * penalties[None, :]) @ active.T
            quadratic_concentration = float(penalties.mean())
            quadratic_spectrum = spectrum_string(penalties)
            # After the Bingham gauge shift, a distinct spectrum requires
            # p-1 factor columns rather than the former rank-one dual.
            model_factor_rank = active_rank
            quadratic_nullity = p - active_rank
            samples, sampling_diagnostics = sample_bingham(
                n,
                a,
                rng,
                max_proposals=max_proposals,
                return_diagnostics=True,
            )
            sampler = "bacg_rejection"
            density = "exp(-x.T@A@x); distinct active eigenvalues"
        elif distribution == "fisher_bingham":
            linear_rank = 1
            kappa = strength * (p - 1)
            a, penalties, deviations = trace_free_fisher_bingham_precision(
                basis,
                modal_dim=1,
                linear_concentration=kappa,
                anisotropy_fraction=args.fisher_bingham_anisotropy,
            )
            linear_concentration = kappa
            quadratic_concentration = float(np.max(np.abs(deviations)))
            quadratic_spectrum = spectrum_string(penalties)
            active_rank = penalties.size
            model_factor_rank = penalties.size
            quadratic_nullity = p - model_factor_rank
            matrix_samples, sampling_diagnostics = mcmc_samples(
                (kappa * mu)[:, None], a, mu[:, None]
            )
            samples = matrix_samples[:, :, 0]
            iid = False
            sampler = "five_chain_geodesic_slice_mcmc"
            density = (
                "exp(kappa*mu.T@x-x.T@A@x); mean curvature matched to vMF/Bingham; "
                "trace-free quadratic anisotropy"
            )
            natural = kappa
        else:
            ratio = 1.0 + strength * active_rank / modal_dim
            quadratic_concentration = math.log(ratio)
            omega = np.eye(p) + (ratio - 1.0) * p_active
            samples = sample_acg(n, omega, rng)
            sampler = "normalized_gaussian"
            density = "|Omega|^(1/2)*(x.T@Omega@x)^(-p/2)"
            natural = math.log(ratio)
            model_factor_rank = modal_dim
            modal_frame_dim = modal_dim
            quadratic_nullity = modal_dim
            sampling_diagnostics = _direct_sampling_diagnostics(n)
        support = f"sphere_S^{p-1}"

    elif distribution in {"matrix_fisher", "matrix_bingham", "matrix_fisher_bingham", "macg"}:
        target = basis[:, :qcols]
        active = basis[:, qcols : qcols + active_rank]
        p_active = projector(active)
        a_natural = strength * active_rank
        penalties = distinct_penalties(active_rank, 0.5 * a_natural)
        a = (active * penalties[None, :]) @ active.T
        matrix_linear_rank = qcols
        f_natural = strength * (p - qcols)
        f = np.zeros((p, qcols))
        f[:, :matrix_linear_rank] = f_natural * target[:, :matrix_linear_rank]
        support = f"Stiefel_V_{qcols}(R^{p})"

        if distribution == "matrix_fisher":
            active_rank = matrix_linear_rank
            model_factor_rank = matrix_linear_rank
            linear_rank = matrix_linear_rank
            modal_dim = qcols - matrix_linear_rank
            modal_frame_dim = qcols
            natural = f_natural
            linear_concentration = f_natural
            quadratic_nullity = p
            samples, sampling_diagnostics = mcmc_samples(
                f, np.zeros((p, p)), target
            )
            iid = False
            sampler = "five_chain_geodesic_slice_mcmc"
            density = "exp(tr(F.T@X))"
        elif distribution == "matrix_bingham":
            linear_rank = 0
            natural = a_natural
            quadratic_concentration = float(penalties.mean())
            quadratic_spectrum = spectrum_string(penalties)
            model_factor_rank = active_rank
            modal_frame_dim = qcols
            quadratic_nullity = p - model_factor_rank
            if sampling_mode == "calibration":
                samples, sampling_diagnostics = mcmc_samples(
                    np.zeros((p, qcols)), a, target
                )
                iid = False
                sampler = "three_chain_geodesic_slice_calibration"
            else:
                try:
                    samples, sampling_diagnostics = sample_matrix_bingham(
                        n,
                        qcols,
                        a,
                        rng,
                        max_proposals=max_proposals,
                        return_diagnostics=True,
                    )
                    sampler = "matrix_bacg_rejection"
                except SamplingEfficiencyError:
                    samples, sampling_diagnostics = mcmc_samples(
                        np.zeros((p, qcols)), a, target
                    )
                    iid = False
                    sampler = "five_chain_geodesic_slice_mcmc_fallback"
            density = "exp(-tr(X.T@A@X)); distinct active eigenvalues"
        elif distribution == "matrix_fisher_bingham":
            linear_rank = matrix_linear_rank
            natural = f_natural
            linear_concentration = f_natural
            a, penalties, deviations = trace_free_fisher_bingham_precision(
                basis,
                modal_dim=qcols,
                linear_concentration=f_natural,
                anisotropy_fraction=args.fisher_bingham_anisotropy,
            )
            quadratic_concentration = float(np.max(np.abs(deviations)))
            quadratic_spectrum = spectrum_string(penalties)
            active_rank = penalties.size
            model_factor_rank = penalties.size
            modal_dim = qcols
            modal_frame_dim = qcols
            quadratic_nullity = p - model_factor_rank
            samples, sampling_diagnostics = mcmc_samples(f, a, target)
            iid = False
            sampler = "five_chain_geodesic_slice_mcmc"
            density = (
                "exp(tr(F.T@X)-tr(X.T@A@X)); mean curvature matched to matrix Fisher/Bingham; "
                "trace-free quadratic anisotropy"
            )
        else:
            ratio = 1.0 + strength * active_rank / modal_dim
            omega = np.eye(p) + (ratio - 1.0) * p_active
            natural = math.log(ratio)
            quadratic_concentration = natural
            model_factor_rank = modal_dim
            modal_frame_dim = modal_dim
            quadratic_nullity = modal_dim
            samples = sample_macg(n, qcols, omega, rng)
            sampling_diagnostics = _direct_sampling_diagnostics(n)
            sampler = "matrix_normal_polar_factor"
            density = "|Omega|^(q/2)*|X.T@Omega@X|^(-p/2)"
    else:
        raise ValueError(f"unsupported distribution: {distribution}")

    details = dict(
        matrix_cols=qcols if is_matrix else 0,
        active_rank=active_rank,
        model_factor_rank=model_factor_rank,
        linear_rank=linear_rank,
        modal_dim=modal_dim,
        modal_frame_dim=modal_frame_dim,
        quadratic_nullity=quadratic_nullity,
        natural_concentration=natural,
        linear_concentration=linear_concentration,
        quadratic_concentration=quadratic_concentration,
        quadratic_spectrum=quadratic_spectrum,
        acg_precision_ratio=ratio,
        iid=iid,
        sampler=sampler,
        burnin_sweeps=int(sampling_diagnostics.get("burnin_sweeps", 0)),
        thin_sweeps=int(sampling_diagnostics.get("thin_sweeps", 0)),
        n_chains=int(sampling_diagnostics.get("n_chains", 0)),
        effective_sample_size=float(sampling_diagnostics.get("effective_sample_size", n)),
        rhat_max=float(sampling_diagnostics.get("rhat_max", math.nan)),
        diagnostic_rhat=float(sampling_diagnostics.get("diagnostic_rhat", math.nan)),
        slice_evaluations=int(sampling_diagnostics.get("slice_evaluations", 0)),
        decorrelation_status=str(sampling_diagnostics.get("decorrelation_status", "not_applicable")),
        mcmc_qc_status=str(sampling_diagnostics.get("mcmc_qc_status", "not_applicable")),
        proposal_count=int(sampling_diagnostics.get("proposal_count", 0)),
        acceptance_rate=float(sampling_diagnostics.get("acceptance_rate", math.nan)),
        sampler_qc_status=str(sampling_diagnostics.get("sampler_qc_status", "pass")),
        support=support,
        density_kernel=density,
    )
    return samples, details


def calibrate_strength(
    args: argparse.Namespace,
    distribution: str,
    p: int,
    rank_label: str,
    concentration_label: str,
    target_alignment: float,
    basis: Array,
) -> dict:
    """Calibrate the family's scalar multiplier to a common modal alignment."""
    started = time.perf_counter()
    if distribution == "wrapped_normal":
        precision = -1.0 / (2.0 * math.log(target_alignment))
        result = {
            "strength": precision / p,
            "method": "analytic_wrapped_resultant",
            "status": "pass",
            "iterations": 0,
            "draws": 0,
            "estimated_alignment": target_alignment,
            "mcse": 0.0,
        }
    elif distribution == "vmf":
        kappa = _bracketed_positive_root(
            lambda value: _vmf_mean_ratio(p, value), target_alignment
        )
        result = {
            "strength": kappa / (p - 1),
            "method": "analytic_bessel_mean_ratio",
            "status": "pass",
            "iterations": 0,
            "draws": 0,
            "estimated_alignment": target_alignment,
            "mcse": 0.0,
        }
    elif distribution == "watson":
        kappa = _bracketed_positive_root(
            lambda value: _watson_alignment(p, value), target_alignment
        )
        result = {
            "strength": 2.0 * kappa / (p - 1),
            "method": "gauss_jacobi_beta_quadrature",
            "status": "pass",
            "iterations": 0,
            "draws": 0,
            "estimated_alignment": target_alignment,
            "mcse": 0.0,
        }
    elif distribution == "acg":
        ratio = _bracketed_positive_root(
            lambda value: _acg_alignment(p, 1.0 + value), target_alignment
        ) + 1.0
        result = {
            "strength": (ratio - 1.0) / (p - 1),
            "method": "gauss_jacobi_gaussian_ratio_quadrature",
            "status": "pass",
            "iterations": 0,
            "draws": 0,
            "estimated_alignment": target_alignment,
            "mcse": 0.0,
        }
    else:
        evaluations: list[dict] = []
        calibration_seed = _stable_seed(
            args.seed,
            "alignment_calibration",
            distribution,
            p,
            rank_label,
            concentration_label,
        )

        def evaluate(strength: float) -> dict:
            evaluation_index = len(evaluations)
            candidate_rng = np.random.default_rng(
                _stable_seed(calibration_seed, evaluation_index)
            )
            samples, details = _generate_at_strength(
                args,
                distribution,
                p,
                rank_label,
                concentration_label,
                strength,
                basis,
                candidate_rng,
                args.calibration_draws,
                sampling_mode="calibration",
            )
            values = alignment_values(
                samples,
                distribution,
                basis,
                details["matrix_cols"],
            )
            ess = max(1.0, min(float(details["effective_sample_size"]), values.size))
            estimate = {
                "strength": float(strength),
                "alignment": float(values.mean()),
                "mcse": float(values.std(ddof=1) / math.sqrt(ess)),
                "draws": int(values.size),
                "pilot_rhat": float(details["rhat_max"]),
            }
            evaluations.append(estimate)
            return estimate

        status = "pass"
        error_message = ""
        try:
            lower = evaluate(1e-6)
            upper_strength = 1.0
            upper = evaluate(upper_strength)
            for _ in range(args.calibration_max_bracket_steps):
                if upper["alignment"] >= target_alignment:
                    break
                upper_strength *= 2.0
                upper = evaluate(upper_strength)
            else:
                status = "failed_to_bracket"

            if status == "pass":
                for _ in range(args.calibration_iterations):
                    middle_strength = 0.5 * (lower["strength"] + upper["strength"])
                    middle = evaluate(middle_strength)
                    if middle["alignment"] < target_alignment:
                        lower = middle
                    else:
                        upper = middle
        except Exception as error:
            status = "pilot_error"
            error_message = f"{type(error).__name__}: {error}"

        finite = [
            item for item in evaluations if math.isfinite(item["alignment"])
        ]
        if not finite:
            raise RuntimeError(
                f"{distribution} p={p}: concentration calibration produced no finite pilot"
            )
        best = min(finite, key=lambda item: abs(item["alignment"] - target_alignment))
        if (
            status == "pass"
            and abs(best["alignment"] - target_alignment)
            > args.calibration_alignment_tolerance
        ):
            status = "alignment_tolerance_warning"
        if args.strict_sampler_qc and status != "pass":
            raise RuntimeError(
                f"{distribution} p={p}: calibration {status}; {error_message}"
            )
        result = {
            "strength": best["strength"],
            "method": "bounded_pilot_bisection",
            "status": status,
            "iterations": len(evaluations),
            "draws": sum(item["draws"] for item in evaluations),
            "estimated_alignment": best["alignment"],
            "mcse": best["mcse"],
        }
    result["seconds"] = time.perf_counter() - started
    return result


def generate_one(
    args: argparse.Namespace,
    distribution: str,
    p: int,
    rank_label: str,
    concentration_label: str,
    target_alignment: float,
    basis: Array,
    rng: np.random.Generator,
    n: int,
) -> tuple[Array, dict]:
    calibration = calibrate_strength(
        args,
        distribution,
        p,
        rank_label,
        concentration_label,
        target_alignment,
        basis,
    )
    samples, details = _generate_at_strength(
        args,
        distribution,
        p,
        rank_label,
        concentration_label,
        calibration["strength"],
        basis,
        rng,
        n,
        sampling_mode="final",
    )
    values = alignment_values(
        samples, distribution, basis, details["matrix_cols"]
    )
    ess = max(1.0, min(float(details["effective_sample_size"]), values.size))
    alignment_mcse = float(values.std(ddof=1) / math.sqrt(ess))
    allowed_alignment_error = max(
        args.calibration_alignment_tolerance,
        3.0 * alignment_mcse,
        3.0 * float(calibration["mcse"]),
    )
    alignment_qc_status = (
        "pass"
        if abs(float(values.mean()) - target_alignment) <= allowed_alignment_error
        else "warning"
    )
    if alignment_qc_status != "pass" and details["sampler_qc_status"] == "pass":
        details["sampler_qc_status"] = "warning"
    details.update(
        concentration_strength=float(calibration["strength"]),
        target_alignment=float(target_alignment),
        observed_alignment=float(values.mean()),
        alignment_mcse=alignment_mcse,
        alignment_qc_status=alignment_qc_status,
        calibration_method=calibration["method"],
        calibration_status=calibration["status"],
        calibration_iterations=int(calibration["iterations"]),
        calibration_draws=int(calibration["draws"]),
        calibration_estimated_alignment=float(calibration["estimated_alignment"]),
        calibration_mcse=float(calibration["mcse"]),
        calibration_seconds=float(calibration["seconds"]),
    )
    return samples, details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate matched samples from directional and matrix-directional distributions."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("overview_paper/directional_samples_protocol_v2"))
    parser.add_argument("--dimensions", default="2,4,8,16,32,64")
    parser.add_argument("--distributions", default="all")
    parser.add_argument("--concentrations", default="low,high", help="Subset of low,high")
    parser.add_argument("--low-alignment", type=float, default=0.20,
                        help="Normalized modal-alignment target for the low regime")
    parser.add_argument("--high-alignment", type=float, default=0.80,
                        help="Normalized modal-alignment target for the high regime")
    parser.add_argument(
        "--fisher-bingham-anisotropy",
        type=float,
        default=0.5,
        help=(
            "Dimension-independent fraction of mean Fisher curvature used for "
            "the peak trace-free quadratic deviation; must lie in (0, 1)"
        ),
    )
    parser.add_argument("--matrix-cols", type=int, default=2,
                        help="q in the p-by-q Stiefel sample (must be < every p)")
    parser.add_argument("--n-samples", type=int, default=400,
                        help="Fixed n, or n at --reference-dim in dimension_power mode")
    parser.add_argument("--sample-size-mode", choices=("fixed", "dimension_power"), default="fixed")
    parser.add_argument("--sample-exponent", type=float, default=1.0,
                        help="n(p)=n_samples*(p/reference_dim)^exponent")
    parser.add_argument("--reference-dim", type=float, default=4.0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--burnin-sweeps", type=int, default=250,
                        help="Initial MCMC warm-up before convergence diagnostics")
    parser.add_argument("--thin-sweeps", type=int, default=1,
                        help="Minimum MCMC sweeps between retained samples")
    parser.add_argument("--diagnostic-sweeps", type=int, default=400,
                        help="MCMC sweeps per convergence/autocorrelation diagnostic batch")
    parser.add_argument("--max-burnin-sweeps", type=int, default=2250,
                        help="Bound on discarded sweeps per MCMC chain")
    parser.add_argument("--max-thin-sweeps", type=int, default=200,
                        help="Bound on adaptively selected MCMC thinning")
    parser.add_argument("--rhat-threshold", type=float, default=1.01)
    parser.add_argument("--max-saved-autocorrelation", type=float, default=0.05)
    parser.add_argument("--minimum-ess-fraction", type=float, default=0.5)
    parser.add_argument("--strict-sampler-qc", action="store_true")
    parser.add_argument("--calibration-draws", type=int, default=240)
    parser.add_argument("--calibration-iterations", type=int, default=5)
    parser.add_argument("--calibration-max-bracket-steps", type=int, default=8)
    parser.add_argument("--calibration-alignment-tolerance", type=float, default=0.05)
    parser.add_argument("--calibration-burnin-sweeps", type=int, default=250)
    parser.add_argument("--calibration-thin-sweeps", type=int, default=10)
    parser.add_argument("--max-proposals-per-sample", type=int, default=5_000,
                        help="Bound for every rejection sampler before failure/fallback")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--orientation", choices=("random", "canonical"), default="random")
    parser.add_argument("--angle-interval", choices=("zero_two_pi", "minus_pi_pi"),
                        default="minus_pi_pi")
    parser.add_argument("--inactive-wrapped-variance", type=float, default=25.0,
                        help="Large variance makes inactive wrapped coordinates nearly uniform")
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Explicitly replace files in a nonempty output directory",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        "--no_overwrite",
        dest="overwrite",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.dimensions = parse_csv_values(args.dimensions, int)
    args.concentrations = parse_csv_values(args.concentrations, str)
    if args.distributions == "all":
        args.distributions = list(ALL_DISTRIBUTIONS)
    else:
        args.distributions = parse_csv_values(args.distributions, str)
    if not args.dimensions or min(args.dimensions) < 2:
        parser.error("all dimensions must be >= 2")
    unknown = set(args.distributions) - set(ALL_DISTRIBUTIONS)
    if unknown:
        parser.error(f"unknown distributions: {sorted(unknown)}")
    if not set(args.concentrations) <= {"low", "high"}:
        parser.error("--concentrations must contain only low and/or high")
    if args.matrix_cols < 1:
        parser.error("--matrix-cols must be positive")
    if args.n_samples < 1 or args.min_samples < 1:
        parser.error("sample counts must be positive")
    if args.reference_dim <= 0:
        parser.error("reference dimension must be positive")
    if not 0.0 < args.low_alignment < args.high_alignment < 1.0:
        parser.error("require 0 < --low-alignment < --high-alignment < 1")
    if not 0.0 < args.fisher_bingham_anisotropy < 1.0:
        parser.error("--fisher-bingham-anisotropy must lie strictly between zero and one")
    if args.burnin_sweeps < 0 or args.thin_sweeps < 1:
        parser.error("burn-in must be nonnegative and thinning must be positive")
    positive_counts = (
        args.diagnostic_sweeps,
        args.max_burnin_sweeps,
        args.max_thin_sweeps,
        args.calibration_draws,
        args.calibration_iterations,
        args.calibration_max_bracket_steps,
        args.calibration_burnin_sweeps,
        args.calibration_thin_sweeps,
        args.max_proposals_per_sample,
    )
    if any(value < 1 for value in positive_counts):
        parser.error("MCMC, calibration, and proposal count settings must be positive")
    if args.max_burnin_sweeps < args.burnin_sweeps + args.diagnostic_sweeps:
        parser.error("--max-burnin-sweeps must allow at least one diagnostic batch")
    if not 1.0 <= args.rhat_threshold < 2.0:
        parser.error("--rhat-threshold must lie in [1, 2)")
    if not 0.0 < args.max_saved_autocorrelation < 1.0:
        parser.error("--max-saved-autocorrelation must lie in (0, 1)")
    if not 0.0 < args.minimum_ess_fraction <= 1.0:
        parser.error("--minimum-ess-fraction must lie in (0, 1]")
    if not 0.0 < args.calibration_alignment_tolerance < 1.0:
        parser.error("--calibration-alignment-tolerance must lie in (0, 1)")
    mcmc_requested = bool(
        set(args.distributions)
        & {"fisher_bingham", "matrix_fisher", "matrix_fisher_bingham"}
    )
    if mcmc_requested:
        for p in args.dimensions:
            if n_for_dim(args, p) % 5:
                parser.error("MCMC cells require every generated sample count to be divisible by five")
    eligibility = [
        (distribution, p, *cell_eligibility(distribution, p, args.matrix_cols))
        for p in args.dimensions
        for distribution in args.distributions
    ]
    if not any(eligible for _, _, eligible, _ in eligibility):
        reasons = "; ".join(
            f"{distribution} p={p}: {reason}"
            for distribution, p, _, reason in eligibility
        )
        parser.error(f"no eligible experiment cells: {reasons}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_args(args, parser)

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(
            f"output directory {output} is nonempty; pass --overwrite only when replacement is intended"
        )
    output.mkdir(parents=True, exist_ok=True)

    # A completion marker is written only after every planned cell succeeds.
    # Remove a stale marker before an overwrite so an interrupted regeneration
    # cannot be mistaken for a complete benchmark sample set.
    completion_path = output / "sampling_complete.json"
    completion_path.unlink(missing_ok=True)

    config = vars(args).copy()
    config["output_dir"] = str(output)
    config["sampler_schema_version"] = SAMPLER_SCHEMA_VERSION
    config_path = output / "experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    manifest: list[ManifestRow] = []
    parameter_dir = output / "parameters"
    parameter_dir.mkdir(exist_ok=True)
    basis_seeds = {p: _stable_seed(args.seed, "basis", p) for p in args.dimensions}
    bases = {p: orthogonal_basis(p, basis_seeds[p], args.orientation) for p in args.dimensions}
    basis_files: dict[int, str] = {}
    for p, basis in bases.items():
        basis_name = f"basis_p{p}.csv"
        basis_path = parameter_dir / basis_name
        np.savetxt(
            basis_path,
            basis,
            delimiter=",",
            header=",".join(f"basis_{j}" for j in range(p)),
            comments="",
        )
        basis_files[p] = str(Path("parameters") / basis_name)
    jobs = list(experiment_rows(args))
    manifest_path = output / "manifest.csv"
    if not args.quiet:
        for p in args.dimensions:
            for distribution in args.distributions:
                eligible, reason = cell_eligibility(distribution, p, args.matrix_cols)
                if not eligible:
                    print(f"[skip] {distribution} p={p}: {reason}", flush=True)
    for job_index, (distribution, p, rank_label, concentration_label, target_alignment) in enumerate(jobs, 1):
        n = n_for_dim(args, p)
        seed = _stable_seed(args.seed, distribution, p, rank_label, concentration_label)
        rng = np.random.default_rng(seed)
        filename = (
            f"{distribution}__p{p}"
            + (f"_q{args.matrix_cols}" if distribution.startswith("matrix_") or distribution == "macg" else "")
            + f"__rank-{rank_label}__conc-{concentration_label}__n{n}.csv"
        )
        path = output / filename
        if not args.quiet:
            print(f"[{job_index:03d}/{len(jobs):03d}] {filename}", flush=True)
        started = time.perf_counter()
        samples, details = generate_one(
            args, distribution, p, rank_label, concentration_label, target_alignment,
            bases[p], rng, n,
        )
        validate_samples(samples, distribution)
        write_sample_csv(path, samples, distribution)
        elapsed = time.perf_counter() - started
        n_test = n // 5 if not n % 5 else max(1, int(round(0.2 * n)))
        n_train = n - n_test
        manifest.append(
            ManifestRow(
                sampler_schema_version=SAMPLER_SCHEMA_VERSION,
                distribution=distribution,
                ambient_dim=p,
                rank_label=rank_label,
                concentration_label=concentration_label,
                fisher_bingham_anisotropy=args.fisher_bingham_anisotropy,
                n_samples=n,
                n_train=n_train,
                n_test=n_test,
                split_index=n_train,
                seed=seed,
                basis_seed=basis_seeds[p],
                basis_filename=basis_files[p],
                orientation=args.orientation,
                elapsed_seconds=elapsed,
                filename=filename,
                **details,
            )
        )

        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(manifest[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in manifest)

    completion = {
        "sampler_schema_version": SAMPLER_SCHEMA_VERSION,
        "generated_cells": len(manifest),
        "planned_cells": len(jobs),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    if not args.quiet:
        print(f"Wrote {len(manifest)} sample files plus manifest/config to {output}")
        print("Note: Fisher--Bingham and matrix Fisher/FB rows are diagnosed MCMC (iid=false).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
