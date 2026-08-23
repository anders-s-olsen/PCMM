"""Additional real-valued PyTorch distributions for PCMM.

The vector Fisher--Bingham normalizer is evaluated by the one-dimensional
continuous-Euler quadrature of Chen and Tanaka (2020).  Matrix Fisher models
on V_2(R^p) use an exact one-dimensional Gauss--Jacobi/Bessel reduction by
default.  The other matrix models use a saddlepoint approximation calibrated
so that the uniform distribution has log normalizer zero.  Consequently,
matrix log densities are relative to normalized Haar measure.

Parameterizations
-----------------
VonMisesFisher
    exp(kappa * mu.T @ x), with kappa = softplus(raw_kappa).
FisherBingham
    exp(kappa * mu.T @ x - ||M.T @ x||^2).
MatrixFisher
    exp(tr(F.T @ X)), with either direct F or F = F_left @ F_right.T.
MatrixBingham
    exp(-tr(X.T @ M @ M.T @ X)).
MatrixFisherBingham
    exp(tr(F.T @ X) - tr(X.T @ M @ M.T @ X)).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.special import betaln, gammaln, hyp0f1, ive, logsumexp, roots_jacobi

from PCMM.PCMMtorchBaseModel import PCMMtorchBaseModel


@lru_cache(maxsize=None)
def _matrix_fisher_q2_quadrature(p: int, quadrature_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return radial nodes and normalized log weights for exact q=2 integration.

    If t is one coordinate of a uniform point on S^(p-1), its unnormalized
    density is (1-t^2)^((p-3)/2). Gauss--Jacobi quadrature therefore evaluates
    the remaining one-dimensional expectation directly. Arrays are immutable
    so callers cannot corrupt the process-wide cache.
    """
    if p < 3:
        raise ValueError('The q=2 Matrix Fisher normalizer requires p >= 3.')
    if quadrature_points < 8:
        raise ValueError('exact_quadrature_points should be at least 8.')
    alpha = (p - 3.0) / 2.0
    nodes, weights = roots_jacobi(quadrature_points, alpha, alpha)
    radial_nodes = np.sqrt(np.maximum(1.0 - nodes * nodes, 0.0))
    log_measure = betaln(0.5, alpha + 1.0)
    log_weights = np.log(weights) - log_measure
    radial_nodes.setflags(write=False)
    log_weights.setflags(write=False)
    return radial_nodes, log_weights


def _log_vmf_uniform_mgf_and_score(ambient_dimension: int, concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable log E[exp(k*x_1)] and its derivative for a uniform sphere.

    scipy.special.ive removes the exponentially growing part of the modified
    Bessel function. Exact zeros are handled analytically, which is also
    required for a well-defined uniform Matrix Fisher score.
    """
    concentration = np.asarray(concentration, dtype=np.float64)
    if np.any(concentration < 0) or np.any(~np.isfinite(concentration)):
        raise ValueError('Matrix Fisher singular values should be finite and non-negative.')
    order = ambient_dimension / 2.0 - 1.0
    log_mgf = np.zeros_like(concentration)
    score = np.zeros_like(concentration)

    # For small arguments and large Bessel order, ive can underflow even
    # though the normalized moment generating function is close to one.
    # The equivalent 0F1 representation is stable in that regime.
    small = (concentration > 0.0) & (concentration <= 50.0)
    if np.any(small):
        argument = np.square(concentration[small]) / 4.0
        shape = ambient_dimension / 2.0
        mgf = hyp0f1(shape, argument)
        next_mgf = hyp0f1(shape + 1.0, argument)
        if np.any(~np.isfinite(mgf)) or np.any(~np.isfinite(next_mgf)) or np.any(mgf <= 0.0):
            raise RuntimeError('Hypergeometric evaluation failed in the exact Matrix Fisher normalizer.')
        log_mgf[small] = np.log(mgf)
        score[small] = (concentration[small] / ambient_dimension * next_mgf / mgf)

    large = concentration > 50.0
    if np.any(large):
        large_concentration = concentration[large]
        scaled_bessel = ive(order, large_concentration)
        scaled_bessel_next = ive(order + 1.0, large_concentration)
        if np.any(~np.isfinite(scaled_bessel)) or np.any(~np.isfinite(scaled_bessel_next)) or np.any(scaled_bessel <= 0.0):
            raise RuntimeError('Scaled Bessel evaluation failed in the exact Matrix Fisher normalizer.')
        log_mgf[large] = (
            gammaln(ambient_dimension / 2.0)
            + order * (math.log(2.0) - np.log(large_concentration))
            + np.log(scaled_bessel)
            + large_concentration
        )
        score[large] = scaled_bessel_next / scaled_bessel
    return log_mgf, score


class _MatrixFisherQ2ExactLogNormalizer(torch.autograd.Function):
    """Exact normalized-Haar q=2 log normalizer with an analytic first score.

    For singular values ``s1, s2``, the reduction is

    ``Z = E[A_{p-1}(s1*R) A_{p-1}(s2*R)]``,

    where ``R=sqrt(1-T^2)``, ``T`` has density proportional to
    ``(1-T^2)^((p-3)/2)``, and ``A_m`` is the uniform-sphere vMF moment
    generating function.  Gauss--Jacobi handles the expectation; the saved
    Bessel ratios are its analytic derivatives with respect to ``s1, s2``.
    """

    @staticmethod
    def forward(ctx, singular_values: torch.Tensor, p: int, quadrature_points: int) -> torch.Tensor:
        if singular_values.ndim != 2 or singular_values.shape[1] != 2:
            raise ValueError('Expected Matrix Fisher singular values with shape (K, 2).')
        radial_nodes, log_weights = _matrix_fisher_q2_quadrature(int(p), int(quadrature_points))
        values = singular_values.detach().double().cpu().numpy()
        concentrations = values[:, :, None] * radial_nodes[None, None, :]
        log_mgf, bessel_score = _log_vmf_uniform_mgf_and_score(int(p) - 1, concentrations)
        log_integrands = log_weights[None, :] + log_mgf.sum(axis=1)
        log_normalizers = logsumexp(log_integrands, axis=1)
        posterior_weights = np.exp(log_integrands - log_normalizers[:, None])
        singular_scores = np.sum(posterior_weights[:, None, :] * radial_nodes[None, None, :] * bessel_score, axis=2)
        uniform_components = np.all(values == 0.0, axis=1)
        log_normalizers[uniform_components] = 0.0
        singular_scores[uniform_components] = 0.0
        if np.any(~np.isfinite(log_normalizers)) or np.any(~np.isfinite(singular_scores)):
            raise RuntimeError('Exact Matrix Fisher quadrature returned a non-finite result.')
        score_tensor = torch.as_tensor(singular_scores, dtype=singular_values.dtype, device=singular_values.device)
        ctx.save_for_backward(score_tensor)
        return torch.as_tensor(log_normalizers, dtype=singular_values.dtype, device=singular_values.device)

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        (singular_scores,) = ctx.saved_tensors
        return output_gradient[:, None] * singular_scores, None, None


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, min=torch.finfo(value.dtype).tiny)
    return value + torch.log(-torch.expm1(-value))


def _mixture_probabilities(labels: torch.Tensor, K: int, dtype: torch.dtype) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=K).to(dtype)
    probabilities = counts / counts.sum()
    return torch.clamp(probabilities, min=torch.finfo(dtype).eps)


def _uniform_mixture_probabilities(K: int, reference: torch.Tensor) -> torch.Tensor:
    return torch.full((K,), 1.0 / K, dtype=reference.dtype, device=reference.device)


def _uses_uniform_initialization(init_method) -> bool:
    return init_method in {'uniform', 'unif'}


def _uses_isotropic_initialization(init_method) -> bool:
    return init_method == 'isotropic'


def _fixed_norm_gaussian(reference: torch.Tensor, shape: tuple[int, ...], norm: float = 1.0) -> torch.Tensor:
    values = torch.randn(shape, dtype=reference.dtype, device=reference.device)
    flattened = values.reshape(shape[0], -1)
    lengths = torch.linalg.vector_norm(flattened, dim=1)
    return values * (norm / lengths).reshape((-1,) + (1,) * (values.ndim - 1))


def _isotropic_matrix_fisher_factors(reference: torch.Tensor, K: int, p: int, q: int, rank: int, norm: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced factors whose product has a fixed Frobenius norm."""
    left_factors = reference.new_empty((K, p, rank))
    right_factors = reference.new_empty((K, q, rank))
    for k in range(K):
        matrix = torch.randn((p, q), dtype=reference.dtype, device=reference.device)
        left, singular_values, right_h = torch.linalg.svd(matrix, full_matrices=False)
        singular_values = singular_values[:rank]
        singular_values = (norm * singular_values / torch.linalg.vector_norm(singular_values))
        roots = torch.sqrt(singular_values)
        left_factors[k] = left[:, :rank] * roots[None, :]
        right_factors[k] = right_h.mT[:, :rank] * roots[None, :]
    return left_factors, right_factors


def _polar_factor(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right_h


def _cluster_labels(X: torch.Tensor, K: int, axial_matrix: bool = False) -> torch.Tensor:
    """Small torch-only initializer; the experiment normally uses K=1."""
    if K < 1 or K > X.shape[0]:
        raise ValueError('K should satisfy 1 <= K <= number of observations.')
    if K == 1:
        return torch.zeros(X.shape[0], dtype=torch.long, device=X.device)

    centers = X[torch.randperm(X.shape[0], device=X.device)[:K]].clone()
    labels = torch.full((X.shape[0],), -1, dtype=torch.long, device=X.device)
    for _ in range(25):
        if X.ndim == 2:
            scores = X @ centers.mT
        elif axial_matrix:
            cross_grams = torch.einsum('npq,kpr->nkqr', X, centers)
            scores = cross_grams.square().sum(dim=(-2, -1))
        else:
            scores = torch.einsum('npq,kpq->nk', X, centers)
        new_labels = scores.argmax(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(K):
            members = X[labels == k]
            if members.shape[0] == 0:
                centers[k] = X[torch.randint(X.shape[0], (), device=X.device)]
            elif X.ndim == 2:
                centers[k] = nn.functional.normalize(members.mean(dim=0), dim=0)
            elif axial_matrix:
                scatter = torch.einsum('npq,nrq->pr', members, members)
                centers[k] = torch.linalg.eigh(scatter).eigenvectors[:, -X.shape[2]:]
            else:
                centers[k] = _polar_factor(members.mean(dim=0))
    return labels


def _validate_sphere_data(X: torch.Tensor, p: int, name: str) -> None:
    if X.ndim != 2 or X.shape[1] != p or torch.is_complex(X):
        raise ValueError(f'{name} expects real data with shape (n, {p}).')
    expected = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
    if not torch.allclose(torch.linalg.norm(X, dim=1), expected, rtol=1e-5, atol=1e-6):
        raise ValueError(f'{name} expects unit-norm observations.')


def _validate_stiefel_data(X: torch.Tensor, p: int, q: int, name: str) -> None:
    if X.ndim != 3 or X.shape[1:] != (p, q) or torch.is_complex(X):
        raise ValueError(f'{name} expects real data with shape (n, {p}, {q}).')
    gram = X.mT @ X
    identity = torch.eye(q, dtype=X.dtype, device=X.device).expand_as(gram)
    if not torch.allclose(gram, identity, rtol=1e-5, atol=1e-6):
        raise ValueError(f'{name} expects X.T @ X = I for every observation.')


def _component_rows(X: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    rows = X[labels == k]
    return rows if rows.shape[0] >= 2 else X


def _vector_initial_values(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = rows.mean(dim=0)
    resultant = torch.clamp(torch.linalg.norm(mean), max=1.0 - 1e-7)
    if resultant < 1e-7:
        mu = nn.functional.normalize(rows[0], dim=0)
    else:
        mu = mean / resultant
    p = rows.shape[1]
    kappa = resultant * (p - resultant.square()) / torch.clamp(1.0 - resultant.square(), min=1e-7)
    return mu, torch.clamp(kappa, min=1e-3, max=1e4)


def _quadratic_factor(rows: torch.Tensor, rank: int) -> torch.Tensor:
    p = rows.shape[1]
    scatter = rows.mT @ rows / rows.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
    active_values = torch.clamp(eigenvalues[:rank], min=torch.finfo(rows.dtype).eps)
    reference = eigenvalues[rank:].mean() if rank < p else eigenvalues.mean()
    penalties = 0.5 * torch.clamp(reference / active_values - 1.0, min=1e-3, max=1e3)
    distinct = 1.0 + 1e-5 * torch.arange(rank, dtype=rows.dtype, device=rows.device)
    return eigenvectors[:, :rank] * torch.sqrt(penalties * distinct)[None, :]


class _SphereFisherBinghamQuadrature:
    """Chen--Tanaka continuous-Euler quadrature for exp(b'x-x'Px)."""

    def _configure_sphere_quadrature(self, integration_points: int, omega_d: float, omega_u: float) -> None:
        if not isinstance(integration_points, int) or integration_points < 1:
            raise ValueError('integration_points should be a positive integer.')
        if not (0 < omega_d <= 1 <= omega_u and omega_d / omega_u <= 0.5):
            raise ValueError('Require 0 < omega_d <= 1 <= omega_u and omega_d / omega_u <= 1/2.')
        self.integration_points = integration_points
        self.omega_d = float(omega_d)
        self.omega_u = float(omega_u)

    @staticmethod
    def _saddle_contour(eigenvalues: torch.Tensor, active_linear: torch.Tensor, perpendicular_linear_square: torch.Tensor, p: int) -> float:
        values = eigenvalues.detach().double().cpu()
        active_square = active_linear.detach().double().cpu().square()
        perpendicular_square = float(perpendicular_linear_square.detach().double().cpu())
        inactive_dimension = p - values.numel()

        def equation(contour: float) -> float:
            first = inactive_dimension / contour
            first += float(torch.sum(1.0 / (contour + values)))
            second = perpendicular_square / contour**2
            second += float(torch.sum(active_square / (contour + values).square()))
            return first + 0.5 * second - 2.0

        lower = 1e-10
        upper = max(p / 2.0, 2.0)
        while equation(upper) > 0 and upper < 1e12:
            upper *= 2.0
        if equation(lower) <= 0 or equation(upper) >= 0:
            return max(p / 2.0, 2.0)
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            if equation(middle) > 0:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)

    def _sphere_log_normalizer(self, linear: torch.Tensor, M: Optional[torch.Tensor]) -> torch.Tensor:
        results = []
        for k in range(linear.shape[0]):
            component_M = None if M is None else M[k]
            near_zero = component_M is not None and component_M.detach().square().sum() <= torch.finfo(linear.dtype).eps
            if component_M is None or near_zero:
                eigenvalues = linear.new_empty(0)
                active_linear = linear.new_empty(0)
                perpendicular_square = linear[k].square().sum()
            else:
                left, singular_values, _ = torch.linalg.svd(component_M, full_matrices=False)
                eigenvalues = singular_values.square()
                active_linear = left.mT @ linear[k]
                perpendicular_square = torch.clamp(linear[k].square().sum() - active_linear.square().sum(), min=0.0)
            contour = self._saddle_contour(eigenvalues, active_linear, perpendicular_square, self.p)
            h = math.sqrt(2.0 * math.pi * contour * (self.omega_d + self.omega_u) / (self.omega_d**2 * self.integration_points))
            window_p = math.sqrt(self.integration_points * h / self.omega_d)
            window_q = math.sqrt(self.omega_d * self.integration_points * h / 4.0)
            indices = torch.arange(-self.integration_points - 1, self.integration_points + 1, dtype=linear.dtype, device=linear.device)
            nodes = indices * h
            weights = 0.5 * torch.erfc(nodes.abs() / window_p - window_q)
            complex_dtype = torch.complex128 if linear.dtype == torch.float64 else torch.complex64
            shifted = contour + 1j * nodes.to(complex_dtype)
            log_determinant = (self.p - eigenvalues.numel()) * torch.log(shifted)
            if eigenvalues.numel():
                log_determinant = log_determinant + torch.log(shifted[:, None] + eigenvalues[None, :].to(complex_dtype)).sum(dim=1)
            inverse_quadratic = perpendicular_square.to(complex_dtype) / shifted
            if active_linear.numel():
                inverse_quadratic = inverse_quadratic + (
                    active_linear.square()[None, :].to(complex_dtype)
                    / (shifted[:, None] + eigenvalues[None, :].to(complex_dtype))
                ).sum(dim=1)
            log_integrand = -0.5 * log_determinant + 0.25 * inverse_quadratic + 1j * nodes
            scale = log_integrand.real.max()
            integral = torch.sum(weights * torch.exp(log_integrand - scale)).real
            if integral <= 0:
                raise RuntimeError('Sphere normalizer was non-positive; increase integration_points.')
            results.append((self.p / 2.0 - 1.0) * math.log(math.pi) + contour + math.log(h) + scale + torch.log(integral))
        return torch.stack(results)


class VonMisesFisher(_SphereFisherBinghamQuadrature, PCMMtorchBaseModel):
    """Real von Mises--Fisher distribution on S^(p-1)."""

    normalizer_kind = 'continuous_euler_quadrature'

    def __init__(self, p: int, K: int = 1, HMM: bool = False, samples_per_sequence=0, params: Optional[dict] = None,
        integration_points: int = 400, omega_d: float = 0.5, omega_u: float = 2.0) -> None:
        super().__init__()
        if p < 2:
            raise ValueError('VonMisesFisher requires p >= 2.')
        self.p, self.K, self.HMM = p, K, HMM
        self.samples_per_sequence = torch.as_tensor(0 if samples_per_sequence is None else samples_per_sequence)
        self.distribution = 'VonMisesFisher'
        self.flag_normalized_input_data = False
        self._configure_sphere_quadrature(integration_points, omega_d, omega_u)
        if params is not None:
            self.unpack_params(params)

    def _initialize_distribution(self, X: torch.Tensor, init_method=None) -> None:
        _validate_sphere_data(X, self.p, 'VonMisesFisher')
        if _uses_isotropic_initialization(init_method):
            mu = _fixed_norm_gaussian(X, (self.K, self.p))
            kappa = torch.full((self.K,), 1e-3, dtype=X.dtype, device=X.device)
            self.unpack_params({'mu': mu, 'kappa': _inverse_softplus(kappa), 'pi': _uniform_mixture_probabilities(self.K, X),})
            return
        if _uses_uniform_initialization(init_method):
            mu = nn.functional.normalize(torch.rand((self.K, self.p), dtype=X.dtype, device=X.device), dim=1)
            # An exact zero concentration makes the direction unidentified;
            # machine epsilon also underflows the quadrature's useful
            # resolution and leaves Adam on an effectively flat objective.
            # Start close to uniform while retaining an optimizable gradient.
            kappa = torch.full((self.K,), 1e-3, dtype=X.dtype, device=X.device)
            self.unpack_params({'mu': mu, 'kappa': _inverse_softplus(kappa), 'pi': _uniform_mixture_probabilities(self.K, X),})
            return
        if init_method is not None:
            raise ValueError(f'Unsupported initialization method for VonMisesFisher: {init_method}')
        labels = _cluster_labels(X, self.K)
        mu = X.new_empty((self.K, self.p))
        raw_kappa = X.new_empty(self.K)
        for k in range(self.K):
            mu[k], kappa = _vector_initial_values(_component_rows(X, labels, k))
            raw_kappa[k] = _inverse_softplus(kappa)
        self.unpack_params({'mu': mu, 'kappa': raw_kappa, 'pi': _mixture_probabilities(labels, self.K, X.dtype)})

    def log_norm_constant(self) -> torch.Tensor:
        mu = nn.functional.normalize(self.mu, dim=1)
        kappa = nn.functional.softplus(self.kappa)
        return self._sphere_log_normalizer(kappa[:, None] * mu, M=None)

    def log_pdf(self, X: torch.Tensor, recompute_statics: bool = False) -> torch.Tensor:
        _validate_sphere_data(X, self.p, 'VonMisesFisher')
        mu = nn.functional.normalize(self.mu, dim=1)
        kappa = nn.functional.softplus(self.kappa)
        kernel = kappa[:, None] * (X @ mu.mT).mT
        return kernel - self.log_norm_constant()[:, None]


class FisherBingham(_SphereFisherBinghamQuadrature, PCMMtorchBaseModel):
    """Low-rank real Fisher--Bingham distribution on S^(p-1)."""

    normalizer_kind = 'continuous_euler_quadrature'

    def __init__(self, p: int, rank: int, K: int = 1, HMM: bool = False, samples_per_sequence=0, params: Optional[dict] = None,
        integration_points: int = 400, omega_d: float = 0.5, omega_u: float = 2.0) -> None:
        super().__init__()
        if p < 2 or not isinstance(rank, int) or rank < 1 or rank > p:
            raise ValueError('FisherBingham requires p >= 2 and 1 <= rank <= p.')
        self.p, self.r, self.K, self.HMM = p, rank, K, HMM
        self.samples_per_sequence = torch.as_tensor(0 if samples_per_sequence is None else samples_per_sequence)
        self.distribution = 'FisherBingham_lowrank'
        self.flag_normalized_input_data = False
        self._configure_sphere_quadrature(integration_points, omega_d, omega_u)
        if params is not None:
            self.unpack_params(params)

    def _initialize_distribution(self, X: torch.Tensor, init_method=None) -> None:
        _validate_sphere_data(X, self.p, 'FisherBingham')
        if _uses_isotropic_initialization(init_method):
            mu = _fixed_norm_gaussian(X, (self.K, self.p))
            kappa = torch.full((self.K,), 1e-3, dtype=X.dtype, device=X.device)
            self.unpack_params({
                'mu': mu,
                'kappa': _inverse_softplus(kappa),
                'M': _fixed_norm_gaussian(X, (self.K, self.p, self.r)),
                'pi': _uniform_mixture_probabilities(self.K, X),
            })
            return
        if _uses_uniform_initialization(init_method):
            mu = nn.functional.normalize(torch.rand((self.K, self.p), dtype=X.dtype, device=X.device), dim=1)
            # Machine epsilon makes the linear term numerically unidentified
            # in the quadrature. Keep it close to uniform but trainable.
            kappa = torch.full((self.K,), 1e-3, dtype=X.dtype, device=X.device)
            self.unpack_params({
                'mu': mu,
                'kappa': _inverse_softplus(kappa),
                'M': torch.rand((self.K, self.p, self.r), dtype=X.dtype, device=X.device),
                'pi': _uniform_mixture_probabilities(self.K, X),
            })
            return
        if init_method is not None:
            raise ValueError(f'Unsupported initialization method for FisherBingham: {init_method}')
        labels = _cluster_labels(X, self.K)
        mu = X.new_empty((self.K, self.p))
        raw_kappa = X.new_empty(self.K)
        M = X.new_empty((self.K, self.p, self.r))
        for k in range(self.K):
            rows = _component_rows(X, labels, k)
            mu[k], kappa = _vector_initial_values(rows)
            raw_kappa[k] = _inverse_softplus(kappa)
            M[k] = _quadratic_factor(rows, self.r)
        params = {'mu': mu, 'kappa': raw_kappa, 'M': M, 'pi': _mixture_probabilities(labels, self.K, X.dtype)}
        self.unpack_params(params)

    def log_norm_constant(self) -> torch.Tensor:
        mu = nn.functional.normalize(self.mu, dim=1)
        kappa = nn.functional.softplus(self.kappa)
        return self._sphere_log_normalizer(kappa[:, None] * mu, self.M)

    def log_pdf(self, X: torch.Tensor, recompute_statics: bool = False) -> torch.Tensor:
        _validate_sphere_data(X, self.p, 'FisherBingham')
        mu = nn.functional.normalize(self.mu, dim=1)
        kappa = nn.functional.softplus(self.kappa)
        linear = kappa[:, None] * (X @ mu.mT).mT
        projection = torch.einsum('np,kpr->knr', X, self.M)
        kernel = linear - projection.square().sum(dim=-1)
        return kernel - self.log_norm_constant()[:, None]


def _matrix_fisher_factors(rows: torch.Tensor, linear_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = rows.mean(dim=0)
    left, singular_values, right_h = torch.linalg.svd(mean, full_matrices=False)
    selected = torch.clamp(singular_values[:linear_rank], max=1.0 - 1e-6)
    concentration = rows.shape[1] * selected / torch.clamp(1.0 - selected.square(), min=1e-6)
    concentration = torch.clamp(concentration, min=1e-3, max=1e4)
    square_root = torch.sqrt(concentration)
    return left[:, :linear_rank] * square_root[None, :], right_h.mT[:, :linear_rank] * square_root[None, :]


def _matrix_quadratic_factor(rows: torch.Tensor, rank: int) -> torch.Tensor:
    p, q = rows.shape[1:]
    scatter = torch.einsum('npq,nrq->pr', rows, rows) / (rows.shape[0] * q)
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
    active_values = torch.clamp(eigenvalues[:rank], min=torch.finfo(rows.dtype).eps)
    reference = eigenvalues[rank:].mean() if rank < p else eigenvalues.mean()
    penalties = 0.5 * torch.clamp(reference / active_values - 1.0, min=1e-3, max=1e3)
    distinct = 1.0 + 1e-5 * torch.arange(rank, dtype=rows.dtype, device=rows.device)
    return eigenvectors[:, :rank] * torch.sqrt(penalties * distinct)[None, :]


class _MatrixFisherBinghamSaddlepoint:
    """Uniform-calibrated saddlepoint normalizer on V_2(R^p).

    Second-order corrections use a scalable five-point theta-grid backend in
    float64 by default.  ``saddlepoint_derivative_backend='autodiff'`` retains
    the nested-autodiff reference implementation.
    """

    normalizer_kind = 'second_order_saddlepoint_relative_to_normalized_haar'

    def unpack_params(self, params: dict) -> None:
        PCMMtorchBaseModel.unpack_params(self, params)
        self._saddle_cache = None

    def _configure_matrix_saddlepoint(self, saddlepoint_iterations: int, saddlepoint_tolerance: float, saddlepoint_order: int = 2,
        saddlepoint_derivative_backend: str = 'finite_difference', saddlepoint_finite_difference_step: float = 5e-3) -> None:
        if self.q != 2:
            raise NotImplementedError('The structured saddlepoint normalizer currently supports q=2 only.')
        if saddlepoint_iterations < 1 or saddlepoint_tolerance <= 0:
            raise ValueError('Saddlepoint iterations and tolerance should be positive.')
        if saddlepoint_order not in {1, 2}:
            raise ValueError('saddlepoint_order must be either 1 or 2.')
        if saddlepoint_derivative_backend not in {'autodiff', 'finite_difference'}:
            raise ValueError("saddlepoint_derivative_backend should be 'autodiff' or " "'finite_difference'.")
        if not math.isfinite(saddlepoint_finite_difference_step) or saddlepoint_finite_difference_step <= 0.0:
            raise ValueError('saddlepoint_finite_difference_step should be finite and positive.')
        self.saddlepoint_iterations = int(saddlepoint_iterations)
        self.saddlepoint_tolerance = float(saddlepoint_tolerance)
        self.saddlepoint_order = int(saddlepoint_order)
        self.saddlepoint_derivative_backend = saddlepoint_derivative_backend
        self.saddlepoint_finite_difference_step = float(saddlepoint_finite_difference_step)
        order_name = 'second' if self.saddlepoint_order == 2 else 'first'
        self.normalizer_kind = f'{order_name}_order_saddlepoint_relative_to_normalized_haar'
        self._saddle_cache: Optional[torch.Tensor] = None
        self._saddle_reference_correction: Optional[torch.Tensor] = None
        self._current_saddle_solve_diagnostic: dict[str, float | int] = {}
        self.last_saddlepoint_newton_iterations: tuple[int, ...] = ()
        self.last_saddlepoint_backtracks: tuple[int, ...] = ()
        self.last_saddlepoint_residuals: tuple[float, ...] = ()
        self.last_saddlepoint_finite_difference_steps: tuple[float, ...] = ()

    def _matrix_gaussian_terms(self, theta: torch.Tensor, M: Optional[torch.Tensor], F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contour = self.p / 2.0
        a = 2.0 * (contour - theta[0])
        d = 2.0 * (contour - theta[1])
        b = -theta[2]
        determinant_zero = a * d - b.square()

        if M is None:
            PF = torch.zeros_like(F)
            log_determinant = self.p * torch.log(determinant_zero)
            Y = torch.stack((d * F[:, 0] - b * F[:, 1], -b * F[:, 0] + a * F[:, 1]), dim=1)
            solved = Y / determinant_zero
        else:
            gram = M.mT @ M
            identity = torch.eye(gram.shape[0], dtype=M.dtype, device=M.device)
            H = 2.0 * (a + d) * identity + 4.0 * gram
            middle = identity + H @ gram / determinant_zero
            sign, logabsdet = torch.linalg.slogdet(middle)
            if sign <= 0:
                raise RuntimeError('Saddlepoint determinant lost positive definiteness.')
            log_determinant = self.p * torch.log(determinant_zero) + logabsdet
            PF = M @ (M.mT @ F)
            Y = torch.stack((d * F[:, 0] + 2.0 * PF[:, 0] - b * F[:, 1], -b * F[:, 0] + a * F[:, 1] + 2.0 * PF[:, 1]), dim=1)
            right = H @ (M.mT @ Y) / determinant_zero
            solved = Y / determinant_zero - M @ torch.linalg.solve(middle, right) / determinant_zero
        quadratic = torch.sum(F * solved)
        return log_determinant, quadratic

    def _matrix_cumulant(self, theta: torch.Tensor, M: Optional[torch.Tensor], F: torch.Tensor) -> torch.Tensor:
        log_determinant, quadratic = self._matrix_gaussian_terms(theta, M, F)
        return 0.5 * (quadratic - log_determinant)

    def _autodiff_cumulant_derivatives(self, theta: torch.Tensor, M: Optional[torch.Tensor],
                                        F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return second through fourth theta derivatives of the cumulant."""
        def cumulant_gradient(value: torch.Tensor) -> torch.Tensor:
            cumulant = self._matrix_cumulant(value, M, F)
            return torch.autograd.grad(cumulant, value, create_graph=True)[0]

        def cumulant_hessian(value: torch.Tensor) -> torch.Tensor:
            return torch.autograd.functional.jacobian(cumulant_gradient, value, create_graph=True, vectorize=True)

        def cumulant_third(value: torch.Tensor) -> torch.Tensor:
            return torch.autograd.functional.jacobian(cumulant_hessian, value, create_graph=True, vectorize=True)

        hessian = cumulant_hessian(theta)
        third = cumulant_third(theta)
        fourth = torch.autograd.functional.jacobian(cumulant_third, theta, create_graph=True, vectorize=True)
        return hessian, third, fourth

    @staticmethod
    def _finite_difference_weights(derivative_order: int, reference: torch.Tensor) -> torch.Tensor:
        """Five-point centered weights before division by the step size."""
        coefficients = {
            0: (0.0, 0.0, 1.0, 0.0, 0.0),
            1: (1.0, -8.0, 0.0, 8.0, -1.0),
            2: (-1.0, 16.0, -30.0, 16.0, -1.0),
            3: (-1.0, 2.0, 0.0, -2.0, 1.0),
            4: (1.0, -4.0, 6.0, -4.0, 1.0),
        }
        divisors = {0: 1.0, 1: 12.0, 2: 12.0, 3: 2.0, 4: 1.0}
        return reference.new_tensor(coefficients[derivative_order]) / divisors[derivative_order]

    def _finite_difference_cumulant_derivatives(self, theta: torch.Tensor, M: Optional[torch.Tensor],
                                                 F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autodiff Hessian plus grid finite differences for orders 3 and 4.

        The 125 cumulant values share one centered 5 x 5 x 5 theta grid.
        Tensor-product centered stencils recover every symmetric third- and
        fourth-derivative entry while retaining parameter gradients through
        the cumulant evaluations.  The step is held fixed during backward,
        as is standard for a local finite-difference approximation.
        """
        central = self._matrix_cumulant(theta, M, F)
        gradient = torch.autograd.grad(central, theta, create_graph=True)[0]
        hessian = torch.autograd.functional.jacobian(
            lambda value: torch.autograd.grad(self._matrix_cumulant(value, M, F), value, create_graph=True)[0],
            theta, create_graph=True, vectorize=True)
        # Fourth-difference roundoff grows with the O(p) cumulant baseline;
        # sqrt(p/4) balances that effect against the O(h^2) stencil error.
        # Large linear natural parameters add further cancellation, handled by
        # a mild dimension-normalized square-root factor.  Both scales are
        # detached: the local numerical step is held fixed during backward.
        dimension_scale = math.sqrt(self.p / 4.0)
        linear_scale = torch.sqrt(1.0 + torch.linalg.vector_norm(F.detach()) / float(self.p))
        linear_scale = torch.clamp(linear_scale, max=4.0)
        step = (theta.new_tensor(self.saddlepoint_finite_difference_step) * dimension_scale * linear_scale)
        # The Gaussian-domain feasible set is convex, so feasibility at all
        # eight cube corners implies feasibility of the complete 5^3 grid.
        # Halving only changes a detached numerical step, not the model graph.
        for _ in range(12):
            corners_are_feasible = all(
                self._saddle_feasible(theta.detach() + 2.0 * step * theta.new_tensor((first, second, third)))
                for first in (-1.0, 1.0)
                for second in (-1.0, 1.0)
                for third in (-1.0, 1.0)
            )
            if corners_are_feasible:
                break
            step = step / 2.0
        else:
            return self._autodiff_cumulant_derivatives(theta, M, F)
        self._current_saddle_solve_diagnostic['finite_difference_step'] = float(step.detach().cpu())
        offsets = torch.arange(-2, 3, dtype=theta.dtype, device=theta.device)
        grid_rows = []
        for first in offsets:
            grid_columns = []
            for second in offsets:
                grid_depth = []
                for third in offsets:
                    displacement = step * torch.stack((first, second, third))
                    grid_depth.append(self._matrix_cumulant(theta + displacement, M, F))
                grid_columns.append(torch.stack(grid_depth))
            grid_rows.append(torch.stack(grid_columns))
        grid = torch.stack(grid_rows)
        # Every requested stencil annihilates polynomials through degree two.
        # Removing the exact local Taylor polynomial before applying weights
        # of order h^-4 substantially reduces avoidable cancellation.
        first_grid, second_grid, third_grid = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
        displacements = step * torch.stack((first_grid, second_grid, third_grid), dim=-1)
        linear_taylor = torch.einsum('...i,i->...', displacements, gradient)
        quadratic_taylor = 0.5 * torch.einsum('...i,ij,...j->...', displacements, hessian, displacements)
        grid = grid - central - linear_taylor - quadratic_taylor

        weights = {order: self._finite_difference_weights(order, theta) / step**order for order in range(5)}
        derivative_cache: dict[tuple[int, int, int], torch.Tensor] = {}

        def derivative(indices: tuple[int, ...]) -> torch.Tensor:
            orders = tuple(indices.count(axis) for axis in range(3))
            if orders not in derivative_cache:
                derivative_cache[orders] = torch.einsum('i,j,k,ijk->', weights[orders[0]], weights[orders[1]], weights[orders[2]], grid)
            return derivative_cache[orders]

        third = torch.stack([
            torch.stack([torch.stack([derivative((first, second, third_index)) for third_index in range(3)]) for second in range(3)])
            for first in range(3)
        ])
        fourth = torch.stack([
            torch.stack([
                torch.stack([torch.stack([derivative((first, second, third_index, fourth_index)) for fourth_index in range(3)]) for third_index in range(3)])
                for second in range(3)
            ])
            for first in range(3)
        ])
        return hessian, third, fourth

    def _cumulant_derivatives(self, theta: torch.Tensor, M: Optional[torch.Tensor], F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fourth differences are intentionally a float64 backend: h^-4
        # amplification is not reliable at float32 precision.  Falling back
        # to the reference preserves the historical behavior for float32
        # callers, while the computational experiment runs in float64.
        if self.saddlepoint_derivative_backend == 'finite_difference' and theta.dtype == torch.float64:
            return self._finite_difference_cumulant_derivatives(theta, M, F)
        return self._autodiff_cumulant_derivatives(theta, M, F)

    @staticmethod
    def _second_order_correction(inverse_hessian: torch.Tensor, third: torch.Tensor, fourth: torch.Tensor) -> torch.Tensor:
        """Kume--Preston--Wood second-order saddlepoint correction.

        This is ``T = rho4/8 - rho13^2/8 - rho23^2/12`` and the normalizer
        uses their published ``1 + T`` variant.  The two cubic contractions
        are the internally paired and fully cross-linked Wick topologies.
        """
        fourth_term = torch.einsum('ijkl,ij,kl->', fourth, inverse_hessian, inverse_hessian)
        cubic_cross = torch.einsum('ijk,lmn,il,jm,kn->', third, third, inverse_hessian, inverse_hessian, inverse_hessian)
        cubic_internal = torch.einsum('ijk,lmn,ij,kl,mn->', third, third, inverse_hessian, inverse_hessian, inverse_hessian)
        return (1.0 + fourth_term / 8.0 - cubic_cross / 12.0 - cubic_internal / 8.0)

    def _uniform_second_order_correction(self, reference: torch.Tensor) -> torch.Tensor:
        cached = self._saddle_reference_correction
        if cached is not None:
            return cached.to(dtype=reference.dtype, device=reference.device)
        # At M=F=theta=0, K(theta) is
        # -p/2 log(((p-2 theta_1)(p-2 theta_2)-theta_3^2)).
        # Substitution of its derivatives in the Kume--Preston--Wood
        # contraction gives the exact normalized-Haar reference correction.
        correction = reference.new_tensor(1.0 - 13.0 / (12.0 * self.p))
        if not torch.isfinite(correction) or correction <= 0:
            raise RuntimeError('The uniform second-order saddlepoint correction was non-positive.')
        self._saddle_reference_correction = correction
        return correction

    def _saddle_objective(self, theta: torch.Tensor, M: Optional[torch.Tensor], F: torch.Tensor) -> torch.Tensor:
        return self._matrix_cumulant(theta, M, F) - theta[0] - theta[1]

    def _saddle_feasible(self, theta: torch.Tensor) -> bool:
        contour = self.p / 2.0
        a = 2.0 * (contour - theta[0])
        d = 2.0 * (contour - theta[1])
        determinant = a * d - theta[2].square()
        return bool(torch.isfinite(theta).all() and a > 1e-10 and d > 1e-10 and determinant > 1e-10)

    def _solve_saddle(self, M: Optional[torch.Tensor], F: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        M_fixed = None if M is None else M.detach()
        F_fixed = F.detach()
        theta = initial.detach().clone()
        if not self._saddle_feasible(theta):
            theta.zero_()

        newton_iterations = 0
        total_backtracks = 0
        for _ in range(self.saddlepoint_iterations):
            theta = theta.detach().requires_grad_(True)
            objective = self._saddle_objective(theta, M_fixed, F_fixed)
            gradient = torch.autograd.grad(objective, theta, create_graph=True)[0]
            residual = torch.linalg.vector_norm(gradient).detach()
            if residual <= self.saddlepoint_tolerance:
                self._current_saddle_solve_diagnostic = {
                    'newton_iterations': newton_iterations,
                    'backtracks': total_backtracks,
                    'pre_refinement_residual': float(residual.cpu()),
                }
                return theta.detach()
            hessian = torch.autograd.functional.hessian(lambda value: self._saddle_objective(value, M_fixed, F_fixed), theta)
            ridge = 1e-8 * torch.eye(3, dtype=theta.dtype, device=theta.device)
            step = torch.linalg.solve(hessian + ridge, gradient.detach())
            accepted = False
            scale = 1.0
            current = objective.detach()
            rejected_scales = 0
            for _ in range(24):
                candidate = theta.detach() - scale * step
                if self._saddle_feasible(candidate):
                    candidate_value = self._saddle_objective(candidate, M_fixed, F_fixed).detach()
                    if torch.isfinite(candidate_value) and candidate_value <= current:
                        theta = candidate
                        accepted = True
                        break
                scale *= 0.5
                rejected_scales += 1
            newton_iterations += 1
            total_backtracks += rejected_scales
            if not accepted:
                self._current_saddle_solve_diagnostic = {
                    'newton_iterations': newton_iterations,
                    'backtracks': total_backtracks,
                    'pre_refinement_residual': float(residual.cpu()),
                }
                return theta.detach()
        self._current_saddle_solve_diagnostic = {
            'newton_iterations': newton_iterations,
            'backtracks': total_backtracks,
            'pre_refinement_residual': float(residual.cpu()),
        }
        return theta.detach()

    def _attach_implicit_saddle_gradient(self, saddle: torch.Tensor, M: Optional[torch.Tensor], F: torch.Tensor) -> torch.Tensor:
        """Attach the saddlepoint's implicit parameter derivative.

        ``_solve_saddle`` deliberately performs its numerical Newton solve on
        detached model parameters.  The value returned by that solve is
        therefore correct, but treating it as an independent leaf would omit
        ``d theta_star / d (M, F)`` from the determinant and higher-order
        saddlepoint corrections.  One differentiable Newton step at a
        converged or nearly converged root both removes the remaining numerical
        residual and supplies the implicit function derivative
        ``-H^{-1} d gradient / d (M, F)``.
        """
        base = saddle.detach().requires_grad_(True)

        def objective(theta: torch.Tensor) -> torch.Tensor:
            return self._saddle_objective(theta, M, F)

        gradient = torch.autograd.grad(objective(base), base, create_graph=True)[0]
        hessian = torch.autograd.functional.hessian(objective, base, create_graph=True)

        corrected = base - torch.linalg.solve(hessian, gradient)

        # Validate the refined value rather than the input from the deliberately
        # damped numerical solve.  A small pre-refinement residual is expected
        # when its ridge or iteration limit is active, while a Newton step in
        # the local basin should reduce that residual quadratically.
        check_theta = corrected.detach().requires_grad_(True)
        check_M = None if M is None else M.detach()
        check_F = F.detach()
        check_gradient = torch.autograd.grad(self._saddle_objective(check_theta, check_M, check_F), check_theta,)[0]
        residual = torch.linalg.vector_norm(check_gradient.detach())
        self._current_saddle_solve_diagnostic['residual'] = float(residual.cpu())
        numerical_floor = 100.0 * torch.finfo(base.dtype).eps
        residual_tolerance = max(10.0 * self.saddlepoint_tolerance, numerical_floor)
        if not self._saddle_feasible(check_theta.detach()) or not torch.isfinite(residual) or residual > residual_tolerance:
            raise RuntimeError('Saddlepoint solver did not converge: gradient residual ' f'{residual.item():.3e} exceeds {residual_tolerance:.3e}.')

        return corrected

    def _matrix_log_normalizer(self, M: Optional[torch.Tensor], F: torch.Tensor) -> torch.Tensor:
        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            result = self._matrix_log_normalizer_with_grad(M, F)
        return result if outer_grad_enabled else result.detach()

    def _matrix_log_normalizer_with_grad(self, M: Optional[torch.Tensor], F: torch.Tensor) -> torch.Tensor:
        if self._saddle_cache is None or self._saddle_cache.shape != (self.K, 3):
            cache = F.new_zeros((self.K, 3))
        else:
            cache = self._saddle_cache.to(dtype=F.dtype, device=F.device)

        solved_saddles = []
        solve_diagnostics = []
        results = []
        for k in range(self.K):
            component_M = None if M is None else M[k]
            saddle = self._solve_saddle(component_M, F[k], cache[k])
            solved_saddles.append(saddle)
            theta = self._attach_implicit_saddle_gradient(saddle, component_M, F[k])
            cumulant = self._matrix_cumulant(theta, component_M, F[k])
            if self.saddlepoint_order == 2:
                hessian, third, fourth = self._cumulant_derivatives(theta, component_M, F[k])
            else:
                hessian = torch.autograd.functional.hessian(lambda value: self._matrix_cumulant(value, component_M, F[k]), theta, create_graph=True)
            sign, log_hessian_determinant = torch.linalg.slogdet(hessian)
            if sign <= 0:
                raise RuntimeError('Saddlepoint Hessian was not positive definite.')
            core = cumulant - theta[0] - theta[1] - 0.5 * log_hessian_determinant
            reference_log_hessian = 2.0 * math.log(2.0 / self.p) + math.log(1.0 / self.p)
            reference_core = -0.5 * reference_log_hessian - self.p * math.log(self.p)
            if self.saddlepoint_order == 2:
                correction = self._second_order_correction(torch.linalg.inv(hessian), third, fourth)
                if not torch.isfinite(correction).detach() or correction.detach() <= 0:
                    raise RuntimeError('The second-order saddlepoint correction was non-positive; ' 'the approximation is not reliable for these parameters.')
                core = core + torch.log(correction)
                reference_core = reference_core + torch.log(self._uniform_second_order_correction(F[k]))
            solve_diagnostics.append(dict(self._current_saddle_solve_diagnostic))
            results.append(core - reference_core)
        self._saddle_cache = torch.stack(solved_saddles).detach()
        self.last_saddlepoint_newton_iterations = tuple(int(diagnostic['newton_iterations']) for diagnostic in solve_diagnostics)
        self.last_saddlepoint_backtracks = tuple(int(diagnostic['backtracks']) for diagnostic in solve_diagnostics)
        self.last_saddlepoint_residuals = tuple(float(diagnostic['residual']) for diagnostic in solve_diagnostics)
        self.last_saddlepoint_finite_difference_steps = (
            tuple(float(diagnostic['finite_difference_step']) for diagnostic in solve_diagnostics)
            if all('finite_difference_step' in diagnostic for diagnostic in solve_diagnostics)
            else ()
        )
        return torch.stack(results)


class MatrixFisher(_MatrixFisherBinghamSaddlepoint, PCMMtorchBaseModel):
    """Rank-structured matrix Fisher (matrix Langevin) model on V_2(R^p).

    The default normalizer is exact for q=2. Set normalizer_method to
    'saddlepoint' to reproduce the first- or second-order approximation.
    """

    def __init__(self, p: int, q: int, linear_rank: int, K: int = 1, HMM: bool = False, samples_per_sequence=0,
        params: Optional[dict] = None, saddlepoint_iterations: int = 20, saddlepoint_tolerance: float = 1e-8, saddlepoint_order: int = 2,
        normalizer_method: str = 'exact', exact_quadrature_points: int = 128, direct_linear_parameterization: bool = False,
        saddlepoint_derivative_backend: str = 'finite_difference', saddlepoint_finite_difference_step: float = 5e-3) -> None:
        super().__init__()
        if p <= q or linear_rank < 1 or linear_rank > q:
            raise ValueError('MatrixFisher requires p > q and 1 <= linear_rank <= q.')
        if normalizer_method not in {'exact', 'saddlepoint'}:
            raise ValueError("normalizer_method should be 'exact' or 'saddlepoint'.")
        if not isinstance(exact_quadrature_points, int) or exact_quadrature_points < 8:
            raise ValueError('exact_quadrature_points should be an integer >= 8.')
        if direct_linear_parameterization and linear_rank != q:
            raise ValueError('Direct Matrix Fisher parameterization requires linear_rank == q.')
        self.p, self.q, self.s, self.K, self.HMM = p, q, linear_rank, K, HMM
        self.normalizer_method = normalizer_method
        self.exact_quadrature_points = exact_quadrature_points
        self.direct_linear_parameterization = bool(direct_linear_parameterization)
        self.samples_per_sequence = torch.as_tensor(0 if samples_per_sequence is None else samples_per_sequence)
        self.distribution = 'MatrixFisher'
        self.flag_normalized_input_data = False
        self._configure_matrix_saddlepoint(saddlepoint_iterations, saddlepoint_tolerance, saddlepoint_order,
                                           saddlepoint_derivative_backend, saddlepoint_finite_difference_step)
        if self.normalizer_method == 'exact':
            self.normalizer_kind = ('exact_gauss_jacobi_bessel_relative_to_normalized_haar')
        if params is not None:
            self.unpack_params(params)

    def _linear_parameter_values(self, F_left: torch.Tensor, F_right: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.direct_linear_parameterization:
            return {'F': F_left @ F_right.mT}
        return {'F_left': F_left, 'F_right': F_right}

    def _initialize_distribution(self, X: torch.Tensor, init_method=None) -> None:
        _validate_stiefel_data(X, self.p, self.q, 'MatrixFisher')
        if _uses_isotropic_initialization(init_method):
            F_left, F_right = _isotropic_matrix_fisher_factors(X, self.K, self.p, self.q, self.s)
            self.unpack_params({**self._linear_parameter_values(F_left, F_right), 'pi': _uniform_mixture_probabilities(self.K, X),})
            return
        if _uses_uniform_initialization(init_method):
            F_left = torch.rand((self.K, self.p, self.s), dtype=X.dtype, device=X.device)
            F_right = torch.rand((self.K, self.q, self.s), dtype=X.dtype, device=X.device)
            self.unpack_params({**self._linear_parameter_values(F_left, F_right), 'pi': _uniform_mixture_probabilities(self.K, X),})
            return
        if init_method is not None:
            raise ValueError(f'Unsupported initialization method for MatrixFisher: {init_method}')
        labels = _cluster_labels(X, self.K)
        F_left = X.new_empty((self.K, self.p, self.s))
        F_right = X.new_empty((self.K, self.q, self.s))
        for k in range(self.K):
            F_left[k], F_right[k] = _matrix_fisher_factors(_component_rows(X, labels, k), self.s)
        params = {**self._linear_parameter_values(F_left, F_right), 'pi': _mixture_probabilities(labels, self.K, X.dtype)}
        self.unpack_params(params)

    def concentration_matrix(self) -> torch.Tensor:
        if self.direct_linear_parameterization:
            return self.F
        return self.F_left @ self.F_right.mT

    def log_norm_constant(self) -> torch.Tensor:
        F = self.concentration_matrix()
        if self.normalizer_method == 'exact':
            singular_values = torch.linalg.svdvals(F)
            return _MatrixFisherQ2ExactLogNormalizer.apply(singular_values, self.p, self.exact_quadrature_points)
        return self._matrix_log_normalizer(M=None, F=F)

    def log_pdf(self, X: torch.Tensor, recompute_statics: bool = False) -> torch.Tensor:
        _validate_stiefel_data(X, self.p, self.q, 'MatrixFisher')
        F = self.concentration_matrix()
        kernel = torch.einsum('kpq,npq->kn', F, X)
        return kernel - self.log_norm_constant()[:, None]


class MatrixBingham(_MatrixFisherBinghamSaddlepoint, PCMMtorchBaseModel):
    """Low-rank matrix Bingham model on V_2(R^p)."""

    def __init__(self, p: int, q: int, rank: int, K: int = 1, HMM: bool = False, samples_per_sequence=0, params: Optional[dict] = None,
        saddlepoint_iterations: int = 20, saddlepoint_tolerance: float = 1e-8, saddlepoint_order: int = 2,
        saddlepoint_derivative_backend: str = 'finite_difference', saddlepoint_finite_difference_step: float = 5e-3) -> None:
        super().__init__()
        if p <= q or rank < 1 or rank > p:
            raise ValueError('MatrixBingham requires p > q and 1 <= rank <= p.')
        self.p, self.q, self.r, self.K, self.HMM = p, q, rank, K, HMM
        self.samples_per_sequence = torch.as_tensor(0 if samples_per_sequence is None else samples_per_sequence)
        self.distribution = 'MatrixBingham_lowrank'
        self.flag_normalized_input_data = False
        self._configure_matrix_saddlepoint(saddlepoint_iterations, saddlepoint_tolerance, saddlepoint_order,
                                           saddlepoint_derivative_backend, saddlepoint_finite_difference_step)
        if params is not None:
            self.unpack_params(params)

    def _initialize_distribution(self, X: torch.Tensor, init_method=None) -> None:
        _validate_stiefel_data(X, self.p, self.q, 'MatrixBingham')
        if _uses_isotropic_initialization(init_method):
            self.unpack_params({'M': _fixed_norm_gaussian(X, (self.K, self.p, self.r)), 'pi': _uniform_mixture_probabilities(self.K, X),})
            return
        if _uses_uniform_initialization(init_method):
            self.unpack_params({'M': torch.rand((self.K, self.p, self.r), dtype=X.dtype, device=X.device), 'pi': _uniform_mixture_probabilities(self.K, X),})
            return
        if init_method is not None:
            raise ValueError(f'Unsupported initialization method for MatrixBingham: {init_method}')
        labels = _cluster_labels(X, self.K, axial_matrix=True)
        M = X.new_empty((self.K, self.p, self.r))
        for k in range(self.K):
            M[k] = _matrix_quadratic_factor(_component_rows(X, labels, k), self.r)
        self.unpack_params({'M': M, 'pi': _mixture_probabilities(labels, self.K, X.dtype)})

    def precision_matrix(self) -> torch.Tensor:
        return self.M @ self.M.mT

    def log_norm_constant(self) -> torch.Tensor:
        F = self.M.new_zeros((self.K, self.p, self.q))
        return self._matrix_log_normalizer(M=self.M, F=F)

    def log_pdf(self, X: torch.Tensor, recompute_statics: bool = False) -> torch.Tensor:
        _validate_stiefel_data(X, self.p, self.q, 'MatrixBingham')
        projection = torch.einsum('npq,kpr->knrq', X, self.M)
        kernel = -projection.square().sum(dim=(-2, -1))
        return kernel - self.log_norm_constant()[:, None]


class MatrixFisherBingham(_MatrixFisherBinghamSaddlepoint, PCMMtorchBaseModel):
    """Structured matrix Fisher--Bingham model on V_2(R^p)."""

    def __init__(self, p: int, q: int, rank: int, linear_rank: int, K: int = 1, HMM: bool = False, samples_per_sequence=0,
        params: Optional[dict] = None, saddlepoint_iterations: int = 20, saddlepoint_tolerance: float = 1e-8, saddlepoint_order: int = 2,
        direct_linear_parameterization: bool = False, saddlepoint_derivative_backend: str = 'finite_difference',
        saddlepoint_finite_difference_step: float = 5e-3) -> None:
        super().__init__()
        if p <= q or rank < 1 or rank > p or linear_rank < 1 or linear_rank > q:
            raise ValueError('MatrixFisherBingham requires p > q, 1 <= rank <= p, and 1 <= linear_rank <= q.')
        if direct_linear_parameterization and linear_rank != q:
            raise ValueError('Direct Matrix Fisher--Bingham linear parameterization requires ' 'linear_rank == q.')
        self.p, self.q, self.r, self.s = p, q, rank, linear_rank
        self.K, self.HMM = K, HMM
        self.direct_linear_parameterization = bool(direct_linear_parameterization)
        self.samples_per_sequence = torch.as_tensor(0 if samples_per_sequence is None else samples_per_sequence)
        self.distribution = 'MatrixFisherBingham_lowrank'
        self.flag_normalized_input_data = False
        self._configure_matrix_saddlepoint(saddlepoint_iterations, saddlepoint_tolerance, saddlepoint_order,
                                           saddlepoint_derivative_backend, saddlepoint_finite_difference_step)
        if params is not None:
            self.unpack_params(params)

    def _linear_parameter_values(self, F_left: torch.Tensor, F_right: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.direct_linear_parameterization:
            return {'F': F_left @ F_right.mT}
        return {'F_left': F_left, 'F_right': F_right}

    def _initialize_distribution(self, X: torch.Tensor, init_method=None) -> None:
        _validate_stiefel_data(X, self.p, self.q, 'MatrixFisherBingham')
        if _uses_isotropic_initialization(init_method):
            F_left, F_right = _isotropic_matrix_fisher_factors(X, self.K, self.p, self.q, self.s)
            self.unpack_params({
                **self._linear_parameter_values(F_left, F_right),
                'M': _fixed_norm_gaussian(X, (self.K, self.p, self.r)),
                'pi': _uniform_mixture_probabilities(self.K, X),
            })
            return
        if _uses_uniform_initialization(init_method):
            F_left = torch.rand((self.K, self.p, self.s), dtype=X.dtype, device=X.device)
            F_right = torch.rand((self.K, self.q, self.s), dtype=X.dtype, device=X.device)
            self.unpack_params({
                **self._linear_parameter_values(F_left, F_right),
                'M': torch.rand((self.K, self.p, self.r), dtype=X.dtype, device=X.device),
                'pi': _uniform_mixture_probabilities(self.K, X),
            })
            return
        if init_method is not None:
            raise ValueError(f'Unsupported initialization method for MatrixFisherBingham: {init_method}')
        labels = _cluster_labels(X, self.K)
        F_left = X.new_empty((self.K, self.p, self.s))
        F_right = X.new_empty((self.K, self.q, self.s))
        M = X.new_empty((self.K, self.p, self.r))
        for k in range(self.K):
            rows = _component_rows(X, labels, k)
            F_left[k], F_right[k] = _matrix_fisher_factors(rows, self.s)
            M[k] = _matrix_quadratic_factor(rows, self.r)
        params = {**self._linear_parameter_values(F_left, F_right), 'M': M, 'pi': _mixture_probabilities(labels, self.K, X.dtype)}
        self.unpack_params(params)

    def concentration_matrix(self) -> torch.Tensor:
        if self.direct_linear_parameterization:
            return self.F
        return self.F_left @ self.F_right.mT

    def precision_matrix(self) -> torch.Tensor:
        return self.M @ self.M.mT

    def log_norm_constant(self) -> torch.Tensor:
        return self._matrix_log_normalizer(M=self.M, F=self.concentration_matrix())

    def log_pdf(self, X: torch.Tensor, recompute_statics: bool = False) -> torch.Tensor:
        _validate_stiefel_data(X, self.p, self.q, 'MatrixFisherBingham')
        F = self.concentration_matrix()
        linear = torch.einsum('kpq,npq->kn', F, X)
        projection = torch.einsum('npq,kpr->knrq', X, self.M)
        kernel = linear - projection.square().sum(dim=(-2, -1))
        return kernel - self.log_norm_constant()[:, None]


__all__ = ['VonMisesFisher', 'FisherBingham', 'MatrixFisher', 'MatrixBingham', 'MatrixFisherBingham']
