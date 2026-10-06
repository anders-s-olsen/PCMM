#!/usr/bin/env python3
"""Fast, independent mathematical checks for the benchmark distributions.

The checks in this script deliberately use identities or numerical references
that are separate from the PCMM implementation being tested.  Matrix Fisher,
matrix Bingham, and matrix Fisher--Bingham saddlepoint accuracy is intentionally
out of scope: those approximations require a dedicated Haar-integration test.

Run from the repository root with::

    conda run --no-capture-output -n hcp \
        python overview_paper/validate_distribution_math.py
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from scipy.special import erf, hyp1f1, i0e, i1e, ive

from PCMM.PCMMtorch import ACG, MACG, Bingham, Watson, WrappedNormal
from PCMM.PCMMtorchAdditional import FisherBingham, VonMisesFisher


DTYPE = torch.float64


def _log_surface_area(p: int) -> float:
    return math.log(2.0) + 0.5 * p * math.log(math.pi) - math.lgamma(0.5 * p)


def _log_uniform_sphere_density(p: int) -> float:
    return -_log_surface_area(p)


def _assert_close(label: str, actual, expected, atol: float) -> float:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    error = float(np.max(np.abs(actual_array - expected_array)))
    if not np.isfinite(error) or error > atol:
        raise AssertionError(
            f"{label}: maximum absolute error {error:.3e} exceeds {atol:.3e}; "
            f"actual={actual_array}, expected={expected_array}"
        )
    return error


def validate_vmf() -> float:
    """Compare the implemented Bessel normalizer with independent formulas."""
    largest_error = 0.0
    for p, kappa in ((2, 0.2), (2, 2.0), (4, 6.0), (16, 33.0)):
        nu = 0.5 * p - 1.0
        exact_log_z = (
            0.5 * p * math.log(2.0 * math.pi)
            + math.log(float(ive(nu, kappa)))
            + kappa
            - nu * math.log(kappa)
        )
        exact_gradient = float(ive(nu + 1.0, kappa) / ive(nu, kappa))

        raw_kappa = math.log(math.expm1(kappa))
        mu = torch.zeros((1, p), dtype=DTYPE)
        mu[0, 0] = 1.0
        model = VonMisesFisher(
            p=p,
            K=1,
            params={
                'mu': mu,
                'kappa': torch.tensor([raw_kappa], dtype=DTYPE),
                'pi': torch.ones(1, dtype=DTYPE),
            },
        )
        log_z = model.log_norm_constant().sum()
        raw_gradient = torch.autograd.grad(log_z, model.kappa)[0][0]
        gradient = raw_gradient / torch.sigmoid(model.kappa.detach()[0])
        largest_error = max(
            largest_error,
            _assert_close(f"vMF log Z (p={p}, kappa={kappa})", log_z.item(), exact_log_z, 1e-6),
            _assert_close(
                f"vMF d log Z / d kappa (p={p}, kappa={kappa})",
                gradient.item(),
                exact_gradient,
                1e-6,
            ),
        )

    # At zero concentration, log Z is the sphere's surface area.
    mu = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    model = VonMisesFisher(
        p=4,
        K=1,
        params={
            'mu': mu,
            'kappa': torch.tensor([-100.0], dtype=DTYPE),
            'pi': torch.ones(1, dtype=DTYPE),
        },
    )
    largest_error = max(
        largest_error,
        _assert_close(
            "vMF uniform constant", model.log_norm_constant().item(),
            _log_surface_area(4), 1e-8,
        ),
    )
    return largest_error


def validate_watson() -> float:
    """Check positive and negative concentrations against 1F1 identities."""
    largest_error = 0.0
    for p in (2, 4, 16):
        a, c = 0.5, 0.5 * p
        mu = torch.zeros((1, p), dtype=DTYPE)
        mu[0, 0] = 1.0
        for kappa in (-20.0, -2.0, 0.0, 2.0, 20.0):
            model = Watson(
                p=p,
                K=1,
                params={
                    "mu": mu.clone(),
                    "kappa": torch.tensor([kappa], dtype=DTYPE),
                    "pi": torch.ones(1, dtype=DTYPE),
                },
            )
            log_constant = model.log_norm_constant().sum()
            gradient = torch.autograd.grad(log_constant, model.kappa)[0][0]
            hypergeometric = float(hyp1f1(a, c, kappa))
            exact_constant = _log_uniform_sphere_density(p) - math.log(hypergeometric)
            exact_gradient = -(
                (a / c)
                * float(hyp1f1(a + 1.0, c + 1.0, kappa))
                / hypergeometric
            )
            largest_error = max(
                largest_error,
                _assert_close(
                    f"Watson log constant (p={p}, kappa={kappa})",
                    log_constant.item(), exact_constant, 1e-8,
                ),
                _assert_close(
                    f"Watson derivative (p={p}, kappa={kappa})",
                    gradient.item(), exact_gradient, 1e-8,
                ),
            )
    return largest_error


def validate_bingham_p2() -> float:
    """Use the exact circular Bingham normalizer based on I0."""
    model = Bingham(p=2, rank=1, K=1, integration_points=400)
    largest_error = 0.0
    for values in ((0.0, 0.0), (0.0, -1.0), (0.0, -10.0), (-2.0, -7.0)):
        eigenvalues = torch.tensor([values], dtype=DTYPE, requires_grad=True)
        log_z = model._log_norm_from_eigenvalues(eigenvalues).sum()
        gradient = torch.autograd.grad(log_z, eigenvalues)[0][0]
        difference = 0.5 * (values[0] - values[1])
        bessel_ratio = float(i1e(difference) / i0e(difference))
        exact_log_z = (
            math.log(2.0 * math.pi)
            + 0.5 * (values[0] + values[1])
            + math.log(float(i0e(difference)))
            + abs(difference)
        )
        exact_gradient = np.array(
            [0.5 + 0.5 * bessel_ratio, 0.5 - 0.5 * bessel_ratio]
        )
        largest_error = max(
            largest_error,
            _assert_close(f"Bingham p=2 log Z {values}", log_z.item(), exact_log_z, 1e-10),
            _assert_close(
                f"Bingham p=2 gradient {values}",
                gradient.detach().numpy(), exact_gradient, 1e-10,
            ),
        )
    return largest_error


def validate_fisher_bingham_p2() -> float:
    """Compare value and gradients with periodic trapezoidal integration."""
    linear_value = np.array([1.2, -0.4])
    factor_value = np.array([[0.7], [0.25]])
    model = FisherBingham(p=2, rank=1, K=1, integration_points=400)
    linear = torch.tensor(linear_value[None, :], dtype=DTYPE, requires_grad=True)
    factor = torch.tensor(factor_value[None, :, :], dtype=DTYPE, requires_grad=True)
    log_z = model._sphere_log_normalizer(linear, factor).sum()
    linear_gradient, factor_gradient = torch.autograd.grad(log_z, (linear, factor))

    number_of_angles = 1 << 16
    angles = np.arange(number_of_angles) * (2.0 * math.pi / number_of_angles)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    log_weights = points @ linear_value - np.sum((points @ factor_value) ** 2, axis=1)
    shift = float(log_weights.max())
    weights = np.exp(log_weights - shift)
    exact_log_z = math.log(2.0 * math.pi) + shift + math.log(float(weights.mean()))
    exact_linear_gradient = (weights[:, None] * points).sum(axis=0) / weights.sum()
    second_moment = np.einsum("n,ni,nj->ij", weights, points, points) / weights.sum()
    exact_factor_gradient = -2.0 * second_moment @ factor_value

    return max(
        _assert_close("Fisher--Bingham p=2 log Z", log_z.item(), exact_log_z, 2e-6),
        _assert_close(
            "Fisher--Bingham p=2 linear gradient",
            linear_gradient.detach().numpy()[0], exact_linear_gradient, 2e-6,
        ),
        _assert_close(
            "Fisher--Bingham p=2 quadratic-factor gradient",
            factor_gradient.detach().numpy()[0], exact_factor_gradient, 2e-6,
        ),
    )


def validate_acg_mapping() -> float:
    """Match sampler precision Omega to PCMM's covariance-factor gauge."""
    p, ratio = 4, 3.5
    modal_axis = np.eye(p)[:, 0]
    factor_value = math.sqrt(ratio - 1.0) * modal_axis[:, None]
    precision = np.eye(p) + (ratio - 1.0) * (
        np.eye(p) - np.outer(modal_axis, modal_axis)
    )
    model = ACG(
        p=p,
        rank=1,
        K=1,
        params={
            "M": torch.tensor(factor_value[None, :, :], dtype=DTYPE),
            "pi": torch.ones(1, dtype=DTYPE),
        },
    )
    rng = np.random.default_rng(1201)
    points = rng.normal(size=(32, p))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    pcmm = model.log_pdf(torch.tensor(points, dtype=DTYPE)).detach().numpy()[0]
    sampler = (
        _log_uniform_sphere_density(p)
        + 0.5 * np.linalg.slogdet(precision)[1]
        - 0.5 * p * np.log(np.einsum("ni,ij,nj->n", points, precision, points))
    )
    return _assert_close("ACG sampler-to-PCMM parameter map", pcmm, sampler, 1e-11)


def _uniform_stiefel(rng: np.random.Generator, n: int, p: int, q: int) -> np.ndarray:
    gaussian = rng.normal(size=(n, p, q))
    frames, triangular = np.linalg.qr(gaussian, mode="reduced")
    signs = np.where(
        np.diagonal(triangular, axis1=1, axis2=2) >= 0.0, 1.0, -1.0
    )
    return frames * signs[:, None, :]


def validate_macg_mapping() -> float:
    """Check the MACG precision/covariance gauge relative to normalized Haar."""
    p, q, ratio = 4, 2, 4.0
    modal_frame = np.eye(p)[:, :q]
    factor_value = math.sqrt(ratio - 1.0) * modal_frame
    precision = np.eye(p) + (ratio - 1.0) * (
        np.eye(p) - modal_frame @ modal_frame.T
    )
    model = MACG(
        p=p,
        q=q,
        rank=q,
        K=1,
        params={
            "M": torch.tensor(factor_value[None, :, :], dtype=DTYPE),
            "pi": torch.ones(1, dtype=DTYPE),
        },
    )
    frames = _uniform_stiefel(np.random.default_rng(1202), 32, p, q)
    pcmm = model.log_pdf(torch.tensor(frames, dtype=DTYPE)).detach().numpy()[0]
    projected_precision = np.einsum("npi,pr,nrj->nij", frames, precision, frames)
    sampler = (
        0.5 * q * np.linalg.slogdet(precision)[1]
        - 0.5 * p * np.linalg.slogdet(projected_precision)[1]
    )
    mapping_error = _assert_close(
        "MACG sampler-to-PCMM parameter map", pcmm, sampler, 1e-11
    )

    # At p=q the MACG density is identically uniform and its covariance is
    # unidentifiable; this documents why the benchmark requires p>q.
    square_frames = _uniform_stiefel(np.random.default_rng(1203), 16, 2, 2)
    square_model = MACG(
        p=2,
        q=2,
        rank=2,
        K=1,
        params={
            "M": torch.tensor([[[0.8, 0.1], [0.2, 0.5]]], dtype=DTYPE),
            "pi": torch.ones(1, dtype=DTYPE),
        },
    )
    square_log_pdf = square_model.log_pdf(
        torch.tensor(square_frames, dtype=DTYPE)
    ).detach().numpy()[0]
    return max(
        mapping_error,
        _assert_close("MACG p=q uniform degeneracy", square_log_pdf, 0.0, 1e-11),
    )


def validate_wrapped_normal_p2() -> float:
    """Integrate the radius-one p=2 density and compare its covered mass."""
    target_alignment = 0.2
    variance = -2.0 * math.log(target_alignment)
    model = WrappedNormal(
        p=2,
        rank=2,
        K=1,
        winding_radius=1,
        winding_chunk_size=9,
        max_winding_vectors=100,
    )
    model.unpack_params(
        {
            "mu": torch.zeros((1, 2), dtype=DTYPE),
            "M": torch.zeros((1, 2, 2), dtype=DTYPE),
            "gamma": torch.tensor([variance], dtype=DTYPE),
            "pi": torch.ones(1, dtype=DTYPE),
        }
    )
    grid_size = 128
    grid = -math.pi + (np.arange(grid_size) + 0.5) * (2.0 * math.pi / grid_size)
    first, second = np.meshgrid(grid, grid, indexing="ij")
    points = torch.tensor(
        np.column_stack((first.ravel(), second.ravel())), dtype=DTYPE
    )
    density = torch.exp(model.log_pdf(points)[0]).detach().numpy()
    integral = float(density.sum() * (2.0 * math.pi / grid_size) ** 2)
    covered_mass = float(
        erf(3.0 * math.pi / math.sqrt(2.0 * variance)) ** 2
    )
    return _assert_close(
        "wrapped-normal p=2 radius-one normalization",
        integral,
        covered_mass,
        2e-8,
    )


def validate_uniform_constants() -> float:
    """Check the surface-area versus normalized-Haar conventions explicitly."""
    p = 4
    points = torch.eye(p, dtype=DTYPE)
    expected_sphere = _log_uniform_sphere_density(p)

    watson = Watson(
        p=p,
        K=1,
        params={
            "mu": points[0][None, :],
            "kappa": torch.zeros(1, dtype=DTYPE),
            "pi": torch.ones(1, dtype=DTYPE),
        },
    )
    acg = ACG(
        p=p,
        rank=1,
        K=1,
        params={
            "M": torch.zeros((1, p, 1), dtype=DTYPE),
            "pi": torch.ones(1, dtype=DTYPE),
        },
    )
    sphere_errors = [
        _assert_close(
            "Watson uniform log density",
            watson.log_pdf(points).detach().numpy(), expected_sphere, 1e-11,
        ),
        _assert_close(
            "ACG uniform log density",
            acg.log_pdf(points).detach().numpy(), expected_sphere, 1e-11,
        ),
    ]

    # MACG is expressed relative to normalized Haar measure, so its uniform
    # density is one and its uniform log density is zero.
    frames = _uniform_stiefel(np.random.default_rng(1204), 8, 4, 2)
    macg = MACG(
        p=4,
        q=2,
        rank=2,
        K=1,
        params={
            "M": torch.zeros((1, 4, 2), dtype=DTYPE),
            "pi": torch.ones(1, dtype=DTYPE),
        },
    )
    haar_error = _assert_close(
        "MACG normalized-Haar uniform log density",
        macg.log_pdf(torch.tensor(frames, dtype=DTYPE)).detach().numpy(),
        0.0,
        1e-11,
    )
    torus_uniform = -2.0 * math.log(2.0 * math.pi)
    torus_error = _assert_close(
        "p=2 torus uniform log density",
        torus_uniform,
        math.log((2.0 * math.pi) ** -2),
        1e-15,
    )
    return max(*sphere_errors, haar_error, torus_error)


def main() -> None:
    torch.set_default_dtype(DTYPE)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    checks = (
        ("vMF Bessel value/gradient", validate_vmf),
        ("Watson 1F1 value/gradient", validate_watson),
        ("Bingham p=2 I0 value/gradient", validate_bingham_p2),
        ("Fisher--Bingham p=2 quadrature", validate_fisher_bingham_p2),
        ("ACG sampler/fit parameter map", validate_acg_mapping),
        ("MACG sampler/fit parameter map", validate_macg_mapping),
        ("wrapped-normal p=2 normalization", validate_wrapped_normal_p2),
        ("uniform measure conventions", validate_uniform_constants),
    )
    started = time.perf_counter()
    for label, function in checks:
        error = function()
        print(f"PASS  {label:<43} max error={error:.3e}")
    print(f"All {len(checks)} distribution-math checks passed in {time.perf_counter() - started:.2f} s.")


if __name__ == "__main__":
    main()
