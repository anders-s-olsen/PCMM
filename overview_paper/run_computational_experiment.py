#!/usr/bin/env python3
"""Fit every matched sampler cell and record reproducible runtime diagnostics.

The input is the completed schema-v2 ``manifest.csv`` written by
``directional_sampler.py``.  The primary estimands are fitting wall time
(initialization, optimizer updates, and fitting finalization) and
warm-up-excluded median time per optimizer update.  A fixed-state objective
value-and-gradient benchmark and a parameter-normalizer value-and-gradient
benchmark separate implementation cost from the optimizer path.  Model
construction, sample loading, held-out evaluation, and serialization are timed
separately or excluded.  Iterations, stopping outcome, stationarity, and
held-out score are reported as quality-control measures.  A settings/source fingerprint prevents
``--resume`` from silently mixing incompatible runs.

Examples
--------
python overview_paper/run_computational_experiment.py --manifest samples/manifest.csv
python overview_paper/run_computational_experiment.py --manifest samples/manifest.csv --fit-repeats 5
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from PCMM.PCMMtorch import ACG, Bingham, MACG, Watson, WrappedNormal
from PCMM.PCMMtorchAdditional import (
    FisherBingham,
    MatrixBingham,
    MatrixFisher,
    MatrixFisherBingham,
    VonMisesFisher,
)
from PCMM.mixture_torch_loop import mixture_torch_loop


SUPPORTED_DISTRIBUTIONS = {
    'wrapped_normal',
    'vmf',
    'watson',
    'bingham',
    'fisher_bingham',
    'acg',
    'matrix_fisher',
    'matrix_bingham',
    'matrix_fisher_bingham',
    'macg',
}

RESULT_SCHEMA_VERSION = 4
SAMPLER_SCHEMA_VERSION = 2
TERMINAL_STATUSES = {'ok', 'skipped', 'time_limit'}
REQUIRED_SAMPLER_COLUMNS = {
    'sampler_schema_version',
    'distribution',
    'ambient_dim',
    'matrix_cols',
    'rank_label',
    'concentration_label',
    'target_alignment',
    'observed_alignment',
    'alignment_qc_status',
    'calibration_status',
    'n_samples',
    'n_train',
    'n_test',
    'split_index',
    'iid',
    'mcmc_qc_status',
    'sampler_qc_status',
    'filename',
}


class SkipCell(RuntimeError):
    """Expected scientific or computational exclusion, recorded as skipped."""


@contextlib.contextmanager
def _best_effort_wall_timeout(seconds: float | None, label: str):
    """Interrupt long Python-level validation work on POSIX systems.

    This guard complements, but does not replace, the optimizer's soft budget:
    some native numerical kernels defer Python signal handling until they
    return.  The batch scheduler wall-time remains the final process-level cap.
    """
    if seconds is None or not hasattr(signal, 'setitimer'):
        yield
        return

    def _raise_timeout(signum, frame):  # pragma: no cover - timing dependent
        del signum, frame
        raise TimeoutError(f'{label} exceeded {seconds:g} seconds.')

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.perf_counter()
    try:
        yield
    finally:
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0.0:
            previous_delay = max(
                previous_delay - (time.perf_counter() - started),
                1e-6,
            )
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)


def _comma_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(',') if item.strip()}


def _optional_ints(value: str) -> set[int]:
    return {int(item) for item in _comma_list(value)}


def _row_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, '')
    return default if value in {'', None} else int(value)


def _row_bool(row: dict[str, str], key: str, default: bool = True) -> bool:
    value = row.get(key, '')
    if value in {'', None}:
        return default
    return value.strip().lower() in {'1', 'true', 'yes'}


def _resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false.')
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _warm_up_timing_runtime(device: torch.device, dtype: torch.dtype) -> float:
    """Pay one-time PyTorch/autograd/Adam startup before measured attempts."""
    started = time.perf_counter()
    parameter = torch.nn.Parameter(
        torch.linspace(0.1, 0.4, 4, dtype=dtype, device=device)
    )
    optimizer = torch.optim.Adam((parameter,), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.square().sum() + torch.logsumexp(parameter, dim=0)
        loss.backward()
        optimizer.step()
    _synchronize(device)
    return time.perf_counter() - started


def _fit_initialization_seed(
    base_seed: int,
    fit_repeat: int,
    repeat_initialization: str,
) -> int:
    """Return a paired seed for one independent initialization replicate.

    Comparable cells use the same seed for a given repeat.  This common-random-
    numbers design removes arbitrary manifest position from initialization
    while retaining genuinely different orientations across fit repeats.
    """
    if repeat_initialization == 'fixed':
        return base_seed
    if repeat_initialization == 'independent':
        return base_seed + fit_repeat * 9176
    raise ValueError(f'Unknown repeat initialization scheme: {repeat_initialization}')


def _load_samples(row: dict[str, str], manifest_dir: Path, dtype: torch.dtype) -> torch.Tensor:
    path = manifest_dir / row['filename']
    values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    p = _row_int(row, 'ambient_dim')
    q = _row_int(row, 'matrix_cols')
    if q:
        expected_columns = p * q
        if values.shape[1] != expected_columns:
            raise ValueError(f'{path} has {values.shape[1]} columns; expected {expected_columns}.')
        values = values.reshape(values.shape[0], p, q)
    elif values.shape[1] != p:
        raise ValueError(f'{path} has {values.shape[1]} columns; expected {p}.')
    return torch.as_tensor(values, dtype=dtype)


def _split_samples(
    X: torch.Tensor,
    test_fraction: float,
    seed: int,
    iid: bool,
    row: dict[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    # New sampler manifests place independent training draws (or four MCMC
    # chains) first and the held-out draws (or fifth chain) last.  Respecting
    # that boundary prevents a later random split from leaking autocorrelated
    # MCMC draws across training and test sets.
    if row is not None:
        declared_train = _row_int(row, 'n_train')
        declared_test = _row_int(row, 'n_test')
        split_index = _row_int(row, 'split_index', default=declared_train)
        if declared_train and declared_test:
            if declared_train + declared_test != X.shape[0] or split_index != declared_train:
                raise ValueError(
                    'Sampler-declared n_train, n_test, and split_index do not match the sample file.'
                )
            method = 'sampler_iid_boundary' if iid else 'heldout_mcmc_chain'
            return X[:split_index], X[split_index:], method

    n_test = max(1, int(round(test_fraction * X.shape[0])))
    if X.shape[0] - n_test < 2:
        raise SkipCell('At least two training observations and one test observation are required.')
    if iid:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        order = torch.randperm(X.shape[0], generator=generator)
        return X[order[n_test:]], X[order[:n_test]], 'random_iid'
    return X[:-n_test], X[-n_test:], 'contiguous_mcmc'


def _model_rank(row: dict[str, str]) -> int:
    """Translate the sampler's penalized rank to each PCMM factor rank."""
    explicit_rank = _row_int(row, 'model_factor_rank')
    if explicit_rank:
        return explicit_rank
    distribution = row['distribution']
    active_rank = _row_int(row, 'active_rank')
    modal_rank = _row_int(row, 'modal_dim')
    if distribution in {'bingham', 'acg', 'macg'}:
        return modal_rank
    return active_rank


def _require_sampling_qc(row: dict[str, str]) -> None:
    checks = {
        'alignment_qc_status': {'pass'},
        'calibration_status': {'pass', 'alignment_tolerance_warning'},
        'sampler_qc_status': {'pass', 'warning'},
        'mcmc_qc_status': {'pass', 'warning', 'not_applicable'},
    }
    failures = [
        f'{field}={row.get(field)}'
        for field, accepted in checks.items()
        if str(row.get(field, '')) not in accepted
    ]
    if failures:
        raise SkipCell('Sampler quality control did not pass: ' + ', '.join(failures))


def _wrapped_radius(row: dict[str, str], args: argparse.Namespace) -> tuple[int, str]:
    p = _row_int(row, 'ambient_dim')
    requested_vectors = (2 * args.wrapped_winding_radius + 1) ** p
    if requested_vectors <= args.max_winding_vectors:
        return args.wrapped_winding_radius, f'finite_winding_lattice_radius_{args.wrapped_winding_radius}'
    if args.wrapped_mode == 'central':
        return 0, 'central_winding_only_not_normalized_on_torus'
    message = (
        f'Wrapped-normal lattice has {requested_vectors:,} vectors, above --max-winding-vectors. '
        'Use --wrapped-mode central only if a local Gaussian approximation is scientifically acceptable.'
    )
    raise SkipCell(message)


def _build_model(row: dict[str, str], args: argparse.Namespace) -> tuple[torch.nn.Module, str, int]:
    distribution = row['distribution']
    p = _row_int(row, 'ambient_dim')
    q = _row_int(row, 'matrix_cols')
    rank = _model_rank(row)
    common = {'K': args.components, 'HMM': False}

    if distribution == 'vmf':
        if torch.device(args.device).type != 'cpu':
            raise RuntimeError(
                'vMF fitting uses SciPy modified-Bessel functions and is CPU-only; '
                'select --device cpu for vMF cells.'
            )
        model = VonMisesFisher(p=p, **common)
    elif distribution == 'watson':
        model = Watson(p=p, tol=args.tolerance, **common)
    elif distribution == 'bingham':
        model = Bingham(p=p, rank=rank, integration_points=args.integration_points, **common)
    elif distribution == 'fisher_bingham':
        model = FisherBingham(p=p, rank=rank, integration_points=args.integration_points, **common)
    elif distribution == 'acg':
        model = ACG(p=p, rank=rank, **common)
    elif distribution == 'matrix_fisher':
        linear_rank = _row_int(row, 'linear_rank', default=max(1, rank))
        model = MatrixFisher(
            p=p,
            q=q,
            linear_rank=linear_rank,
            saddlepoint_iterations=args.saddlepoint_iterations,
            saddlepoint_tolerance=args.saddlepoint_tolerance,
            saddlepoint_order=args.saddlepoint_order,
            saddlepoint_derivative_backend=args.saddlepoint_derivative_backend,
            saddlepoint_finite_difference_step=(
                args.saddlepoint_finite_difference_step
            ),
            direct_linear_parameterization=args.direct_matrix_linear_parameterization,
            **common,
        )
    elif distribution == 'matrix_bingham':
        model = MatrixBingham(
            p=p,
            q=q,
            rank=rank,
            saddlepoint_iterations=args.saddlepoint_iterations,
            saddlepoint_tolerance=args.saddlepoint_tolerance,
            saddlepoint_order=args.saddlepoint_order,
            saddlepoint_derivative_backend=args.saddlepoint_derivative_backend,
            saddlepoint_finite_difference_step=(
                args.saddlepoint_finite_difference_step
            ),
            **common,
        )
    elif distribution == 'matrix_fisher_bingham':
        model = MatrixFisherBingham(
            p=p,
            q=q,
            rank=rank,
            linear_rank=_row_int(row, 'linear_rank'),
            saddlepoint_iterations=args.saddlepoint_iterations,
            saddlepoint_tolerance=args.saddlepoint_tolerance,
            saddlepoint_order=args.saddlepoint_order,
            saddlepoint_derivative_backend=args.saddlepoint_derivative_backend,
            saddlepoint_finite_difference_step=(
                args.saddlepoint_finite_difference_step
            ),
            direct_linear_parameterization=args.direct_matrix_linear_parameterization,
            **common,
        )
    elif distribution == 'macg':
        model = MACG(p=p, q=q, rank=rank, **common)
    elif distribution == 'wrapped_normal':
        radius, normalizer = _wrapped_radius(row, args)
        model = WrappedNormal(
            p=p,
            rank=rank,
            winding_radius=radius,
            winding_chunk_size=args.winding_chunk_size,
            max_winding_vectors=args.max_winding_vectors,
            **common,
        )
        return model, normalizer, rank
    else:
        raise SkipCell(f'Unsupported distribution: {distribution}')

    normalizer = getattr(model, 'normalizer_kind', _normalizer_kind(distribution))
    return model, normalizer, rank


def _normalizer_kind(distribution: str) -> str:
    kinds = {
        'watson': 'kummer_series',
        'bingham': 'continuous_euler_quadrature',
        'acg': 'closed_form',
        'macg': 'closed_form_relative_to_normalized_haar',
    }
    return kinds.get(distribution, 'unknown')


def _intrinsic_dimension(row: dict[str, str]) -> int:
    distribution = row['distribution']
    p = _row_int(row, 'ambient_dim')
    q = _row_int(row, 'matrix_cols')
    if distribution == 'wrapped_normal':
        return p
    if distribution in {'matrix_bingham', 'macg'}:
        return q * (p - q)
    if distribution in {'matrix_fisher', 'matrix_fisher_bingham'}:
        return p * q - q * (q + 1) // 2
    return p - 1


def _uniform_log_density(row: dict[str, str]) -> tuple[float, int]:
    distribution = row['distribution']
    p = _row_int(row, 'ambient_dim')
    q = _row_int(row, 'matrix_cols')
    intrinsic_dimension = _intrinsic_dimension(row)
    if distribution == 'wrapped_normal':
        return -p * math.log(2.0 * math.pi), intrinsic_dimension
    if q:
        return 0.0, intrinsic_dimension
    log_uniform = math.lgamma(p / 2.0) - math.log(2.0) - p / 2.0 * math.log(math.pi)
    return log_uniform, intrinsic_dimension


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _uses_matrix_saddlepoint(model: torch.nn.Module) -> bool:
    return hasattr(model, 'saddlepoint_order')


def _parameter_normalizer_expression(
    model: torch.nn.Module,
    distribution: str,
) -> tuple[torch.Tensor, str]:
    """Return the parameter-only normalizer term used by one component.

    The sign of the returned scalar is immaterial for timing.  For ACG, MACG,
    and the wrapped normal, the expression is the parameter-dependent closed-
    form determinant contribution.  The wrapped density's expensive winding
    sum remains part of the fixed-state objective benchmark, where it belongs.
    """
    if distribution in {
        'vmf',
        'fisher_bingham',
        'matrix_fisher',
        'matrix_bingham',
        'matrix_fisher_bingham',
    }:
        return model.log_norm_constant().sum(), str(model.normalizer_kind)
    if distribution == 'watson':
        return model.log_norm_constant().sum(), 'kummer_series'
    if distribution == 'bingham':
        _, eigenvalues = model._concentration_and_eigenvalues()
        return (
            model._log_norm_from_eigenvalues(eigenvalues).sum(),
            'eigendecomposition_plus_continuous_euler_quadrature',
        )
    if distribution == 'acg':
        identity = torch.eye(
            model.r, dtype=model.M.dtype, device=model.M.device
        )
        determinant = torch.linalg.slogdet(model.M.mH @ model.M + identity).logabsdet
        return model.a.to(determinant) * determinant.sum(), 'closed_form_log_determinant'
    if distribution == 'macg':
        identity = torch.eye(
            model.r, dtype=model.M.dtype, device=model.M.device
        )
        determinant = torch.linalg.slogdet(model.M.mT @ model.M + identity).logabsdet
        return (model.q / 2.0) * determinant.sum(), 'closed_form_log_determinant'
    if distribution == 'wrapped_normal':
        gamma = torch.nn.functional.softplus(model.gamma)
        if model.force_gamma_same:
            gamma = gamma.mean().expand_as(gamma)
        scaled = model.M / torch.sqrt(gamma[:, None, None])
        identity = torch.eye(
            model.r, dtype=model.M.dtype, device=model.M.device
        )
        determinant = (
            model.p * torch.log(gamma)
            + torch.linalg.slogdet(scaled.mT @ scaled + identity).logabsdet
        )
        return 0.5 * determinant.sum(), 'closed_form_gaussian_log_determinant'
    raise ValueError(f'No parameter-normalizer benchmark is defined for {distribution}.')


def _gradient_diagnostics(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the restored state and report parameterization-scale QC."""
    model.zero_grad(set_to_none=True)
    _synchronize(device)
    objective = -model(X)
    objective.backward()
    _synchronize(device)
    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {
            'final_gradient_norm': float('nan'),
            'final_gradient_norm_per_observation': float('nan'),
            'final_gradient_rms_per_parameter_per_observation': float('nan'),
            'final_gradient_max_abs_per_observation': float('nan'),
        }
    vector = torch.cat(gradients)
    n = float(X.shape[0])
    return {
        'final_gradient_norm': float(torch.linalg.vector_norm(vector).cpu()),
        'final_gradient_norm_per_observation': float(torch.linalg.vector_norm(vector).cpu()) / n,
        'final_gradient_rms_per_parameter_per_observation': float(torch.sqrt(vector.square().mean()).cpu()) / n,
        'final_gradient_max_abs_per_observation': float(vector.abs().max().cpu()) / n,
    }


def _timing_quantiles(values: list[float], prefix: str) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f'{prefix}_median_seconds': '',
            f'{prefix}_q1_seconds': '',
            f'{prefix}_q3_seconds': '',
        }
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    return {
        f'{prefix}_median_seconds': float(median),
        f'{prefix}_q1_seconds': float(q1),
        f'{prefix}_q3_seconds': float(q3),
    }


def _benchmark_value_and_gradient(
    model: torch.nn.Module,
    expression,
    device: torch.device,
    repetitions: int,
    warmup: int,
    max_seconds: float,
    reset_before_call=None,
) -> dict[str, Any]:
    """Time repeated scalar values plus reverse-mode parameter gradients.

    A bounded number of whole evaluations is used instead of interrupting a
    numerical kernel mid-call.  At least one measured evaluation is retained;
    the count makes an early benchmark budget stop explicit.
    """
    for _ in range(warmup):
        if reset_before_call is not None:
            reset_before_call()
        model.zero_grad(set_to_none=True)
        value = expression()
        if value.ndim:
            value = value.sum()
        value.backward()
        _synchronize(device)

    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    benchmark_started = time.perf_counter()
    for _ in range(repetitions):
        if reset_before_call is not None:
            reset_before_call()
        model.zero_grad(set_to_none=True)
        _synchronize(device)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        value = expression()
        if value.ndim:
            value = value.sum()
        value.backward()
        _synchronize(device)
        wall_seconds.append(time.perf_counter() - wall_started)
        cpu_seconds.append(time.process_time() - cpu_started)
        if time.perf_counter() - benchmark_started >= max_seconds:
            break

    result = {
        'evaluations': len(wall_seconds),
        'requested_evaluations': repetitions,
        'warmup_evaluations': warmup,
        'budget_reached': len(wall_seconds) < repetitions,
    }
    result.update(_timing_quantiles(wall_seconds, 'wall'))
    result.update(_timing_quantiles(cpu_seconds, 'cpu'))
    return result


def _prefixed(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f'{prefix}_{key}': value for key, value in values.items()}


def _cpu_parameters(params: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in params.items():
        result[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validate_sampler_dataset(
    manifest: Path,
    rows: list[dict[str, str]],
) -> None:
    """Reject legacy, partial, or internally inconsistent sampler outputs."""
    if not rows:
        raise ValueError(f'Sampler manifest is empty: {manifest}')
    missing = sorted(REQUIRED_SAMPLER_COLUMNS - set(rows[0]))
    if missing:
        raise ValueError(
            'The sampler manifest predates the controlled benchmark schema or is incomplete; '
            f'missing columns: {missing}. Regenerate it with directional_sampler.py.'
        )
    versions = {str(row.get('sampler_schema_version', '')) for row in rows}
    if versions != {str(SAMPLER_SCHEMA_VERSION)}:
        raise ValueError(
            f'Expected sampler schema {SAMPLER_SCHEMA_VERSION}, found {sorted(versions)}.'
        )

    completion_path = manifest.parent / 'sampling_complete.json'
    if not completion_path.exists():
        raise ValueError(
            f'Missing {completion_path.name}; the sampling job may be incomplete. '
            'Regenerate or finish the sample set before fitting.'
        )
    completion = json.loads(completion_path.read_text())
    if int(completion.get('sampler_schema_version', -1)) != SAMPLER_SCHEMA_VERSION:
        raise ValueError('The sampling completion marker has an incompatible schema version.')
    if int(completion.get('generated_cells', -1)) != len(rows):
        raise ValueError('The sampling completion marker does not match the manifest row count.')
    if int(completion.get('planned_cells', -1)) != len(rows):
        raise ValueError('The sampling job did not complete every planned experiment cell.')
    if completion.get('manifest_sha256') != _sha256_file(manifest):
        raise ValueError('The sampler manifest changed after its completion marker was written.')

    filenames: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        filename = row.get('filename', '')
        if not filename or filename in filenames:
            raise ValueError(f'Missing or duplicate filename at manifest line {line_number}.')
        filenames.add(filename)
        n_samples = _row_int(row, 'n_samples')
        n_train = _row_int(row, 'n_train')
        n_test = _row_int(row, 'n_test')
        split_index = _row_int(row, 'split_index')
        if n_samples < 3 or n_train + n_test != n_samples or split_index != n_train:
            raise ValueError(
                f'Invalid sampler-declared train/test split at manifest line {line_number}.'
            )


def _sample_dataset_hash(
    manifest: Path,
    selected: list[tuple[int, dict[str, str]]],
) -> str:
    digest = hashlib.sha256()
    digest.update(_sha256_file(manifest).encode())
    for _, row in sorted(selected, key=lambda item: item[1]['filename']):
        sample_path = manifest.parent / row['filename']
        if not sample_path.exists():
            raise FileNotFoundError(f'Manifest sample file is missing: {sample_path}')
        digest.update(row['filename'].encode())
        digest.update(_sha256_file(sample_path).encode())
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ['git', *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'
    return completed.stdout.strip()


def _cpu_model() -> str:
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors='replace').splitlines():
            if line.lower().startswith('model name'):
                return line.split(':', 1)[-1].strip()
    return platform.processor() or 'unknown'


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _run_identity(
    args: argparse.Namespace,
    manifest_hash: str,
    sample_dataset_hash: str,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    excluded = {'output_dir', 'resume', 'resolved_device'}
    settings = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if key not in excluded
    }
    settings['resolved_device'] = args.resolved_device
    settings['manifest_sha256'] = manifest_hash
    settings['sample_dataset_sha256'] = sample_dataset_hash
    settings['result_schema_version'] = RESULT_SCHEMA_VERSION
    settings['runtime_identity'] = {
        'cpu_model': _cpu_model(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'torch_version': str(torch.__version__),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
        'thread_environment': {
            name: os.environ.get(name, '')
            for name in (
                'OMP_NUM_THREADS',
                'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS',
            )
        },
    }

    source_paths = {
        'runner': Path(__file__).resolve(),
        'directional_sampler': (Path(__file__).resolve().parent / 'directional_sampler.py'),
        'mixture_torch_loop': (Path(__file__).resolve().parents[1] / 'PCMM/mixture_torch_loop.py'),
        'mixture_EM_loop': (Path(__file__).resolve().parents[1] / 'PCMM/mixture_EM_loop.py'),
        'PCMMtorch': (Path(__file__).resolve().parents[1] / 'PCMM/PCMMtorch.py'),
        'PCMMtorchAdditional': (Path(__file__).resolve().parents[1] / 'PCMM/PCMMtorchAdditional.py'),
        'PCMMtorchBaseModel': (Path(__file__).resolve().parents[1] / 'PCMM/PCMMtorchBaseModel.py'),
        'PCMMnumpy': (Path(__file__).resolve().parents[1] / 'PCMM/PCMMnumpy.py'),
        'PCMMnumpyBaseModel': (Path(__file__).resolve().parents[1] / 'PCMM/PCMMnumpyBaseModel.py'),
    }
    source_hashes = {
        name: _sha256_file(path) if path.exists() else 'missing'
        for name, path in source_paths.items()
    }
    payload = {'settings': settings, 'source_sha256': source_hashes}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()[:16], payload, source_hashes


def _structural_warmup_key(
    row: dict[str, str],
    X_train: torch.Tensor,
) -> tuple[Any, ...]:
    """Identify cells that execute the same model-shaped fitting kernels."""
    return (
        row['distribution'],
        _row_int(row, 'ambient_dim'),
        _row_int(row, 'matrix_cols'),
        _model_rank(row),
        _row_int(row, 'linear_rank'),
        X_train.shape[0],
    )


def _warm_up_model_configuration(
    row: dict[str, str],
    X_train: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Run and discard one complete optimizer update outside benchmark timing."""
    started = time.perf_counter()
    model, _, _ = _build_model(row, args)
    model = model.to(device=device)
    data = X_train.to(device=device)
    output_context = (
        contextlib.nullcontext()
        if args.show_progress
        else contextlib.redirect_stdout(io.StringIO())
    )
    with output_context, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mixture_torch_loop(
            model=model,
            data=data,
            tol=args.tolerance,
            max_iter=1,
            num_repl=1,
            init=None if args.initialization == 'default' else args.initialization,
            LR=args.learning_rate,
            suppress_output=not args.show_progress,
            threads=args.threads,
            num_comparison=args.convergence_window,
            convergence_normalization=args.convergence_normalization,
            intrinsic_dimension=_intrinsic_dimension(row),
            return_diagnostics=False,
            timing_warmup_iterations=0,
            initialization_tol=args.initialization_tolerance,
            max_optimization_seconds=None,
            max_learning_rate_reductions=0,
            learning_rate_reduction_factor=args.learning_rate_reduction_factor,
            isotropic_natural_scale=args.isotropic_natural_scale,
            isotropic_wrapped_variance=args.isotropic_wrapped_variance,
        )
    _synchronize(device)
    return time.perf_counter() - started


def _fit_one(
    row: dict[str, str],
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    setup_started = time.perf_counter()
    model, normalizer, model_rank = _build_model(row, args)
    model = model.to(device=device)
    X_train = X_train.to(device=device)
    X_test = X_test.to(device=device)
    _synchronize(device)
    setup_seconds = time.perf_counter() - setup_started

    intrinsic_dimension = _intrinsic_dimension(row)
    output_context = (
        contextlib.nullcontext()
        if args.show_progress
        else contextlib.redirect_stdout(io.StringIO())
    )
    _synchronize(device)
    fit_started = time.perf_counter()
    with output_context, warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'No initialization method was specified for .*',
            category=UserWarning,
        )
        params, _, log_likelihood, diagnostics = mixture_torch_loop(
            model=model,
            data=X_train,
            tol=args.tolerance,
            max_iter=args.max_iterations,
            num_repl=args.optimizer_restarts,
            init=None if args.initialization == 'default' else args.initialization,
            LR=args.learning_rate,
            suppress_output=not args.show_progress,
            threads=args.threads,
            decrease_lr_on_plateau=args.decrease_lr_on_plateau,
            num_comparison=args.convergence_window,
            convergence_normalization=args.convergence_normalization,
            intrinsic_dimension=intrinsic_dimension,
            return_diagnostics=True,
            timing_warmup_iterations=args.timing_warmup_iterations,
            initialization_tol=args.initialization_tolerance,
            max_optimization_seconds=args.max_fit_seconds,
            max_learning_rate_reductions=args.max_learning_rate_reductions,
            learning_rate_reduction_factor=args.learning_rate_reduction_factor,
            isotropic_natural_scale=args.isotropic_natural_scale,
            isotropic_wrapped_variance=args.isotropic_wrapped_variance,
        )
    _synchronize(device)
    total_fit_seconds = time.perf_counter() - fit_started

    evaluation_started = time.perf_counter()
    with torch.no_grad():
        train_total = float(model(X_train).detach().cpu())
        test_total_tensor, _ = model.test_log_likelihood(X_test)
        test_total = float(test_total_tensor.detach().cpu())
    _synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    mean_train = train_total / X_train.shape[0]
    mean_test = test_total / X_test.shape[0]

    order1_mean_test: float | str = ''
    order2_mean_test: float | str = ''
    order2_difference: float | str = ''
    matrix_order_seconds: float | str = ''
    matrix_order_error = ''
    if _uses_matrix_saddlepoint(model):
        if args.saddlepoint_order == 1:
            order1_mean_test = mean_test
        else:
            order2_mean_test = mean_test
    if (
        not args.skip_second_order_validation
        and _uses_matrix_saddlepoint(model)
    ):
        validation_started = time.perf_counter()
        original_order = model.saddlepoint_order
        original_kind = model.normalizer_kind
        alternate_order = 2 if original_order == 1 else 1
        try:
            model.saddlepoint_order = alternate_order
            order_name = 'second' if alternate_order == 2 else 'first'
            model.normalizer_kind = (
                f'{order_name}_order_saddlepoint_relative_to_normalized_haar'
            )
            model._saddle_cache = None
            with _best_effort_wall_timeout(
                args.second_order_validation_max_seconds,
                'Matrix normalizer-order sensitivity evaluation',
            ):
                with torch.no_grad():
                    alternate_total, _ = model.test_log_likelihood(X_test)
            _synchronize(device)
            alternate_mean = float(alternate_total.detach().cpu()) / X_test.shape[0]
            if alternate_order == 1:
                order1_mean_test = alternate_mean
            else:
                order2_mean_test = alternate_mean
            if isinstance(order1_mean_test, float) and isinstance(order2_mean_test, float):
                order2_difference = order2_mean_test - order1_mean_test
        except Exception as error:  # Sensitivity failure must not discard a valid fit.
            matrix_order_error = f'{type(error).__name__}: {str(error)[:500]}'
        finally:
            model.saddlepoint_order = original_order
            model.normalizer_kind = original_kind
            model._saddle_cache = None
            matrix_order_seconds = time.perf_counter() - validation_started

    gradient_qc = _gradient_diagnostics(model, X_train, device)
    gradient_rms = gradient_qc['final_gradient_rms_per_parameter_per_observation']
    stationarity_pass = bool(
        np.isfinite(gradient_rms) and gradient_rms <= args.stationarity_tolerance
    )

    objective_benchmark = _benchmark_value_and_gradient(
        model=model,
        expression=lambda: -model(X_train),
        device=device,
        repetitions=args.benchmark_repetitions,
        warmup=args.benchmark_warmup_evaluations,
        max_seconds=args.benchmark_max_seconds,
    )
    normalizer_expression = lambda: _parameter_normalizer_expression(
        model, row['distribution']
    )[0]
    _, normalizer_benchmark_kind = _parameter_normalizer_expression(
        model, row['distribution']
    )
    normalizer_benchmark = _benchmark_value_and_gradient(
        model=model,
        expression=normalizer_expression,
        device=device,
        repetitions=args.benchmark_repetitions,
        warmup=args.benchmark_warmup_evaluations,
        max_seconds=args.benchmark_max_seconds,
    )
    cold_normalizer_benchmark: dict[str, Any] = {}
    if _uses_matrix_saddlepoint(model):
        cold_normalizer_benchmark = _benchmark_value_and_gradient(
            model=model,
            expression=normalizer_expression,
            device=device,
            repetitions=args.cold_saddle_benchmark_repetitions,
            warmup=0,
            max_seconds=args.benchmark_max_seconds,
            reset_before_call=lambda: setattr(model, '_saddle_cache', None),
        )

    uniform_log_density, intrinsic_dimension = _uniform_log_density(row)
    gain = mean_test - uniform_log_density
    stopping_reason = diagnostics['stopping_reason']
    fit_outcome = (
        'objective_plateau'
        if diagnostics['converged']
        else 'optimization_time_limit'
        if stopping_reason == 'max_optimization_seconds'
        else 'max_iterations'
        if stopping_reason == 'max_iter'
        else 'nonconverged'
    )
    matrix_order_elapsed = (
        matrix_order_seconds if isinstance(matrix_order_seconds, float) else 0.0
    )
    statistics = {
        # fit_seconds now means optimization only; initialization and the
        # combined wall time are reported separately.
        'fit_seconds': diagnostics['optimization_seconds'],
        'setup_seconds': setup_seconds,
        'initialization_seconds': diagnostics['initialization_seconds'],
        'post_update_evaluation_seconds': diagnostics['post_update_evaluation_seconds'],
        'finalization_seconds': diagnostics['finalization_seconds'],
        'total_fit_seconds': total_fit_seconds,
        'evaluation_seconds': evaluation_seconds,
        'matrix_order_sensitivity_seconds': matrix_order_seconds,
        'second_order_validation_seconds': matrix_order_seconds,
        'mean_heldout_log_likelihood_saddlepoint_order1': order1_mean_test,
        'mean_heldout_log_likelihood_saddlepoint_order2': order2_mean_test,
        'order2_minus_order1_heldout_log_likelihood': order2_difference,
        'matrix_order_sensitivity_error': matrix_order_error,
        'second_order_validation_error': matrix_order_error,
        'fit_plus_evaluation_seconds': setup_seconds + total_fit_seconds + evaluation_seconds,
        'pre_serialization_seconds': (
            setup_seconds + total_fit_seconds + evaluation_seconds + matrix_order_elapsed
        ),
        'optimizer_overhead_seconds': diagnostics['unattributed_overhead_seconds'],
        'timing_accounting_residual_seconds': (
            total_fit_seconds
            - diagnostics['initialization_seconds']
            - diagnostics['optimization_seconds']
            - diagnostics['post_update_evaluation_seconds']
            - diagnostics['unattributed_overhead_seconds']
            - diagnostics['finalization_seconds']
        ),
        'seconds_per_iteration': diagnostics['optimization_seconds'] / max(len(log_likelihood), 1),
        'median_seconds_per_iteration': diagnostics['median_iteration_seconds'],
        'median_iteration_cpu_seconds': diagnostics['median_iteration_cpu_seconds'],
        'median_objective_forward_seconds': diagnostics['median_objective_forward_seconds'],
        'median_objective_backward_seconds': diagnostics['median_objective_backward_seconds'],
        'median_parameter_snapshot_seconds': diagnostics['median_parameter_snapshot_seconds'],
        'median_optimizer_step_seconds': diagnostics['median_optimizer_step_seconds'],
        'median_seconds_per_iteration_per_training_observation': (
            diagnostics['median_iteration_seconds'] / X_train.shape[0]
        ),
        'timing_warmup_iterations_used': diagnostics['timing_warmup_iterations'],
        'iterations': len(log_likelihood),
        'optimizer_normalizer_evaluations': len(log_likelihood) + 1,
        'stopping_reason': stopping_reason,
        'fit_outcome': fit_outcome,
        'converged': diagnostics['converged'],
        'convergence_normalization': diagnostics['convergence_normalization'],
        'convergence_scale': diagnostics['convergence_scale'],
        'objective_plateau_reached': diagnostics['converged'],
        'learning_rate_reductions': diagnostics['learning_rate_reductions'],
        'final_learning_rate': diagnostics['final_learning_rate'],
        'stationarity_tolerance': args.stationarity_tolerance,
        'stationarity_qc_pass': stationarity_pass,
        'last_iteration_gradient_norm': diagnostics['last_iteration_gradient_norm'],
        'last_pre_update_gradient_norm': diagnostics['last_iteration_gradient_norm'],
        'final_optimizer_log_likelihood': diagnostics['post_update_log_likelihood'],
        'best_optimizer_log_likelihood': diagnostics['best_log_likelihood'],
        'initial_optimizer_log_likelihood': float(log_likelihood[0]),
        'initial_mean_train_log_likelihood': float(log_likelihood[0]) / X_train.shape[0],
        'initial_train_log_score_gain_over_uniform': (
            float(log_likelihood[0]) / X_train.shape[0] - uniform_log_density
        ),
        'optimizer_gain_per_observation': (
            diagnostics['best_log_likelihood'] - float(log_likelihood[0])
        ) / X_train.shape[0],
        'mean_train_log_likelihood': mean_train,
        'mean_heldout_log_likelihood': mean_test,
        'uniform_mean_log_likelihood': uniform_log_density,
        'heldout_log_score_gain': gain,
        'heldout_log_score_gain_per_dof': gain / intrinsic_dimension,
        'train_heldout_log_score_gap': mean_train - mean_test,
        'intrinsic_support_dimension': intrinsic_dimension,
        'parameter_count': _parameter_count(model),
        'model_factor_rank': model_rank,
        'normalizer_kind': normalizer,
        'normalizer_benchmark_kind': normalizer_benchmark_kind,
        'saddlepoint_order': (
            getattr(model, 'saddlepoint_order', '')
            if _uses_matrix_saddlepoint(model)
            else 'not_applicable'
        ),
        'saddlepoint_derivative_backend': (
            getattr(model, 'saddlepoint_derivative_backend', '')
            if _uses_matrix_saddlepoint(model)
            else 'not_applicable'
        ),
        'saddlepoint_finite_difference_step': (
            getattr(model, 'saddlepoint_finite_difference_step', '')
            if _uses_matrix_saddlepoint(model)
            else ''
        ),
        'isotropic_natural_scale': diagnostics['isotropic_natural_scale'],
        'isotropic_wrapped_variance': diagnostics['isotropic_wrapped_variance'],
        'initial_natural_parameter_norm_median': diagnostics.get(
            'initial_natural_parameter_norm_median', ''
        ),
        'initial_natural_parameter_norm_max': diagnostics.get(
            'initial_natural_parameter_norm_max', ''
        ),
        'initial_quadratic_natural_norm_median': diagnostics.get(
            'initial_quadratic_natural_norm_median', ''
        ),
        'initial_quadratic_raw_norm_median': diagnostics.get(
            'initial_quadratic_raw_norm_median', ''
        ),
        'initial_quadratic_positive_eigenvalue_condition_median': diagnostics.get(
            'initial_quadratic_positive_eigenvalue_condition_median', ''
        ),
        'initial_matrix_linear_natural_norm_median': diagnostics.get(
            'initial_matrix_linear_natural_norm_median', ''
        ),
        'initial_matrix_linear_singular_condition_median': diagnostics.get(
            'initial_matrix_linear_singular_condition_median', ''
        ),
        'initial_kappa_natural_norm_median': diagnostics.get(
            'initial_kappa_natural_norm_median', ''
        ),
        'initial_kappa_raw_norm_median': diagnostics.get(
            'initial_kappa_raw_norm_median', ''
        ),
    }
    statistics.update(gradient_qc)
    statistics.update(_prefixed(objective_benchmark, 'fixed_objective_value_gradient'))
    statistics.update(_prefixed(normalizer_benchmark, 'normalizer_value_gradient'))
    if cold_normalizer_benchmark:
        statistics.update(
            _prefixed(cold_normalizer_benchmark, 'cold_normalizer_value_gradient')
        )
    saddle_diagnostics = (
        'last_saddlepoint_newton_iterations',
        'last_saddlepoint_backtracks',
        'last_saddlepoint_residuals',
        'last_saddlepoint_finite_difference_steps',
    )
    for name in saddle_diagnostics:
        if not hasattr(model, name):
            continue
        raw_values = getattr(model, name)
        if torch.is_tensor(raw_values):
            values = raw_values.detach().cpu().reshape(-1).tolist()
        elif isinstance(raw_values, (tuple, list)):
            values = list(raw_values)
        else:
            values = [raw_values]
        numeric = np.asarray(values, dtype=float)
        statistics[name] = json.dumps(values)
        if numeric.size:
            statistics[f'{name}_max'] = float(np.max(np.abs(numeric)))
            statistics[f'{name}_min'] = float(np.min(numeric))
    if hasattr(model, '_last_kummer_terms'):
        terms = getattr(model, '_last_kummer_terms')
        statistics['last_kummer_series_terms'] = max(terms) if terms else 0
    return statistics, _cpu_parameters(params)


def _selected_rows(rows: Iterable[dict[str, str]], args: argparse.Namespace) -> Iterable[tuple[int, dict[str, str]]]:
    for index, row in enumerate(rows):
        if row['distribution'] not in args.distributions:
            continue
        if args.dimensions and _row_int(row, 'ambient_dim') not in args.dimensions:
            continue
        if args.ranks and row.get('rank_label') not in args.ranks:
            continue
        if args.concentrations and row.get('concentration_label') not in args.concentrations:
            continue
        yield index, row


def _scheduled_attempts(
    selected: list[tuple[int, dict[str, str]]],
    args: argparse.Namespace,
) -> list[tuple[int, dict[str, str], int]]:
    """Construct a reproducible schedule without changing statistical cells."""
    attempts = [
        (manifest_index, row, fit_repeat)
        for manifest_index, row in selected
        for fit_repeat in range(args.fit_repeats)
    ]
    if args.execution_order == 'manifest':
        return attempts
    if args.execution_order == 'randomized':
        rng = random.Random(args.seed + 74_921)
        rng.shuffle(attempts)
        return attempts

    by_dimension: dict[int, list[tuple[int, dict[str, str]]]] = {}
    for manifest_index, row in selected:
        by_dimension.setdefault(_row_int(row, 'ambient_dim'), []).append(
            (manifest_index, row)
        )
    scheduled = []
    for dimension in sorted(by_dimension):
        cells = sorted(by_dimension[dimension], key=lambda value: value[0])
        for fit_repeat in range(args.fit_repeats):
            # Alternating the within-p order makes every model occur early and
            # late in a dimension block while retaining ascending p by default.
            ordered = cells if fit_repeat % 2 == 0 else list(reversed(cells))
            scheduled.extend(
                (manifest_index, row, fit_repeat)
                for manifest_index, row in ordered
            )
    return scheduled


def _result_key(result: dict[str, Any]) -> tuple[str, int]:
    return str(result['source_filename']), int(result['fit_repeat'])


def _read_completed(
    path: Path,
    run_id: str,
) -> tuple[list[dict[str, str]], set[tuple[str, int]]]:
    if not path.exists():
        return [], set()
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    mismatched = {row.get('run_id', '') for row in rows if row.get('run_id', '') != run_id}
    if mismatched:
        raise RuntimeError(
            'results.csv contains a different or legacy run_id; use a fresh output directory.'
        )
    completed = {
        _result_key(row) for row in rows if row.get('status') in TERMINAL_STATUSES
    }
    return rows, completed


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_timing_summary(
    path: Path,
    rows: list[dict[str, Any]],
    requested_repeats: int,
) -> None:
    """Summarize every observed attempt without success-selection."""
    latest_by_repeat: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        filename = str(row.get('source_filename', ''))
        if not filename:
            continue
        latest_by_repeat[(filename, int(row.get('fit_repeat', 0)))] = row

    grouped: dict[str, list[dict[str, Any]]] = {}
    for (filename, _), row in latest_by_repeat.items():
        grouped.setdefault(filename, []).append(row)

    summary = []
    identity_fields = (
        'sampler_schema_version',
        'distribution',
        'ambient_dim',
        'matrix_cols',
        'rank_label',
        'active_rank',
        'model_factor_rank',
        'linear_rank',
        'modal_dim',
        'concentration_label',
        'fisher_bingham_anisotropy',
        'n_samples',
        'source_filename',
        'density_kernel',
        'initialization',
        'isotropic_natural_scale',
        'isotropic_wrapped_variance',
        'initialization_seed_scheme',
        'repeat_initialization',
        'convergence_normalization',
        'normalizer_kind',
        'normalizer_benchmark_kind',
        'saddlepoint_order',
        'saddlepoint_derivative_backend',
        'saddlepoint_finite_difference_step',
        'device',
        'dtype',
        'run_id',
        'result_schema_version',
        'target_alignment',
        'observed_alignment',
        'alignment_qc_status',
        'sampling_qc_warning',
        'effective_sample_size',
        'rhat_max',
        'calibration_status',
        'mcmc_qc_status',
        'sampler_qc_status',
    )

    def metric_summary(
        source: list[dict[str, Any]],
        key: str,
        output_key: str | None = None,
    ) -> dict[str, Any]:
        values = []
        for row in source:
            try:
                value = float(row.get(key, ''))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        name = output_key or key
        if values.size == 0:
            return {
                f'{name}_median': '',
                f'{name}_q1': '',
                f'{name}_q3': '',
            }
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        return {
            f'{name}_median': float(median),
            f'{name}_q1': float(q1),
            f'{name}_q3': float(q3),
        }

    for filename, group in grouped.items():
        fitted = [
            row
            for row in group
            if row.get('status') == 'ok' and row.get('fit_seconds') not in {'', None}
        ]
        timed = []
        for row in group:
            try:
                observed = float(row.get('observed_fit_seconds', ''))
            except (TypeError, ValueError):
                continue
            if row.get('status') in {'ok', 'time_limit'} and np.isfinite(observed):
                timed.append(row)
        plateau_rows = [
            row
            for row in fitted
            if row.get('stopping_reason') in {
                'objective_plateau',
                'no_significant_improvement',
            }
            or str(row.get('converged', '')).lower() == 'true'
        ]
        stationary_rows = [
            row
            for row in fitted
            if str(row.get('stationarity_qc_pass', '')).lower() == 'true'
        ]
        reference = fitted[0] if fitted else group[-1]
        record = {field: reference.get(field, '') for field in identity_fields}
        error_count = sum(row.get('status') == 'error' for row in group)
        skipped_count = sum(row.get('status') == 'skipped' for row in group)
        soft_time_limit_count = sum(
            row.get('stopping_reason') == 'max_optimization_seconds'
            for row in fitted
        )
        attempt_time_limit_count = sum(row.get('status') == 'time_limit' for row in group)
        time_limit_count = soft_time_limit_count + attempt_time_limit_count
        max_iteration_count = sum(
            row.get('fit_outcome') == 'max_iterations'
            or row.get('stopping_reason') == 'max_iter'
            for row in fitted
        )
        if group and all(row.get('status') == 'skipped' for row in group):
            cell_status = 'structurally_unavailable'
        elif error_count and not timed:
            cell_status = 'error'
        elif len(group) < requested_repeats:
            cell_status = 'incomplete'
        elif time_limit_count or max_iteration_count or error_count:
            cell_status = 'complete_with_limits'
        elif len(plateau_rows) == requested_repeats:
            cell_status = 'complete'
        else:
            cell_status = 'complete_with_nonplateau'
        record.update(
            {
                'fit_repeats_requested': requested_repeats,
                'fit_repeats_recorded': len(group),
                'timed_fits': len(timed),
                'successful_fits': len(fitted),
                'objective_plateau_fits': len(plateau_rows),
                'converged_fits': len(plateau_rows),
                'stationarity_qc_pass_fits': len(stationary_rows),
                'nonplateau_fits': len(fitted) - len(plateau_rows),
                'time_limit_fits': time_limit_count,
                'attempt_time_limit_fits': attempt_time_limit_count,
                'max_iteration_fits': max_iteration_count,
                'error_fits': error_count,
                'skipped_fits': skipped_count,
                'primary_timing_basis': 'all_observed_attempts_until_stopping',
                'per_update_timing_basis': 'all_fitted_attempts',
                'conditional_completion_timing_basis': 'objective_plateau_fits_only',
                'status': cell_status,
            }
        )
        for metric in (
            'observed_fit_seconds',
            'total_fit_seconds',
            'fit_seconds',
            'initialization_seconds',
            'post_update_evaluation_seconds',
            'iterations',
            'median_seconds_per_iteration',
            'median_iteration_cpu_seconds',
            'median_seconds_per_iteration_per_training_observation',
            'fixed_objective_value_gradient_wall_median_seconds',
            'fixed_objective_value_gradient_cpu_median_seconds',
            'normalizer_value_gradient_wall_median_seconds',
            'normalizer_value_gradient_cpu_median_seconds',
            'cold_normalizer_value_gradient_wall_median_seconds',
            'heldout_log_score_gain',
            'heldout_log_score_gain_per_dof',
            'final_gradient_rms_per_parameter_per_observation',
        ):
            record.update(metric_summary(timed if metric == 'observed_fit_seconds' else fitted, metric))
        for metric in ('total_fit_seconds', 'fit_seconds', 'iterations'):
            record.update(
                metric_summary(
                    plateau_rows,
                    metric,
                    output_key=f'{metric}_objective_plateau_only',
                )
            )
        summary.append(record)

    summary.sort(key=lambda row: str(row['source_filename']))
    _write_results(path, summary)


def _base_result(
    row: dict[str, str],
    manifest_index: int,
    fit_repeat: int,
    split_method: str,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    fit_seed: int,
    split_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    uses_saddlepoint = row['distribution'] in {
        'matrix_fisher',
        'matrix_bingham',
        'matrix_fisher_bingham',
    }
    keep = [
        'sampler_schema_version',
        'distribution',
        'ambient_dim',
        'matrix_cols',
        'rank_label',
        'active_rank',
        'model_factor_rank',
        'linear_rank',
        'modal_dim',
        'modal_frame_dim',
        'quadratic_nullity',
        'concentration_label',
        'concentration_strength',
        'target_alignment',
        'observed_alignment',
        'alignment_mcse',
        'alignment_qc_status',
        'calibration_method',
        'calibration_status',
        'calibration_iterations',
        'calibration_draws',
        'calibration_estimated_alignment',
        'calibration_mcse',
        'calibration_seconds',
        'fisher_bingham_anisotropy',
        'natural_concentration',
        'linear_concentration',
        'quadratic_concentration',
        'quadratic_spectrum',
        'n_samples',
        'iid',
        'sampler',
        'burnin_sweeps',
        'thin_sweeps',
        'n_chains',
        'rhat_max',
        'diagnostic_rhat',
        'effective_sample_size',
        'slice_evaluations',
        'decorrelation_status',
        'mcmc_qc_status',
        'proposal_count',
        'acceptance_rate',
        'sampler_qc_status',
        'density_kernel',
    ]
    result = {key: row.get(key, '') for key in keep}
    result.update(
        {
            'manifest_index': manifest_index,
            'source_filename': row['filename'],
            'fit_repeat': fit_repeat,
            'fit_seed': fit_seed,
            'initialization_seed_scheme': 'paired_by_fit_repeat',
            'split_seed': split_seed,
            'sampling_seconds': row.get('sampling_seconds', row.get('elapsed_seconds', '')),
            'sample_write_seconds': row.get('write_seconds', ''),
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
            'split_method': split_method,
            'components': args.components,
            'model_factor_rank': _model_rank(row),
            'dtype': args.dtype,
            'device': args.resolved_device,
            'learning_rate': args.learning_rate,
            'threads': args.threads,
            'max_iterations': args.max_iterations,
            'max_fit_seconds': args.max_fit_seconds,
            'max_attempt_seconds': args.max_attempt_seconds,
            'tolerance': args.tolerance,
            'initialization_tolerance': args.initialization_tolerance,
            'convergence_window': args.convergence_window,
            'optimizer_restarts': args.optimizer_restarts,
            'initialization': args.initialization,
            'isotropic_natural_scale': args.isotropic_natural_scale,
            'isotropic_wrapped_variance': args.isotropic_wrapped_variance,
            'repeat_initialization': args.repeat_initialization,
            'convergence_normalization': args.convergence_normalization,
            'timing_warmup_iterations': args.timing_warmup_iterations,
            'max_learning_rate_reductions': args.max_learning_rate_reductions,
            'learning_rate_reduction_factor': args.learning_rate_reduction_factor,
            'stationarity_tolerance': args.stationarity_tolerance,
            'benchmark_repetitions': args.benchmark_repetitions,
            'benchmark_warmup_evaluations': args.benchmark_warmup_evaluations,
            'cold_saddle_benchmark_repetitions': args.cold_saddle_benchmark_repetitions,
            'benchmark_max_seconds': args.benchmark_max_seconds,
            'saddlepoint_order': (
                args.saddlepoint_order if uses_saddlepoint else 'not_applicable'
            ),
            'saddlepoint_derivative_backend': (
                args.saddlepoint_derivative_backend
                if uses_saddlepoint
                else 'not_applicable'
            ),
            'saddlepoint_finite_difference_step': (
                args.saddlepoint_finite_difference_step if uses_saddlepoint else ''
            ),
            'saddlepoint_iterations': args.saddlepoint_iterations,
            'saddlepoint_tolerance': args.saddlepoint_tolerance,
            'direct_matrix_linear_parameterization': args.direct_matrix_linear_parameterization,
            'second_order_validation_max_seconds': args.second_order_validation_max_seconds,
            'skip_second_order_validation': args.skip_second_order_validation,
            'integration_points': args.integration_points,
            'wrapped_winding_radius': args.wrapped_winding_radius,
            'show_progress': args.show_progress,
            'run_id': args.run_id,
            'result_schema_version': RESULT_SCHEMA_VERSION,
            'hostname': args.environment_metadata['hostname'],
            'cpu_model': args.environment_metadata['cpu_model'],
            'python_version': args.environment_metadata['python_version'],
            'numpy_version': args.environment_metadata['numpy_version'],
            'scipy_version': args.environment_metadata['scipy_version'],
            'torch_version': args.environment_metadata['torch_version'],
            'torch_num_interop_threads': args.environment_metadata['torch_num_interop_threads'],
            'git_revision': args.environment_metadata['git_revision'],
            'git_dirty': args.environment_metadata['git_dirty'],
            'omp_num_threads': args.environment_metadata['thread_environment']['OMP_NUM_THREADS'],
            'mkl_num_threads': args.environment_metadata['thread_environment']['MKL_NUM_THREADS'],
            'openblas_num_threads': args.environment_metadata['thread_environment']['OPENBLAS_NUM_THREADS'],
            'numexpr_num_threads': args.environment_metadata['thread_environment']['NUMEXPR_NUM_THREADS'],
            'intrinsic_support_dimension': _intrinsic_dimension(row),
            'sampling_qc_warning': any(
                str(row.get(field, '')) == 'warning'
                or str(row.get(field, '')).endswith('_warning')
                for field in (
                    'alignment_qc_status',
                    'calibration_status',
                    'mcmc_qc_status',
                    'sampler_qc_status',
                )
            ),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=Path("overview_paper/directional_samples_protocol_v2/manifest.csv"), help='Schema-v2 sampler manifest.csv')
    parser.add_argument('--output-dir', type=Path, default=Path("overview_paper/computational_results_schema_v4"))
    parser.add_argument('--distributions', default=','.join(sorted(SUPPORTED_DISTRIBUTIONS)))
    parser.add_argument('--dimensions', default='', help='Optional comma-separated ambient dimensions')
    parser.add_argument('--ranks', default='', help='Optional comma-separated rank labels')
    parser.add_argument('--concentrations', default='', help='Optional comma-separated concentration labels')
    parser.add_argument('--components', type=int, default=1)
    parser.add_argument(
        '--fit-repeats',
        type=int,
        default=5,
        help='Independent paired-seed fitting repeats summarized robustly within each cell.',
    )
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=20250308)
    parser.add_argument('--device', default='cpu', help='cpu by default; cuda or a concrete torch device may be requested explicitly')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--max-iterations', type=int, default=20000)
    parser.add_argument('--max-fit-seconds', type=float, default=1000.0,
                        help='Soft per-fit optimization budget checked between updates; capped fits are marked non-converged.')
    parser.add_argument(
        '--max-attempt-seconds',
        type=float,
        default=1200.0,
        help='Best-effort wall-time guard for one complete fit/evaluation attempt.',
    )
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Minimum material log-likelihood gain per training observation.')
    parser.add_argument('--initialization-tolerance', type=float, default=1e-8,
                        help='Tolerance used only by model-specific initializers.')
    parser.add_argument('--convergence-window', type=int, default=25)
    parser.add_argument('--optimizer-restarts', type=int, default=1)
    parser.add_argument(
        '--initialization',
        choices=('default', 'uniform', 'isotropic'),
        default='isotropic',
        help=(
            'Common fixed-norm, near-isotropic initialization used by default. '
            "'default' selects each model's practical data-derived initializer."
        ),
    )
    parser.add_argument(
        '--isotropic-natural-scale',
        type=float,
        default=1e-2,
        help=(
            'Total identifiable Euclidean/Frobenius norm of the common '
            'nonzero perturbation; combined models split it equally across blocks.'
        ),
    )
    parser.add_argument(
        '--isotropic-wrapped-variance',
        type=float,
        default=5.0,
        help=(
            'Residual variance of the diffuse seeded wrapped-normal start; '
            'five suppresses random-mean advantages without making its gradient vanish.'
        ),
    )
    parser.add_argument(
        '--convergence-normalization',
        choices=('none', 'observations', 'observations_and_dimension'),
        default='observations',
        help='Scale only the early-stopping diagnostic; the fitted objective remains summed NLL.',
    )
    parser.add_argument('--timing-warmup-iterations', type=int, default=10)
    parser.add_argument('--decrease-lr-on-plateau', action='store_true')
    parser.add_argument('--max-learning-rate-reductions', type=int, default=2,
                        help='Number of best-state Adam restarts at a smaller learning rate before declaring an objective plateau.')
    parser.add_argument('--learning-rate-reduction-factor', type=float, default=0.1)
    parser.add_argument('--stationarity-tolerance', type=float, default=1e-4,
                        help='QC threshold for gradient RMS per parameter per training observation; it does not alter stopping.')
    parser.add_argument('--benchmark-repetitions', type=int, default=20,
                        help='Maximum fixed-state value-and-gradient evaluations per timing benchmark.')
    parser.add_argument('--benchmark-warmup-evaluations', type=int, default=3)
    parser.add_argument('--cold-saddle-benchmark-repetitions', type=int, default=5)
    parser.add_argument('--benchmark-max-seconds', type=float, default=10.0,
                        help='Per-benchmark soft budget checked between complete evaluations.')
    parser.add_argument('--integration-points', type=int, default=400)
    parser.add_argument(
        '--factorized-matrix-linear-parameterization',
        action='store_false',
        dest='direct_matrix_linear_parameterization',
        help='Use redundant low-rank factors for the matrix linear term instead of direct full-q F.',
    )
    parser.set_defaults(direct_matrix_linear_parameterization=True)
    parser.add_argument('--saddlepoint-iterations', type=int, default=20)
    parser.add_argument('--saddlepoint-tolerance', type=float, default=1e-8)
    parser.add_argument('--saddlepoint-order', type=int, choices=(1, 2), default=2,
                        help='Matrix Fisher/Bingham/Fisher--Bingham saddlepoint order; order two is the primary fit.')
    parser.add_argument(
        '--saddlepoint-derivative-backend',
        choices=('finite_difference', 'autodiff'),
        default='finite_difference',
        help=(
            'Evaluate third/fourth cumulant derivatives for the order-two '
            'matrix saddlepoint by the validated scalable stencil (default) '
            'or full autodiff reference implementation.'
        ),
    )
    parser.add_argument(
        '--saddlepoint-finite-difference-step',
        type=float,
        default=5e-3,
        help='Base step for the scale-aware matrix cumulant derivative stencil.',
    )
    parser.add_argument(
        '--skip-second-order-validation',
        action='store_true',
        help='Skip the post-fit comparison with the alternate matrix saddlepoint order.',
    )
    parser.add_argument(
        '--second-order-validation-max-seconds',
        type=float,
        default=30.0,
        help='Best-effort wall-time guard for the alternate-order held-out re-score.',
    )
    parser.add_argument('--wrapped-winding-radius', type=int, default=1)
    parser.add_argument('--winding-chunk-size', type=int, default=64)
    parser.add_argument('--max-winding-vectors', type=int, default=200000)
    parser.add_argument('--wrapped-mode', choices=('skip', 'central'), default='skip')
    parser.add_argument(
        '--execution-order',
        choices=('dimension_balanced', 'randomized', 'manifest'),
        default='dimension_balanced',
        help=(
            'By default process dimensions in ascending order and alternate '
            'the within-dimension cell order across timing repeats. Manifest '
            'and fully randomized schedules are available as sensitivities.'
        ),
    )
    parser.add_argument(
        '--repeat-initialization',
        choices=('fixed', 'independent'),
        default='independent',
        help=(
            'Use a different paired seed for every fitting repeat by default; '
            'the same repeat seed is shared across comparable cells. Fixed '
            'starts remain available as a timing-only sensitivity.'
        ),
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Resume an exactly matching run, or start fresh when the output '
            'directory is absent or empty.'
        ),
    )
    parser.add_argument('--show-progress', action='store_true')
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_fraction < 1:
        raise ValueError('--test-fraction should be strictly between zero and one.')
    positive = [
        args.components,
        args.fit_repeats,
        args.threads,
        args.max_iterations,
        args.convergence_window,
        args.optimizer_restarts,
        args.integration_points,
        args.saddlepoint_iterations,
        args.benchmark_repetitions,
        args.cold_saddle_benchmark_repetitions,
    ]
    if any(value < 1 for value in positive):
        raise ValueError('Count-valued arguments should be positive.')
    if (
        args.learning_rate <= 0
        or args.isotropic_natural_scale <= 0
        or args.isotropic_wrapped_variance <= 0
        or args.tolerance <= 0
        or args.initialization_tolerance <= 0
        or args.saddlepoint_tolerance <= 0
        or args.saddlepoint_finite_difference_step <= 0
        or args.max_fit_seconds <= 0
        or args.max_attempt_seconds <= 0
        or args.second_order_validation_max_seconds <= 0
        or args.stationarity_tolerance <= 0
        or args.benchmark_max_seconds <= 0
    ):
        raise ValueError('Learning rate and tolerances should be positive.')
    if (
        args.timing_warmup_iterations < 0
        or args.benchmark_warmup_evaluations < 0
        or args.max_learning_rate_reductions < 0
    ):
        raise ValueError('Warm-up and learning-rate reduction counts must be non-negative.')
    if not 0 < args.learning_rate_reduction_factor < 1:
        raise ValueError('--learning-rate-reduction-factor must lie strictly between zero and one.')
    if args.optimizer_restarts != 1:
        raise ValueError(
            'This timing protocol requires --optimizer-restarts 1; use --fit-repeats '
            'for independent timing replications.'
        )
    if args.max_attempt_seconds < args.max_fit_seconds:
        raise ValueError('--max-attempt-seconds must be at least --max-fit-seconds.')
    unknown = args.distributions - SUPPORTED_DISTRIBUTIONS
    if unknown:
        raise ValueError(f'Unsupported --distributions entries: {sorted(unknown)}')


def main() -> int:
    args = parse_args()
    args.distributions = _comma_list(args.distributions)
    args.dimensions = _optional_ints(args.dimensions)
    args.ranks = _comma_list(args.ranks)
    args.concentrations = _comma_list(args.concentrations)
    _validate_args(args)

    manifest = args.manifest.resolve()
    if not manifest.exists():
        raise FileNotFoundError(f'Cannot find sampler manifest: {manifest}')
    output_dir = (args.output_dir or manifest.parent / 'fit_results').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_dir = output_dir / 'parameters'
    results_path = output_dir / 'results.csv'
    timing_summary_path = output_dir / 'timing_summary.csv'
    config_path = output_dir / 'experiment_config.json'

    device = _resolve_device(args.device)
    args.resolved_device = str(device)
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    torch.set_default_dtype(dtype)
    native_thread_limiter = threadpool_limits(limits=args.threads)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work begins.  The
        # actual value is recorded below if an embedding already initialized it.
        pass
    runtime_warmup_seconds = _warm_up_timing_runtime(device, dtype)
    with manifest.open(newline='') as handle:
        manifest_rows = list(csv.DictReader(handle))
    _validate_sampler_dataset(manifest, manifest_rows)
    selected = list(_selected_rows(manifest_rows, args))
    if not selected:
        raise ValueError('No manifest cells matched the requested filters.')
    manifest_hash = _sha256_file(manifest)
    sample_dataset_hash = _sample_dataset_hash(manifest, selected)
    run_id, identity_payload, source_hashes = _run_identity(
        args, manifest_hash, sample_dataset_hash
    )
    args.run_id = run_id
    args.environment_metadata = {
        'hostname': socket.gethostname(),
        'cpu_model': _cpu_model(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'torch_version': str(torch.__version__),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
        'native_threadpools': threadpool_info(),
        'git_revision': _git_value('rev-parse', 'HEAD'),
        'git_dirty': bool(_git_value('status', '--porcelain')),
        'thread_environment': {
            name: os.environ.get(name, '')
            for name in (
                'OMP_NUM_THREADS',
                'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS',
            )
        },
    }

    resume_existing = False
    if args.resume and config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if existing_config.get('run_id') != run_id:
            raise RuntimeError(
                'The requested settings, manifest, or fitting source differ from the existing run. '
                'Use a fresh output directory rather than mixing benchmark configurations.'
            )
        resume_existing = True
    elif args.resume and any(output_dir.iterdir()):
        raise RuntimeError(
            f'--resume found a nonempty output directory without a matching '
            f'experiment_config.json: {output_dir}. Use a fresh output directory.'
        )
    elif any(output_dir.iterdir()):
        raise FileExistsError(
            f'{output_dir} is nonempty. Pass --resume only for an exact configuration '
            'match, or select a new --output-dir.'
        )

    parameter_dir.mkdir(exist_ok=True)

    scheduled_attempts = _scheduled_attempts(selected, args)
    previous_rows, completed = (
        _read_completed(results_path, run_id) if resume_existing else ([], set())
    )
    results: list[dict[str, Any]] = list(previous_rows)

    pending_manifest_indices = {
        manifest_index
        for manifest_index, row, fit_repeat in scheduled_attempts
        if (row['filename'], fit_repeat) not in completed
    }
    loaded: dict[int, tuple[torch.Tensor, torch.Tensor, str, int]] = {}
    for manifest_index, row in selected:
        if manifest_index not in pending_manifest_indices:
            continue
        X = _load_samples(row, manifest.parent, dtype)
        iid = _row_bool(row, 'iid', default=True)
        split_seed = args.seed + manifest_index * 1009
        X_train, X_test, split_method = _split_samples(
            X, args.test_fraction, split_seed, iid, row=row
        )
        loaded[manifest_index] = (X_train, X_test, split_method, split_seed)

    structural_warmups = []
    warmed_structures: set[tuple[Any, ...]] = set()
    for manifest_index, row in selected:
        if manifest_index not in loaded:
            continue
        X_train = loaded[manifest_index][0]
        structure = _structural_warmup_key(row, X_train)
        if structure in warmed_structures:
            continue
        warmed_structures.add(structure)
        _set_seed(args.seed + 501_701 + len(warmed_structures) * 31, device)
        try:
            _require_sampling_qc(row)
            with _best_effort_wall_timeout(
                args.max_attempt_seconds,
                'Unmeasured structural model warm-up',
            ):
                elapsed = _warm_up_model_configuration(
                    row, X_train, args, device
                )
            structural_warmups.append(
                {
                    'structure': list(structure),
                    'status': 'ok',
                    'seconds': elapsed,
                }
            )
        except SkipCell as error:
            structural_warmups.append(
                {
                    'structure': list(structure),
                    'status': 'structurally_unavailable',
                    'seconds': 0.0,
                    'reason': str(error),
                }
            )
        except Exception as error:
            raise RuntimeError(
                f'Unmeasured warm-up failed for structure {structure}: '
                f'{type(error).__name__}: {error}'
            ) from error

    config = {
        'run_id': run_id,
        'result_schema_version': RESULT_SCHEMA_VERSION,
        'manifest': str(manifest),
        'manifest_sha256': manifest_hash,
        'sample_dataset_sha256': sample_dataset_hash,
        'output_dir': str(output_dir),
        'settings': identity_payload['settings'],
        'source_sha256': source_hashes,
        'environment': args.environment_metadata,
        'selected_cells': len(selected),
        'fit_repeats': args.fit_repeats,
        'unmeasured_runtime_warmup_seconds': runtime_warmup_seconds,
        'unmeasured_structural_warmup_count': sum(
            record['status'] == 'ok' for record in structural_warmups
        ),
        'unmeasured_structural_warmup_seconds': sum(
            float(record['seconds']) for record in structural_warmups
        ),
        'unmeasured_structural_warmups': structural_warmups,
    }
    if not resume_existing:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')

    for attempt_number, (manifest_index, row, fit_repeat) in enumerate(
        scheduled_attempts, start=1
    ):
        key = (row['filename'], fit_repeat)
        if key in completed:
            continue
        X_train, X_test, split_method, split_seed = loaded[manifest_index]
        fit_seed = _fit_initialization_seed(
            args.seed,
            fit_repeat,
            args.repeat_initialization,
        )
        _set_seed(fit_seed, device)
        result = _base_result(
            row,
            manifest_index,
            fit_repeat,
            split_method,
            X_train,
            X_test,
            fit_seed,
            split_seed,
            args,
        )
        result['execution_attempt_index'] = attempt_number
        attempt_started = time.perf_counter()
        try:
            _require_sampling_qc(row)
            with _best_effort_wall_timeout(
                args.max_attempt_seconds,
                'Complete fit attempt',
            ):
                statistics, params = _fit_one(row, X_train, X_test, args, device)
            result.update(statistics)
            result.update({'status': 'ok', 'error_type': '', 'error_message': ''})
            stem = Path(row['filename']).stem
            parameter_path = parameter_dir / f'{stem}__fitrep-{fit_repeat:03d}.pt'
            torch.save(params, parameter_path)
            result['parameter_filename'] = str(parameter_path.relative_to(output_dir))
        except SkipCell as error:
            result.update({'status': 'skipped', 'error_type': type(error).__name__, 'error_message': str(error)})
            if row['distribution'] == 'wrapped_normal':
                result['normalizer_kind'] = 'intractable_winding_lattice_skipped'
        except TimeoutError as error:
            result.update(
                {
                    'status': 'time_limit',
                    'fit_outcome': 'attempt_time_limit',
                    'stopping_reason': 'max_attempt_seconds',
                    'converged': False,
                    'attempt_time_limit_seconds': args.max_attempt_seconds,
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                }
            )
        except Exception as error:
            result.update({'status': 'error', 'error_type': type(error).__name__, 'error_message': str(error)[:1000]})
        result['attempt_seconds'] = time.perf_counter() - attempt_started
        if result['status'] == 'ok':
            result['observed_fit_seconds'] = result.get(
                'total_fit_seconds', result['attempt_seconds']
            )
        elif result['status'] == 'time_limit':
            result['observed_fit_seconds'] = result['attempt_seconds']
        results.append(result)
        _write_results(results_path, results)
        _write_timing_summary(timing_summary_path, results, args.fit_repeats)
        print(
            f'[{attempt_number}/{len(scheduled_attempts)}] {row["distribution"]} '
            f'p={row["ambient_dim"]} rank={row["rank_label"]} '
            f'concentration={row["concentration_label"]} repeat={fit_repeat + 1}: '
            f'{result["status"]} ({result.get("stopping_reason", "not fitted")})',
            flush=True,
        )

    print(f'Wrote {len(results)} fit records to {results_path}')
    print(f'Wrote robust timing summaries to {timing_summary_path}')
    latest = {_result_key(row): row for row in results}
    status_counts: dict[str, int] = {}
    for row in latest.values():
        status = str(row.get('status', 'missing'))
        status_counts[status] = status_counts.get(status, 0) + 1
    expected_records = len(selected) * args.fit_repeats
    print(f'Final repeat-status counts: {status_counts}; expected={expected_records}')
    if len(latest) != expected_records or status_counts.get('error', 0):
        print(
            'One or more required fit attempts failed or are missing; results were preserved '
            'for an exact --resume retry.',
            file=sys.stderr,
        )
        native_thread_limiter.unregister()
        return 1
    native_thread_limiter.unregister()
    return 0


if __name__ == '__main__':
    sys.exit(main())
