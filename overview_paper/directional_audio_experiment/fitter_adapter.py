"""Adapter between the directional-audio experiment and the PyTorch PCMM API.

All public methods receive raw complex observations with shape ``(n, p)``.
The fitted object owns the family-specific representation so that training,
posterior prediction, and samplewise likelihood always use the same transform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Protocol

import numpy as np
import torch

from PCMM.helper_functions import test_model, train_model


class FrozenModel(Protocol):
    def posterior(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities (K,n), preserving fitted component order."""
        ...

    def score_samples(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized mixture log densities (n,), INCLUDING weights.

        Needed for held-out likelihood selection of K. Posteriors alone do
        not determine data likelihood. Include all K-dependent normalizers.
        """
        ...

    def export_params(self) -> dict[str, np.ndarray]:
        """Return detached fitted parameters suitable for ``numpy.savez``."""
        ...

    def seat_scores(self, steering_f: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return fitted component-by-seat directional probabilities (K,S)."""
        ...


def unit_complex(x: torch.Tensor) -> torch.Tensor:
    """Complex projective representative (n,p); relative magnitudes retained."""
    return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)


# The experiment supplies independent restarts, so each adapter fit performs a
# single optimization replication.  All models use the PyTorch path (LR > 0).
# Rank-(p-1) factors give Bingham and ACG full matrix expressiveness while
# leaving a residual direction for PCMM's data-driven initializer.
_COMMON_OPTIONS = {
    "LR": 0.1,
    "HMM": False,
    "tol": 1e-5,
    "max_iter": 2_000,
    "num_repl": 1,
    "threads": 8,
    "decrease_lr_on_plateau": False,
    "num_comparison": 25,
}


def _representation(x: torch.Tensor, model_name: str) -> torch.Tensor:
    """Construct the observation space used by one configured family."""
    if x.ndim != 2 or not torch.is_complex(x):
        raise ValueError("Expected raw complex observations with shape (n, p).")
    if not torch.isfinite(x).all():
        raise ValueError("Observations must be finite.")

    if model_name in {"complex_watson", "complex_bingham", "complex_acg"}:
        return unit_complex(x)
    if model_name == "uniform_vmvm":
        # The oscillatory VMVM density itself quotients out a common phase.
        # It therefore receives all p raw microphone phases, rather than an
        # arbitrarily reference-channel-anchored (p-1)-vector.
        return torch.angle(x)
    if model_name == "complex_gaussian":
        return x
    raise ValueError(f"Unknown model family: {model_name!r}.")


def _pcmm_options(model_name: str, dimension: int, n_components: int) -> dict:
    """Map experiment family names to existing PCMM model options."""
    options = dict(_COMMON_OPTIONS)
    factor_rank = max(1, dimension - 1)
    if model_name == "complex_watson":
        options.update(
            modelname="Complex_Watson",
            # PCMM has a deterministic spectral initializer specifically for
            # the one-component Watson model.
            init=None if n_components == 1 else "dc",
        )
    elif model_name == "complex_bingham":
        options.update(modelname="Complex_Bingham", rank=factor_rank, init="dc")
    elif model_name == "complex_acg":
        options.update(modelname="Complex_ACG", rank=factor_rank, init="dc")
    elif model_name == "uniform_vmvm":
        options.update(
            modelname="VMVM",
            init="dc",
            # This flag is part of the statistical family, not merely a fit
            # option. PCMM.helper_functions must forward it both when fitting
            # and when reconstructing a frozen model for held-out scoring.
            oscillatory_data=True,
        )
    elif model_name == "complex_gaussian":
        # PCMM's Complex_Normal is a proper, zero-mean complex Gaussian with
        # one fixed rank-one-plus-isotropic covariance per mixture component.
        # Rank one matches a narrowband point-source image.  The isotropic
        # initializer is robust when a frequency bin is itself nearly rank
        # one. It does not add a time-varying source-power parameter.
        options.update(modelname="Complex_Normal", rank=1, init="isotropic")
    else:
        raise ValueError(f"Unknown model family: {model_name!r}.")
    return options


@dataclass
class _FrozenPCMMModel:
    """Fitted PCMM parameters plus the raw-observation prediction interface."""

    params: dict
    n_components: int
    model_name: str
    options: dict
    data_scale: float = 1.0
    _cached_input: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_posterior: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_scores: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _evaluate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The runner requests posterior and likelihood consecutively for the
        # same tensor.  Cache that pair because test_model otherwise rebuilds
        # and evaluates the PCMM model twice (notably costly for Bingham).
        if x is self._cached_input:
            assert self._cached_posterior is not None
            assert self._cached_scores is not None
            return self._cached_posterior, self._cached_scores

        data = _representation(x, self.model_name) / self.data_scale
        _, posterior, scores = test_model(
            data_test=data,
            params=self.params,
            K=self.n_components,
            options=self.options,
        )
        real_dtype = x.real.dtype
        posterior = torch.as_tensor(posterior, dtype=real_dtype, device=x.device)
        scores = torch.as_tensor(scores, dtype=real_dtype, device=x.device)
        if self.model_name == "complex_gaussian":
            # If y=x/s in complex dimension p, f_x(x)=f_y(y)/s**(2p).
            scores = scores - 2 * x.shape[1] * math.log(self.data_scale)
        expected_posterior = (self.n_components, x.shape[0])
        if posterior.shape != expected_posterior:
            raise RuntimeError(
                f"PCMM returned posterior shape {tuple(posterior.shape)}; "
                f"expected {expected_posterior}."
            )
        if scores.shape != (x.shape[0],):
            raise RuntimeError(
                f"PCMM returned samplewise likelihood shape {tuple(scores.shape)}; "
                f"expected {(x.shape[0],)}."
            )

        self._cached_input = x
        self._cached_posterior = posterior.detach()
        self._cached_scores = scores.detach()
        return self._cached_posterior, self._cached_scores

    def posterior(self, x: torch.Tensor) -> torch.Tensor:
        return self._evaluate(x)[0]

    def score_samples(self, x: torch.Tensor) -> torch.Tensor:
        # test_model delegates to PCMM.test_log_likelihood, whose samplewise
        # values are logsumexp(log component density + log mixture weight).
        return self._evaluate(x)[1]

    def export_params(self) -> dict[str, np.ndarray]:
        """Export a self-contained, non-aliasing NumPy parameter snapshot.

        The model-family key and frequency live in the runner's result record;
        this method deliberately returns arrays only so its output can be
        written directly with ``numpy.savez``. ``data_scale`` is included
        because Gaussian fitting uses an exact scalar change of coordinates.
        """
        exported: dict[str, np.ndarray] = {}
        for name, value in self.params.items():
            if torch.is_tensor(value):
                array = value.detach().cpu().numpy()
            else:
                array = np.asarray(value)
            exported[name] = np.array(array, copy=True)
        exported["data_scale"] = np.asarray(self.data_scale, dtype=np.float64)
        if self.model_name == "uniform_vmvm":
            exported["oscillatory_data"] = np.asarray(True)
        return exported

    def seat_scores(self, steering_f: np.ndarray | torch.Tensor) -> np.ndarray:
        """Evaluate each learned component over measured candidate seats.

        Parameters
        ----------
        steering_f:
            Complex narrowband steering vectors with shape ``(S,p)``. These
            should be the measured seat impulse-response signatures at the
            frequency represented by this fitted model.

        Returns
        -------
        ndarray, (K,S)
            Component-conditional scores normalized across the supplied seats.
            Mixture weights are intentionally excluded. Watson, Bingham, ACG,
            and VMVM use their fitted directional densities. For the
            zero-mean complex Gaussian, this uses the complex ACG angular law
            induced by its fitted covariance, eliminating arbitrary IR level.

        Notes
        -----
        These are parameter-derived spatial readouts at measured support
        points, not Euclidean location means or covariance ellipses.
        """
        log_scores = self._seat_log_scores(steering_f)
        probabilities = torch.softmax(log_scores, dim=1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Fitted component seat scores are non-finite.")
        return probabilities.detach().cpu().numpy()

    def _seat_log_scores(
        self, steering_f: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Return unweighted component log-density kernels with shape (K,S)."""
        reference = next(iter(self.params.values()))
        if torch.is_tensor(reference):
            device = reference.device
            real_dtype = reference.real.dtype
        else:
            reference_array = np.asarray(reference)
            device = torch.device("cpu")
            real_dtype = (
                torch.float64
                if reference_array.dtype
                in {np.dtype("float64"), np.dtype("complex128")}
                else torch.float32
            )
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )
        steering = torch.as_tensor(steering_f, device=device)
        if steering.ndim != 2 or steering.shape[1] < 1:
            raise ValueError("steering_f must have shape (n_seats, n_microphones).")
        if steering.shape[1] != self._model_dimension:
            raise ValueError(
                f"steering_f has {steering.shape[1]} microphones; fitted model "
                f"expects {self._model_dimension}."
            )
        if not torch.is_complex(steering):
            raise ValueError("steering_f must contain complex steering vectors.")
        steering = steering.to(dtype=complex_dtype)
        if not torch.isfinite(steering).all():
            raise ValueError("steering_f must be finite.")
        norms = torch.linalg.vector_norm(steering, dim=1)
        if torch.any(norms <= torch.finfo(real_dtype).tiny):
            raise ValueError("Every candidate seat must have a nonzero steering vector.")
        unit = steering / norms[:, None]

        def parameter(name: str, *, complex_value: bool = False) -> torch.Tensor:
            if name not in self.params:
                raise RuntimeError(
                    f"Fitted {self.model_name} parameters do not contain {name!r}."
                )
            dtype = complex_dtype if complex_value else real_dtype
            return torch.as_tensor(self.params[name], dtype=dtype, device=device)

        if self.model_name == "complex_watson":
            mu = torch.nn.functional.normalize(
                parameter("mu", complex_value=True), dim=1,
            )
            kappa = parameter("kappa")
            projection = torch.abs(unit @ mu.mH).square().T
            return kappa[:, None] * projection

        if self.model_name == "complex_bingham":
            factors = parameter("M", complex_value=True)
            concentration = factors @ factors.mH
            # Any identity shift used to identify the Bingham concentration
            # adds the same constant for every unit steering vector and thus
            # cancels in the across-seat normalization below.
            return torch.einsum(
                "sp,kpq,sq->ks", unit.conj(), concentration, unit,
            ).real

        if self.model_name in {"complex_acg", "complex_gaussian"}:
            factors = parameter("M", complex_value=True)
            if self.model_name == "complex_acg":
                diagonal = torch.ones(
                    self.n_components, dtype=real_dtype, device=device,
                )
            else:
                diagonal = parameter("gamma")
            identity = torch.eye(
                self._model_dimension, dtype=complex_dtype, device=device,
            )
            covariance = factors @ factors.mH + diagonal[:, None, None] * identity
            scores = torch.empty(
                (self.n_components, unit.shape[0]),
                dtype=real_dtype,
                device=device,
            )
            for component in range(self.n_components):
                solved = torch.linalg.solve(covariance[component], unit.T).T
                quadratic = torch.sum(unit.conj() * solved, dim=1).real
                quadratic = quadratic.clamp_min(torch.finfo(real_dtype).tiny)
                _, log_abs_determinant = torch.linalg.slogdet(covariance[component])
                scores[component] = (
                    -log_abs_determinant
                    - self._model_dimension * torch.log(quadratic)
                )
            return scores

        if self.model_name == "uniform_vmvm":
            mu = parameter("mu")
            binding = parameter("lambda")
            phases = torch.angle(unit).to(real_dtype)
            coefficients = binding.to(complex_dtype) * torch.exp(
                -1j * mu.to(complex_dtype)
            )
            observations = torch.exp(1j * phases.to(complex_dtype))
            resultant = torch.abs(observations @ coefficients.T).T

            def log_i0(value: torch.Tensor) -> torch.Tensor:
                return torch.log(torch.special.i0e(value)) + torch.abs(value)

            # The uniform-marginal base measure is common to components and
            # seats. The remaining expression is the exact oscillatory VMVM
            # component log density up to that common constant.
            return log_i0(resultant) - log_i0(binding).sum(dim=1, keepdim=True)

        raise ValueError(f"Unknown model family: {self.model_name!r}.")

    @property
    def _model_dimension(self) -> int:
        if self.model_name in {"complex_watson", "uniform_vmvm"}:
            return int(self.params["mu"].shape[-1])
        return int(self.params["M"].shape[-2])


def fit_model(*, x_train: torch.Tensor, model_name: str, n_components: int,
              frequency_hz: float, seed: int, fit_tolerance: float = 1e-5,
              fit_max_iterations: int = 2_000, fit_patience: int = 25,
              fit_learning_rate: float = .1, fit_threads: int = 8) -> FrozenModel:
    """Fit one configured family with the existing PyTorch PCMM optimizer.

    Parameters
    ----------
    x_train : complex64 torch.Tensor, (n,p), on the configured device
    model_name : complex_watson / complex_bingham / complex_acg /
                 uniform_vmvm / complex_gaussian
    n_components : candidate K, NOT the known number of physical sources
    frequency_hz : current STFT frequency
    seed : deterministic initialization seed

    Return a frozen object exposing posterior(x)->(K,n) and
    score_samples(x)->(n,). The runner handles restarts, train/test prediction,
    alignment and evaluation. Never refit on test observations.
    No source signals, source labels or impulse responses are passed here.
    """
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    if x_train.shape[0] < n_components:
        raise ValueError("n_components cannot exceed the number of observations.")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive.")
    if not math.isfinite(fit_tolerance) or fit_tolerance <= 0:
        raise ValueError("fit_tolerance must be finite and positive.")
    if not math.isfinite(fit_learning_rate) or fit_learning_rate <= 0:
        raise ValueError("fit_learning_rate must be finite and positive.")
    if any(not isinstance(value, int) or value < 1 for value in
           [fit_max_iterations, fit_patience, fit_threads]):
        raise ValueError("Fit iteration, patience, and thread counts must be positive integers.")

    # fit_model is deterministic even when invoked outside experiment.py.
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if x_train.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    data_train = _representation(x_train, model_name)
    data_scale = 1.0
    if model_name == "complex_gaussian":
        # A scalar change of coordinates keeps the same covariance family but
        # avoids initializing gamma=1 many orders of magnitude above STFT data.
        # The frozen scorer applies the exact Jacobian correction.
        data_scale = float(
            torch.sqrt(torch.mean(torch.abs(data_train).square())).detach().cpu()
        )
        if not math.isfinite(data_scale) or data_scale <= 0:
            raise ValueError("Cannot scale degenerate complex Gaussian observations.")
        data_train = data_train / data_scale
    options = _pcmm_options(model_name, data_train.shape[1], n_components)
    options.update(
        LR=fit_learning_rate,
        tol=fit_tolerance,
        max_iter=fit_max_iterations,
        num_comparison=fit_patience,
        threads=fit_threads,
    )
    params, _, _ = train_model(
        data_train=data_train,
        K=n_components,
        options=options,
        suppress_output=True,
    )
    return _FrozenPCMMModel(
        params=params,
        n_components=n_components,
        model_name=model_name,
        options=options,
        data_scale=data_scale,
    )


def fit_partitioned_model(
    *, x_train: torch.Tensor, assignments: np.ndarray | torch.Tensor,
    mixture_weights: np.ndarray | torch.Tensor, model_name: str,
    n_components: int, frequency_hz: float, seed: int,
    fit_tolerance: float = 1e-5, fit_max_iterations: int = 2_000,
    fit_patience: int = 25, fit_learning_rate: float = .1,
    fit_threads: int = 8,
) -> FrozenModel:
    """Fit K component densities to one shared, externally supplied partition.

    Each component is optimized as a one-component PCMM on the frames assigned
    to it.  The K fitted parameter rows are then combined with the supplied
    cross-frequency mixture weights.  Assignments and weights must have been
    obtained from training data; this function never sees held-out samples.
    """
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive.")
    if not math.isfinite(fit_tolerance) or fit_tolerance <= 0:
        raise ValueError("fit_tolerance must be finite and positive.")
    if not math.isfinite(fit_learning_rate) or fit_learning_rate <= 0:
        raise ValueError("fit_learning_rate must be finite and positive.")
    if any(not isinstance(value, int) or value < 1 for value in
           [fit_max_iterations, fit_patience, fit_threads]):
        raise ValueError("Fit iteration, patience, and thread counts must be positive integers.")

    labels = torch.as_tensor(assignments, dtype=torch.long, device=x_train.device)
    if labels.shape != (x_train.shape[0],):
        raise ValueError("assignments must have one integer label per training frame.")
    if torch.any(labels < 0) or torch.any(labels >= n_components):
        raise ValueError("assignments contain a label outside 0..K-1.")
    counts = torch.bincount(labels, minlength=n_components)
    if torch.any(counts == 0):
        raise ValueError("Every component needs at least one assigned training frame.")

    weights = torch.as_tensor(
        mixture_weights, dtype=x_train.real.dtype, device=x_train.device,
    )
    if weights.shape != (n_components,) or not torch.isfinite(weights).all():
        raise ValueError("mixture_weights must be a finite length-K vector.")
    if torch.any(weights <= 0):
        raise ValueError("Every shared mixture weight must be strictly positive.")
    weights = weights / weights.sum()

    data_train = _representation(x_train, model_name)
    data_scale = 1.0
    if model_name == "complex_gaussian":
        # Use one scale for all subsets, so their fitted component densities
        # retain a common coordinate system when recombined.
        data_scale = float(
            torch.sqrt(torch.mean(torch.abs(data_train).square())).detach().cpu()
        )
        if not math.isfinite(data_scale) or data_scale <= 0:
            raise ValueError("Cannot scale degenerate complex Gaussian observations.")
        data_train = data_train / data_scale

    single_options = _pcmm_options(model_name, data_train.shape[1], 1)
    single_options.update(
        LR=fit_learning_rate,
        tol=fit_tolerance,
        max_iter=fit_max_iterations,
        num_comparison=fit_patience,
        threads=fit_threads,
    )
    component_params: list[dict] = []
    for component in range(n_components):
        component_seed = int(np.random.SeedSequence(
            [seed, component]
        ).generate_state(1)[0])
        random.seed(component_seed)
        np.random.seed(component_seed % (2**32 - 1))
        torch.manual_seed(component_seed)
        if x_train.device.type == "cuda":
            torch.cuda.manual_seed_all(component_seed)
        params, _, _ = train_model(
            data_train=data_train[labels == component],
            K=1,
            options=single_options,
            suppress_output=True,
        )
        component_params.append(params)

    parameter_keys = set(component_params[0]) - {"pi", "T"}
    if any((set(params) - {"pi", "T"}) != parameter_keys
           for params in component_params[1:]):
        raise RuntimeError("One-component fits returned inconsistent parameter keys.")
    merged: dict = {}
    for name in sorted(parameter_keys):
        rows = [torch.as_tensor(params[name], device=x_train.device)
                for params in component_params]
        if any(row.ndim < 1 or row.shape[0] != 1 for row in rows):
            raise RuntimeError(
                f"Cannot combine fitted parameter {name!r}: expected leading K=1 axis."
            )
        merged[name] = torch.cat(rows, dim=0)
    merged["pi"] = weights.detach().clone()

    combined_options = _pcmm_options(model_name, data_train.shape[1], n_components)
    combined_options.update(
        LR=fit_learning_rate,
        tol=fit_tolerance,
        max_iter=fit_max_iterations,
        num_comparison=fit_patience,
        threads=fit_threads,
    )
    return _FrozenPCMMModel(
        params=merged,
        n_components=n_components,
        model_name=model_name,
        options=combined_options,
        data_scale=data_scale,
    )
