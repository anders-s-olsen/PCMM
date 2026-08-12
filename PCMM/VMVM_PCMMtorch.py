"""PyTorch implementation of the von-Mises/von-Mises CBMD (VMVM).

The public constructor deliberately mirrors the PCMM PyTorch models::

    model = VMVM(K=K, p=p, params=params, HMM=HMM,
                 samples_per_sequence=samples_per_sequence)

Input observations are angles in radians with shape ``(n, p)``.  The
component log-densities returned by :meth:`log_pdf` have shape ``(K, n)``.

Mixture weights, HMM likelihoods/posteriors, and sequence handling are provided
by :class:`PCMMtorchBaseModel`; this module implements the VMVM-specific density,
initialization, parameter conversion, and sampling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from PCMM.PCMMtorchBaseModel import PCMMtorchBaseModel


_TWO_PI = 2.0 * math.pi


def _log_i0(x: Tensor) -> Tensor:
    """Stable ``log(I0(x))`` using the exponentially scaled Bessel function."""
    return torch.log(torch.special.i0e(x)) + torch.abs(x)


def _inverse_softplus(y: Tensor) -> Tensor:
    """Stable inverse of softplus for strictly positive ``y``."""
    y = torch.clamp(y, min=torch.finfo(y.dtype).tiny)
    return y + torch.log(-torch.expm1(-y))


def _wrap_pi(x: Tensor) -> Tensor:
    """Map angles to [-pi, pi)."""
    return torch.remainder(x + math.pi, _TWO_PI) - math.pi


def _as_tensor(
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


class VMVM(PCMMtorchBaseModel):
    """Mixture/HMM of multivariate von-Mises/von-Mises CBMD components.

    Parameters
    ----------
    K:
        Number of mixture components or HMM states.
    p:
        Number of angular variables.
    params:
        Optional parameter dictionary. Recognized keys are ``mu``/``means``,
        ``kappa``/``marginal_kappa``, ``lambda``/``binding_kappa``,
        ``q``/``qs``, ``weights``/``alpha``/``pi``, and, for an HMM,
        ``initial``/``pi0`` and ``transition``/``T``.
    HMM:
        Add HMM initial and transition probabilities.
    samples_per_sequence:
        Sequence lengths for HMM likelihood/posterior calculations. An integer
        is repeated as needed; a sequence gives explicit lengths. ``None`` or
        zero treats all observations as one sequence.
    qs:
        Optional fixed signs of shape ``(p,)`` or ``(K,p)``. Every entry must
        be -1 or +1. The first coordinate is canonicalized to +1 because a
        simultaneous sign flip leaves the component unchanged.
    cdf_grid_size:
        Number of intervals used by the differentiable von Mises CDF table.
        2048 is generally accurate for fitting; increase for very concentrated
        marginals.
    min_concentration:
        Positive floor used for both concentration families.
    oscillatory_data:
        Model oscillatory observations with exact circular-uniform marginals
        and positive dependence signs. In this mode ``mu`` represents
        dependence-phase offsets and every entry of ``q`` is fixed to +1.
    dtype, device:
        Parameter dtype and device. If omitted they follow PyTorch defaults.

    Notes
    -----
    ``forward(x)`` returns the total mixture/HMM log-likelihood.  Component
    log-densities are available from ``log_pdf(x)`` and have shape ``(K,n)``.
    """

    def __init__(
        self,
        K: int,
        p: int,
        params: Optional[Mapping[str, Any]] = None,
        HMM: bool = False,
        samples_per_sequence: Optional[int | Sequence[int]] = None,
        *,
        qs: Optional[Any] = None,
        cdf_grid_size: int = 2048,
        min_concentration: float = 1e-5,
        oscillatory_data: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        if not isinstance(K, int) or K < 1:
            raise ValueError("K must be a positive integer")
        if not isinstance(p, int) or p < 1:
            raise ValueError("p must be a positive integer")
        if not isinstance(cdf_grid_size, int) or cdf_grid_size < 64:
            raise ValueError("cdf_grid_size must be an integer >= 64")
        if min_concentration <= 0:
            raise ValueError("min_concentration must be positive")

        self.K = K
        self.p = p
        self.distribution = "VMVM"
        self.HMM = bool(HMM)
        if samples_per_sequence is None:
            samples_per_sequence = 0
        self.samples_per_sequence = torch.as_tensor(samples_per_sequence)
        self.cdf_grid_size = cdf_grid_size
        self.min_concentration = float(min_concentration)
        self.oscillatory_data = bool(oscillatory_data)

        dtype = dtype or torch.get_default_dtype()
        device = torch.device(device) if device is not None else torch.device("cpu")

        params = dict(params or {})

        mu0 = self._read_parameter(
            params, ("mu", "means", "mean"), default=None,
            dtype=dtype, device=device,
        )
        if mu0 is None:
            mu0 = torch.empty(K, p, dtype=dtype, device=device).uniform_(-math.pi, math.pi)
        mu0 = self._expand_kp(mu0, "mu")

        marginal0 = self._read_parameter(
            params, ("kappa", "marginal_kappa", "marginal_kappas"), default=1.0,
            dtype=dtype, device=device,
        )
        marginal0 = self._expand_kp(marginal0, "kappa").clamp_min(self.min_concentration)

        binding0 = self._read_parameter(
            params, ("lambda", "binding_kappa", "binding_kappas", "circula_kappa"),
            default=0.5, dtype=dtype, device=device,
        )
        binding0 = self._expand_kp(binding0, "binding_kappa").clamp_min(self.min_concentration)

        if self.oscillatory_data:
            q0 = torch.ones(K, p, dtype=dtype, device=device)
        else:
            q0 = qs
            if q0 is None:
                q0 = self._read_parameter(
                    params, ("q", "qs"), default=torch.ones(K, p),
                    dtype=dtype, device=device,
                )
            else:
                q0 = _as_tensor(q0, dtype=dtype, device=device)
            q0 = self._expand_kp(q0, "q")
            if not torch.all((q0 == 1) | (q0 == -1)):
                raise ValueError("Every q entry must equal -1 or +1")
            # q and -q encode the same component. Canonicalize q[:, 0] = +1.
            q0 = q0 * q0[:, :1]

        self.mu = nn.Parameter(_wrap_pi(mu0))
        self.raw_marginal_kappa = nn.Parameter(
            _inverse_softplus(marginal0 - self.min_concentration + torch.finfo(dtype).eps),
            requires_grad=not self.oscillatory_data,
        )
        self.raw_binding_kappa = nn.Parameter(
            _inverse_softplus(binding0 - self.min_concentration + torch.finfo(dtype).eps)
        )
        self.register_buffer("q", q0)

        weights0 = self._read_parameter(
            params, ("weights", "alpha", "pi", "mixing_weights"), default=torch.ones(K) / K,
            dtype=dtype, device=device,
        )
        weights0 = self._expand_vector(weights0, K, "weights").clamp_min(torch.finfo(dtype).tiny)
        weights0 = weights0 / weights0.sum()

        if self.HMM:
            initial0 = self._read_parameter(
                params, ("initial", "pi0", "initial_probs"), default=weights0,
                dtype=dtype, device=device,
            )
            initial0 = self._expand_vector(initial0, K, "initial").clamp_min(torch.finfo(dtype).tiny)
            initial0 = initial0 / initial0.sum()

            transition0 = self._read_parameter(
                params, ("transition", "T", "transition_matrix"),
                default=torch.eye(K, dtype=dtype, device=device) * 0.9 + 0.1 / K,
                dtype=dtype, device=device,
            )
            transition0 = _as_tensor(transition0, dtype=dtype, device=device)
            if transition0.shape != (K, K):
                raise ValueError(f"transition must have shape {(K, K)}, got {tuple(transition0.shape)}")
            transition0 = transition0.clamp_min(torch.finfo(dtype).tiny)
            transition0 = transition0 / transition0.sum(dim=1, keepdim=True)

            # The base model uses ``pi`` as mixture weights or HMM initial
            # logits, and ``T`` as transition logits.
            self.pi = nn.Parameter(torch.log(initial0))
            self.T = nn.Parameter(torch.log(transition0))
        else:
            self.pi = nn.Parameter(torch.log(weights0))

    # ------------------------------------------------------------------
    # Constrained parameters
    # ------------------------------------------------------------------
    @property
    def marginal_kappa(self) -> Tensor:
        if self.oscillatory_data:
            return torch.zeros_like(self.raw_marginal_kappa)
        return F.softplus(self.raw_marginal_kappa) + self.min_concentration

    @property
    def kappa(self) -> Tensor:
        """Alias for the marginal concentrations."""
        return self.marginal_kappa

    @property
    def binding_kappa(self) -> Tensor:
        return F.softplus(self.raw_binding_kappa) + self.min_concentration

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.pi, dim=0)

    @property
    def initial(self) -> Tensor:
        return torch.softmax(self.pi, dim=0)

    @property
    def transition(self) -> Tensor:
        if not self.HMM:
            raise AttributeError("transition is only defined when HMM=True")
        return torch.softmax(self.T, dim=1)

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------
    def log_pdf(self, x: Tensor, recompute_statics: bool = False) -> Tensor:
        """Return component log-densities with shape ``(K, n)``."""
        del recompute_statics
        x = self._validate_x(x)
        mu = _wrap_pi(self.mu).to(dtype=x.dtype, device=x.device)
        binding_kappa = self.binding_kappa.to(dtype=x.dtype, device=x.device)

        if self.oscillatory_data:
            # For q=+1, a+ib = sum_j lambda_j exp(i(x_j-mu_j)).
            # A matrix product evaluates all component resultants without ever
            # constructing the otherwise dominant (K,n,p) phase tensor.
            complex_dtype = (
                torch.complex128 if x.dtype == torch.float64 else torch.complex64
            )
            observations = torch.exp(1j * x.to(complex_dtype))
            coefficients = binding_kappa.to(complex_dtype) * torch.exp(
                -1j * mu.to(complex_dtype)
            )
            resultant = torch.abs(observations @ coefficients.T).T
            marginal_log_pdf = -self.p * math.log(_TWO_PI)
        else:
            # Shape: (K,n,p). Nonuniform marginals require each component's
            # shifted angle for both its marginal density and CDF transform.
            centered = _wrap_pi(x.unsqueeze(0) - mu.unsqueeze(1))
            concentrations = binding_kappa.unsqueeze(1)
            marginal_kappa = self.marginal_kappa.to(dtype=x.dtype, device=x.device)
            marginal_log_pdf = (
                marginal_kappa.unsqueeze(1) * torch.cos(centered)
                - math.log(_TWO_PI)
                - _log_i0(marginal_kappa).unsqueeze(1)
            ).sum(dim=-1)
            cdf = self._von_mises_cdf(centered, marginal_kappa)
            u = _TWO_PI * cdf
            a = torch.sum(concentrations * torch.cos(u), dim=-1)
            b = torch.sum(
                concentrations
                * self.q.to(dtype=x.dtype, device=x.device).unsqueeze(1)
                * torch.sin(u),
                dim=-1,
            )
            resultant = torch.hypot(a, b)

        dependence_log_pdf = (
            _log_i0(resultant)
            - torch.sum(_log_i0(binding_kappa), dim=-1, keepdim=True)
        )
        return marginal_log_pdf + dependence_log_pdf

    def pdf(self, x: Tensor) -> Tensor:
        return torch.exp(self.log_pdf(x))

    def log_prob(self, x: Tensor) -> Tensor:
        """Alias used by distribution-like code; returns ``(K,n)``."""
        return self.log_pdf(x)

    def component_log_prob(self, x: Tensor) -> Tensor:
        return self.log_pdf(x)

    def negative_log_likelihood(self, x: Tensor, *, reduction: str = "sum") -> Tensor:
        ll = self.forward(x)
        if reduction == "sum":
            return -ll
        if reduction == "mean":
            return -ll / x.shape[0]
        raise ValueError("reduction must be 'sum' or 'mean'")

    def loss(self, x: Tensor) -> Tensor:
        return self.negative_log_likelihood(x)

    # ------------------------------------------------------------------
    # Differentiable von Mises CDF
    # ------------------------------------------------------------------
    def _von_mises_cdf(self, centered: Tensor, kappa: Tensor) -> Tensor:
        """Approximate F(theta) on [-pi,pi] by a differentiable CDF table.

        The table is computed from a stabilized unnormalized density. Its
        cumulative trapezoid is normalized to end exactly at one, so no Bessel
        normalizer is required in this step. Gradients propagate through the
        interpolation and through every table value with respect to kappa.
        """
        m = self.cdf_grid_size
        dtype, device = centered.dtype, centered.device
        grid = torch.linspace(-math.pi, math.pi, m + 1, dtype=dtype, device=device)
        step = _TWO_PI / m

        # (K,p,m+1); subtracting 1 inside the exponential prevents overflow.
        density = torch.exp(kappa.unsqueeze(-1) * (torch.cos(grid) - 1.0))
        increments = 0.5 * (density[..., :-1] + density[..., 1:]) * step
        cumulative = torch.cat(
            [torch.zeros_like(increments[..., :1]), torch.cumsum(increments, dim=-1)],
            dim=-1,
        )
        cumulative = cumulative / cumulative[..., -1:].clamp_min(torch.finfo(dtype).tiny)

        position = ((centered + math.pi) / step).clamp(0.0, float(m))
        left = torch.floor(position).to(torch.long).clamp(max=m - 1)
        fraction = position - left.to(dtype)

        # Expand table over observations and gather along the grid dimension.
        table = cumulative.unsqueeze(1).expand(-1, centered.shape[1], -1, -1)
        left_value = torch.gather(table, -1, left.unsqueeze(-1)).squeeze(-1)
        right_value = torch.gather(table, -1, (left + 1).unsqueeze(-1)).squeeze(-1)
        return left_value + fraction * (right_value - left_value)

    # ------------------------------------------------------------------
    # Initialization and parameter IO
    # ------------------------------------------------------------------
    @torch.no_grad()
    def initialize(
        self,
        data: Optional[Tensor] = None,
        posterior: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "VMVM":
        """Initialize component parameters from data and optional assignments.

        ``posterior`` may be shaped ``(K,n)``, ``(n,K)``, or be an integer
        label vector of length ``n``. If it is omitted, ``init_method='tc'``
        uses torus K-means, ``init_method='qtc'`` restores a zero reference
        phase and uses quotient-torus K-means, and ``init_method='dc'`` uses
        complex diametrical K-means. Explicit posterior assignments always
        take precedence.
        """
        X = kwargs.pop("X", None)
        init_method = kwargs.pop("init_method", None)
        tol = kwargs.pop("tol", 1e-10)
        if kwargs:
            raise TypeError(f"Unexpected initialization arguments: {tuple(kwargs)}")
        del args
        data = self._resolve_x(data, X)
        data = self._validate_x(data)
        n = data.shape[0]

        if posterior is None and init_method in {
            "tc", "tc++", "torus", "torus_clustering",
        }:
            from PCMM.phase_coherence_kmeans import torus_clustering

            _, labels, _ = torus_clustering(
                data.detach().cpu().numpy(),
                K=self.K,
                init="++",
                num_repl=1,
                tol=tol,
                suppress_output=True,
            )
            posterior = torch.as_tensor(labels, device=data.device)
        elif posterior is None and init_method in {
            "qtc", "qtc++", "quotient_torus", "quotient_torus_clustering",
        }:
            from PCMM.phase_coherence_kmeans import quotient_torus_clustering

            quotient_data = torch.cat(
                [data, torch.zeros_like(data[:, :1])],
                dim=1,
            )
            _, labels, _ = quotient_torus_clustering(
                quotient_data.detach().cpu().numpy(),
                K=self.K,
                init="++",
                num_repl=1,
                tol=tol,
                suppress_output=False,
            )
            posterior = torch.as_tensor(labels, device=data.device)
        elif posterior is None and init_method in {
            "dc", "dc++", "diametrical", "diametrical_clustering",
        }:
            from PCMM.phase_coherence_kmeans import diametrical_clustering

            projective = torch.exp(1j * data) / math.sqrt(self.p)
            _, labels, _ = diametrical_clustering(
                projective.detach().cpu().numpy(),
                K=self.K,
                init="++",
                num_repl=1,
                tol=tol,
                suppress_output=False,
            )
            posterior = torch.as_tensor(labels, device=data.device)
        elif posterior is None and init_method not in {None, "unif", "uniform"}:
            raise ValueError(
                "VMVM init_method must be 'tc', 'qtc', 'dc', 'unif', or 'uniform'"
            )

        if posterior is None:
            indices = torch.linspace(0, max(n - 1, 0), self.K, device=data.device).round().long()
            responsibilities = F.one_hot(torch.arange(n, device=data.device) % self.K, self.K).T
            if n >= self.K:
                responsibilities = torch.zeros(self.K, n, dtype=data.dtype, device=data.device)
                responsibilities[torch.arange(self.K, device=data.device), indices] = 1.0
                # Assign remaining points to their closest initial angular center.
                centers = data[indices]
                dist = 1.0 - torch.cos(data.unsqueeze(1) - centers.unsqueeze(0)).mean(dim=-1)
                labels = torch.argmin(dist, dim=1)
                responsibilities.zero_()
                responsibilities[labels, torch.arange(n, device=data.device)] = 1.0
            else:
                responsibilities = responsibilities.to(dtype=data.dtype)
        else:
            posterior = torch.as_tensor(posterior, device=data.device)
            if posterior.ndim == 1:
                labels = posterior.to(torch.long)
                if labels.numel() != n or labels.min() < 0 or labels.max() >= self.K:
                    raise ValueError("label vector must have length n and values in [0,K)")
                responsibilities = F.one_hot(labels, self.K).T.to(dtype=data.dtype)
            elif posterior.shape == (self.K, n):
                responsibilities = posterior.to(dtype=data.dtype)
            elif posterior.shape == (n, self.K):
                responsibilities = posterior.T.to(dtype=data.dtype)
            else:
                raise ValueError("posterior must have shape (K,n), (n,K), or (n,)")

        responsibilities = responsibilities.clamp_min(0)
        responsibilities = responsibilities / responsibilities.sum(dim=0, keepdim=True).clamp_min(1e-12)
        totals = responsibilities.sum(dim=1).clamp_min(1e-8)

        z = torch.exp(1j * data.to(torch.complex128 if data.dtype == torch.float64 else torch.complex64))
        if self.oscillatory_data:
            # A rotating carrier makes ordinary marginal means vanish. Use a
            # gauge-fixed relative-phase template for dependence initialization.
            z = z * z[:, :1].conj()
        weighted_z = torch.einsum("kn,np->kp", responsibilities.to(z.dtype), z)
        means = torch.angle(weighted_z)
        resultant = torch.abs(weighted_z) / totals.unsqueeze(1)
        kappas = self._a1_inverse(resultant.clamp(0.0, 1.0 - 1e-6)).to(data.dtype)
        kappas = kappas.clamp(min=0.05, max=100.0)

        self.mu.copy_(means.to(self.mu))
        if not self.oscillatory_data:
            self.raw_marginal_kappa.copy_(
                _inverse_softplus(kappas.to(self.mu) - self.min_concentration + torch.finfo(self.mu.dtype).eps)
            )
        # Mild dependence is a safer optimizer starting point than near independence.
        bind = torch.full_like(self.mu, 0.5)
        self.raw_binding_kappa.copy_(
            _inverse_softplus(bind - self.min_concentration + torch.finfo(self.mu.dtype).eps)
        )

        weights = (totals / totals.sum()).to(self.pi)
        self.pi.copy_(torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny)))

        if self.oscillatory_data:
            self.q.fill_(1)
        else:
            # Pick signs from the stronger sum/difference circular resultant
            # after the current marginal CDF transformation.
            transformed = _TWO_PI * self._von_mises_cdf(
                _wrap_pi(data.unsqueeze(0) - self.mu.unsqueeze(1)), self.marginal_kappa
            )
            q_new = torch.ones_like(self.q)
            for k in range(self.K):
                w = responsibilities[k]
                for j in range(1, self.p):
                    same = torch.abs(torch.sum(w * torch.exp(1j * (transformed[k, :, j] - transformed[k, :, 0]))))
                    opposite = torch.abs(torch.sum(w * torch.exp(1j * (transformed[k, :, j] + transformed[k, :, 0]))))
                    q_new[k, j] = 1.0 if same >= opposite else -1.0
            self.q.copy_(q_new)

        if self.HMM:
            self.pi.copy_(torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny)))
            transition_counts = torch.ones(self.K, self.K, dtype=data.dtype, device=data.device) * 1e-2
            labels = torch.argmax(responsibilities, dim=0)
            lengths, starts = self._format_samples_per_sequence(n)
            for length, offset in zip(lengths.tolist(), starts.tolist()):
                seq_labels = labels[offset : offset + length]
                if length > 1:
                    transition_counts.index_put_(
                        (seq_labels[:-1], seq_labels[1:]),
                        torch.ones(length - 1, dtype=data.dtype, device=data.device),
                        accumulate=True,
                    )
            transition = transition_counts / transition_counts.sum(dim=1, keepdim=True)
            self.T.copy_(torch.log(transition.to(self.T)))
        return self

    def get_params(self, *, numpy: bool = False) -> dict[str, Any]:
        """Return a restartable parameter dictionary."""
        out: dict[str, Tensor] = {
            "mu": _wrap_pi(self.mu.detach()).clone(),
            "kappa": self.marginal_kappa.detach().clone(),
            "lambda": self.binding_kappa.detach().clone(),
            "q": self.q.detach().clone(),
            "pi": self.weights.detach().clone(),
        }
        if self.HMM:
            out["T"] = self.transition.detach().clone()
        if numpy:
            return {key: value.cpu().numpy() for key, value in out.items()}
        return out

    @torch.no_grad()
    def set_params(self, params: Mapping[str, Any]) -> "VMVM":
        """Set recognized parameters in place."""
        dtype, device = self.mu.dtype, self.mu.device
        if any(k in params for k in ("mu", "means", "mean")):
            value = self._read_parameter(params, ("mu", "means", "mean"), None, dtype, device)
            self.mu.copy_(_wrap_pi(self._expand_kp(value, "mu")))
        if any(k in params for k in ("kappa", "marginal_kappa", "marginal_kappas")):
            value = self._read_parameter(
                params, ("kappa", "marginal_kappa", "marginal_kappas"), None, dtype, device
            )
            value = self._expand_kp(value, "kappa").clamp_min(self.min_concentration)
            if not self.oscillatory_data:
                self.raw_marginal_kappa.copy_(
                    _inverse_softplus(value - self.min_concentration + torch.finfo(dtype).eps)
                )
        if any(k in params for k in ("lambda", "binding_kappa", "binding_kappas", "circula_kappa")):
            value = self._read_parameter(
                params, ("lambda", "binding_kappa", "binding_kappas", "circula_kappa"),
                None, dtype, device,
            )
            value = self._expand_kp(value, "binding_kappa").clamp_min(self.min_concentration)
            self.raw_binding_kappa.copy_(
                _inverse_softplus(value - self.min_concentration + torch.finfo(dtype).eps)
            )
        if self.oscillatory_data:
            self.q.fill_(1)
        elif any(k in params for k in ("q", "qs")):
            value = self._read_parameter(params, ("q", "qs"), None, dtype, device)
            value = self._expand_kp(value, "q")
            if not torch.all((value == 1) | (value == -1)):
                raise ValueError("Every q entry must equal -1 or +1")
            self.q.copy_(value * value[:, :1])
        if any(k in params for k in ("weights", "alpha", "pi", "mixing_weights")):
            value = self._read_parameter(
                params, ("weights", "alpha", "pi", "mixing_weights"), None, dtype, device
            )
            value = self._expand_vector(value, self.K, "weights").clamp_min(torch.finfo(dtype).tiny)
            self.pi.copy_(torch.log(value / value.sum()))
        if self.HMM and any(k in params for k in ("initial", "pi0", "initial_probs")):
            value = self._read_parameter(params, ("initial", "pi0", "initial_probs"), None, dtype, device)
            value = self._expand_vector(value, self.K, "initial").clamp_min(torch.finfo(dtype).tiny)
            self.pi.copy_(torch.log(value / value.sum()))
        if self.HMM and any(k in params for k in ("transition", "T", "transition_matrix")):
            value = self._read_parameter(
                params, ("transition", "T", "transition_matrix"), None, dtype, device
            )
            if value.shape != (self.K, self.K):
                raise ValueError(f"transition must have shape {(self.K, self.K)}")
            value = value.clamp_min(torch.finfo(dtype).tiny)
            self.T.copy_(torch.log(value / value.sum(dim=1, keepdim=True)))
        return self

    # Common aliases used by generic fitting code.
    get_parameters = get_params
    set_parameters = set_params
    init_params = initialize

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        size: int = 1,
        *,
        component: Optional[int | Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Draw mixture samples, returned in ``[-pi,pi)`` with shape ``(size,p)``."""
        if not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer")
        device, dtype = self.mu.device, self.mu.dtype
        if component is None:
            components = torch.multinomial(self.weights, size, replacement=True, generator=generator)
        elif isinstance(component, int):
            if component < 0 or component >= self.K:
                raise ValueError("component index out of range")
            components = torch.full((size,), component, dtype=torch.long, device=device)
        else:
            components = torch.as_tensor(component, dtype=torch.long, device=device)
            if components.shape != (size,) or components.min() < 0 or components.max() >= self.K:
                raise ValueError("component tensor must have shape (size,) and entries in [0,K)")

        result = torch.empty(size, self.p, dtype=dtype, device=device)
        for k in torch.unique(components).tolist():
            mask = components == k
            count = int(mask.sum())
            phi = torch.rand(count, 1, dtype=dtype, device=device, generator=generator) * _TWO_PI
            shifts = torch.distributions.VonMises(
                torch.zeros(self.p, dtype=dtype, device=device), self.binding_kappa[k]
            ).sample((count,))
            uniforms = torch.remainder(shifts + self.q[k].unsqueeze(0) * phi, _TWO_PI) / _TWO_PI
            if self.oscillatory_data:
                centered = _TWO_PI * uniforms - math.pi
            else:
                centered = self._von_mises_icdf(uniforms, self.marginal_kappa[k])
            result[mask] = _wrap_pi(centered + self.mu[k])
        return result

    def _von_mises_icdf(self, probabilities: Tensor, kappa: Tensor) -> Tensor:
        m = self.cdf_grid_size
        dtype, device = probabilities.dtype, probabilities.device
        grid = torch.linspace(-math.pi, math.pi, m + 1, dtype=dtype, device=device)
        step = _TWO_PI / m
        density = torch.exp(kappa.unsqueeze(-1) * (torch.cos(grid) - 1.0))
        increments = 0.5 * (density[..., :-1] + density[..., 1:]) * step
        cdf = torch.cat(
            [torch.zeros_like(increments[..., :1]), torch.cumsum(increments, dim=-1)], dim=-1
        )
        cdf = cdf / cdf[..., -1:].clamp_min(torch.finfo(dtype).tiny)

        out = torch.empty_like(probabilities)
        for j in range(self.p):
            values = probabilities[:, j].clamp(0.0, 1.0)
            idx = torch.searchsorted(cdf[j].contiguous(), values.contiguous(), right=True)
            idx = idx.clamp(1, m)
            c0, c1 = cdf[j, idx - 1], cdf[j, idx]
            fraction = (values - c0) / (c1 - c0).clamp_min(torch.finfo(dtype).eps)
            out[:, j] = grid[idx - 1] + fraction * step
        return out

    # ------------------------------------------------------------------
    # Validation and shape helpers
    # ------------------------------------------------------------------
    def _validate_x(self, x: Tensor) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.mu.dtype, device=self.mu.device)
        else:
            x = x.to(device=self.mu.device)
            if not x.is_floating_point():
                x = x.to(dtype=self.mu.dtype)
        if x.ndim != 2 or x.shape[1] != self.p:
            raise ValueError(f"x must have shape (n,{self.p}), got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or infinite values")
        return x

    @staticmethod
    def _resolve_x(x: Optional[Tensor], X: Optional[Tensor]) -> Tensor:
        if x is not None and X is not None:
            raise TypeError("Provide only one of x or X")
        value = x if x is not None else X
        if value is None:
            raise TypeError("Missing required input data (x or X)")
        return value

    @staticmethod
    def _read_parameter(
        params: Mapping[str, Any],
        names: Sequence[str],
        default: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[Tensor]:
        for name in names:
            if name in params:
                return _as_tensor(params[name], dtype=dtype, device=device)
        if default is None:
            return None
        return _as_tensor(default, dtype=dtype, device=device)

    def _expand_kp(self, value: Tensor, name: str) -> Tensor:
        value = torch.as_tensor(value, dtype=self.mu.dtype, device=self.mu.device) if hasattr(self, "mu") else value
        if value.ndim == 0:
            return value.expand(self.K, self.p).clone()
        if value.shape == (self.p,):
            return value.unsqueeze(0).expand(self.K, -1).clone()
        if value.shape == (1, self.p):
            return value.expand(self.K, -1).clone()
        if value.shape != (self.K, self.p):
            raise ValueError(f"{name} must be scalar or have shape ({self.p},) or {(self.K, self.p)}")
        return value.clone()

    @staticmethod
    def _expand_vector(value: Tensor, length: int, name: str) -> Tensor:
        if value.ndim == 0:
            return value.expand(length).clone()
        value = value.reshape(-1)
        if value.numel() != length:
            raise ValueError(f"{name} must contain {length} values")
        return value.clone()

    @staticmethod
    def _a1_inverse(r: Tensor) -> Tensor:
        low = 2 * r + r**3 + 5 * r**5 / 6
        mid = -0.4 + 1.39 * r + 0.43 / (1 - r)
        high = 1 / (r**3 - 4 * r**2 + 3 * r)
        return torch.where(r < 0.53, low, torch.where(r < 0.85, mid, high))


__all__ = ["VMVM"]
