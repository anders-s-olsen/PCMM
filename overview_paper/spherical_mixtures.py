"""
Sample a two-component Bingham mixture on S^2 and visualize fitted
Watson, angular central Gaussian (ACG), and Bingham mixtures.

The density is evaluated on a fine angular grid, numerically normalized
there, and only then interpolated onto a coarser sphere used for plotting.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.interpolate import RegularGridInterpolator
import torch

# Allow both ``python -m overview_paper.spherical_mixtures`` and direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PCMM.PCMMtorch import ACG, Bingham, Watson
from PCMM.mixture_torch_loop import mixture_torch_loop


# ---------------------------------------------------------------------
# Geometry and sampling
# ---------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    """Return v normalized to unit length."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("A zero vector cannot be normalized.")
    return v / n


def tangent_frame(mu: np.ndarray) -> np.ndarray:
    """
    Return a 3x3 orthogonal matrix [e1, e2, mu].

    e1 and e2 span the tangent plane perpendicular to mu.
    """
    mu = unit(mu)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(mu @ ref) > 0.90:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = unit(np.cross(ref, mu))
    e2 = np.cross(mu, e1)
    return np.column_stack((e1, e2, mu))


def sample_uniform_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly from S^2."""
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def sample_bingham(
    n: int,
    A: np.ndarray,
    rng: np.random.Generator,
    min_batch: int = 4096,
    max_proposals: int = 20_000_000,
) -> np.ndarray:
    r"""
    Sample from the Bingham density on S^2

        f(x) proportional to exp(x.T @ A @ x),   ||x|| = 1.

    A must be symmetric. Adding c*I to A does not alter the distribution,
    so the code shifts its largest eigenvalue to zero. This makes
    x.T @ A @ x <= 0 and permits rejection sampling from the uniform sphere.

    This simple sampler is convenient for moderate concentration. For very
    concentrated fits, replace it with a dedicated Bingham sampler.
    """
    A = np.asarray(A, dtype=float)
    if A.shape != (3, 3):
        raise ValueError("A must be a 3x3 matrix.")
    A = 0.5 * (A + A.T)

    # Shift to the standard identifiable form: largest eigenvalue = 0.
    A = A - np.linalg.eigvalsh(A).max() * np.eye(3)

    if np.allclose(A, 0.0):
        return sample_uniform_sphere(n, rng)

    accepted: list[np.ndarray] = []
    n_accepted = 0
    n_proposed = 0
    acceptance_guess = 0.10

    while n_accepted < n:
        remaining = n - n_accepted
        batch = max(min_batch, int(1.4 * remaining / max(acceptance_guess, 1e-3)))

        x = sample_uniform_sphere(batch, rng)
        log_acceptance = np.einsum("ni,ij,nj->n", x, A, x)
        keep = np.log(rng.random(batch)) < log_acceptance

        if np.any(keep):
            accepted.append(x[keep])
            n_accepted += int(keep.sum())

        n_proposed += batch
        acceptance_guess = max(n_accepted / n_proposed, 1e-4)

        if n_proposed > max_proposals and n_accepted < n:
            raise RuntimeError(
                "Bingham rejection sampler is too inefficient for this A. "
                "Use weaker concentration or a specialized sampler."
            )

    return np.concatenate(accepted, axis=0)[:n]


def sample_bingham_mixture(
    n: int,
    weights: np.ndarray,
    A_components: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a finite Bingham mixture and return points and component labels."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    if len(weights) != len(A_components):
        raise ValueError("weights and A_components must have the same length.")

    labels = rng.choice(len(weights), size=n, p=weights)
    x = np.empty((n, 3), dtype=float)

    for k, A in enumerate(A_components):
        idx = np.flatnonzero(labels == k)
        if idx.size:
            x[idx] = sample_bingham(idx.size, A, rng)

    return x, labels


# ---------------------------------------------------------------------
# PCMM fits
# ---------------------------------------------------------------------

def mixture_weights(pi_logits: torch.Tensor) -> np.ndarray:
    """Convert the PCMM mixture-weight logits to probabilities."""
    return torch.softmax(pi_logits, dim=0).detach().cpu().numpy()


def fit_pcmm_models(
    points: np.ndarray,
    reference_directions: list[np.ndarray],
) -> dict[str, tuple[list[dict[str, np.ndarray | float]], np.ndarray]]:
    """Fit Watson, ACG, and Bingham mixtures through the PCMM torch API."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    np.random.seed(0)  # PCMM's analytical initializers use NumPy's global RNG.

    common = dict(
        data=points,
        tol=1e-7,
        max_iter=1500,
        num_repl=1,
        init="dc++",
        suppress_output=True,
        threads=1,
        decrease_lr_on_plateau=False,
        num_comparison=50,
    )

    watson_params, _, _ = mixture_torch_loop(
        Watson(p=3, K=2), LR=0.03, **common
    )
    acg_params, _, _ = mixture_torch_loop(
        ACG(p=3, rank=3, K=2), LR=0.03, **common
    )
    bingham_params, _, _ = mixture_torch_loop(
        Bingham(p=3, K=2), LR=0.03, **common
    )

    watson_mu = watson_params["mu"].detach().cpu().numpy()
    watson_mu /= np.linalg.norm(watson_mu, axis=1, keepdims=True)
    watson_fit = [
        {"mu": watson_mu[k], "kappa": float(watson_params["kappa"][k])}
        for k in range(2)
    ]

    M = acg_params["M"].detach().cpu().numpy()
    Sigma = np.eye(3)[None, :, :] + M @ np.swapaxes(M, -1, -2)
    acg_fit = [{"Sigma": Sigma[k]} for k in range(2)]

    A = bingham_params["A"].detach().cpu().numpy()
    bingham_fit = [{"A": A[k]} for k in range(2)]

    raw_fits = {
        "watson": (watson_fit, mixture_weights(watson_params["pi"])),
        "acg": (acg_fit, mixture_weights(acg_params["pi"])),
        "bingham": (bingham_fit, mixture_weights(bingham_params["pi"])),
    }

    # Keep component colors consistent across panels by aligning each fit's
    # first component with the first generating direction (sign is irrelevant).
    ordered_fits = {}
    reference = unit(reference_directions[0])
    for model, (params, weights) in raw_fits.items():
        if model == "watson":
            directions = [unit(component["mu"]) for component in params]
        elif model == "acg":
            directions = [
                np.linalg.eigh(component["Sigma"])[1][:, -1]
                for component in params
            ]
        else:
            directions = [
                np.linalg.eigh(component["A"])[1][:, -1]
                for component in params
            ]

        order = np.argsort([-abs(direction @ reference) for direction in directions])
        ordered_fits[model] = ([params[k] for k in order], weights[order])

    return ordered_fits


# ---------------------------------------------------------------------
# Log densities
# ---------------------------------------------------------------------

def watson_logpdf_unnormalized(
    x: np.ndarray, mu: np.ndarray, kappa: float
) -> np.ndarray:
    r"""Watson log density, up to its normalizing constant."""
    mu = unit(mu)
    return float(kappa) * (x @ mu) ** 2


def acg_logpdf_unnormalized(x: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    r"""
    Angular central Gaussian log density on S^2, up to an additive constant:

        log f(x) = -1/2 log|Sigma| - 3/2 log(x.T Sigma^{-1} x).

    Multiplying Sigma by a positive scalar does not change the distribution.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)

    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("ACG Sigma must be symmetric positive definite.")

    inv_Sigma = np.linalg.inv(Sigma)
    quadratic = np.einsum("ni,ij,nj->n", x, inv_Sigma, x)
    return -0.5 * logdet - 1.5 * np.log(quadratic)


def bingham_logpdf_unnormalized(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    r"""
    Bingham log density, up to its normalizing constant:

        log f(x) = x.T A x.

    Some software calls A a concentration, scatter, or Psi matrix. Pass the
    matrix that actually appears in the quadratic form. Adding c*I is harmless.
    """
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    return np.einsum("ni,ij,nj->n", x, A, x)


def component_logpdf(
    model: str, params: dict[str, np.ndarray | float], x: np.ndarray
) -> np.ndarray:
    """Dispatch a component log-density by model name."""
    model = model.lower()

    if model == "watson":
        return watson_logpdf_unnormalized(
            x=x,
            mu=np.asarray(params["mu"]),
            kappa=float(params["kappa"]),
        )
    if model == "acg":
        return acg_logpdf_unnormalized(
            x=x,
            Sigma=np.asarray(params["Sigma"]),
        )
    if model == "bingham":
        return bingham_logpdf_unnormalized(
            x=x,
            A=np.asarray(params["A"]),
        )

    raise ValueError(f"Unknown model: {model!r}")


# ---------------------------------------------------------------------
# Fine-grid evaluation and coarse-grid rendering
# ---------------------------------------------------------------------

def sphere_grid(n_theta: int, n_phi: int):
    """
    Build a latitude-longitude grid.

    theta is polar angle in [0, pi], and phi is azimuth in [0, 2*pi].
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    sin_theta = np.sin(theta_grid)
    x = sin_theta * np.cos(phi_grid)
    y = sin_theta * np.sin(phi_grid)
    z = np.cos(theta_grid)

    xyz = np.stack((x, y, z), axis=-1)
    return theta, phi, theta_grid, phi_grid, xyz


def normalize_density_on_grid(
    log_density: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Numerically normalize a density over S^2 using dOmega = sin(theta)dtheta dphi.
    """
    shifted = np.exp(log_density - np.max(log_density))
    integrand = shifted * np.sin(theta)[:, None]

    # np.trapz is retained for compatibility with older NumPy versions.
    z_phi = np.trapz(integrand, phi, axis=1)
    normalizer = np.trapz(z_phi, theta)

    if not np.isfinite(normalizer) or normalizer <= 0:
        raise FloatingPointError("Density normalization failed.")

    return shifted / normalizer


def evaluate_mixture_on_fine_grid(
    model: str,
    component_params: list[dict[str, np.ndarray | float]],
    weights: np.ndarray,
    eval_grid: tuple[int, int] = (361, 721),
):
    """
    Evaluate and normalize each component on a fine grid.

    Returns fine-grid normalized component densities and flags indicating
    which components are effectively isotropic.
    """
    n_theta_eval, n_phi_eval = eval_grid
    theta, phi, _, _, xyz = sphere_grid(n_theta_eval, n_phi_eval)
    points = xyz.reshape(-1, 3)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    normalized = []
    isotropic = []

    for params in component_params:
        logp = component_logpdf(model, params, points).reshape(
            n_theta_eval, n_phi_eval
        )
        normalized.append(normalize_density_on_grid(logp, theta, phi))
        isotropic.append(np.ptp(logp) < 1e-10)

    return theta, phi, np.stack(normalized, axis=0), np.asarray(isotropic), weights


def interpolate_fine_to_visual_grid(
    theta_fine: np.ndarray,
    phi_fine: np.ndarray,
    values_fine: np.ndarray,
    theta_visual: np.ndarray,
    phi_visual: np.ndarray,
) -> np.ndarray:
    """Interpolate one fine-grid scalar field onto the visual sphere grid."""
    interpolator = RegularGridInterpolator(
        (theta_fine, phi_fine),
        values_fine,
        bounds_error=False,
        fill_value=None,
    )
    query = np.column_stack((theta_visual.ravel(), phi_visual.ravel()))
    return interpolator(query).reshape(theta_visual.shape)


VISUAL_GRID = (81, 161)
COMPONENT_COLORS = ("#8b0000", "#006400")  # dark red, dark green
WIREFRAME_COLOR = "#aab1b7"
WIREFRAME_LINEWIDTH = 0.85
WIREFRAME_ALPHA = 0.42


def plot_background_wireframe(ax, xyz: np.ndarray):
    """Draw the identical sphere wireframe in every panel."""
    ax.plot_wireframe(
        xyz[..., 0],
        xyz[..., 1],
        xyz[..., 2],
        rstride=5,
        cstride=9,
        color=WIREFRAME_COLOR,
        linewidth=WIREFRAME_LINEWIDTH,
        alpha=WIREFRAME_ALPHA,
    )


def style_3d_sphere(ax, elev: float = 18.0, azim: float = -55.0):
    """Apply consistent axis limits and camera settings."""
    # Zoom fills the subplot's 3D projection box and removes the large margins
    # Matplotlib otherwise leaves around 3D axes.
    ax.set_box_aspect((1, 1, 1), zoom=1.28)
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_zlim(-1.03, 1.03)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def plot_sample_sphere(
    ax,
    points: np.ndarray,
    labels: np.ndarray,
    visual_grid: tuple[int, int] = VISUAL_GRID,
):
    """Plot sampled points with a faint wireframe sphere."""
    _, _, _, _, xyz = sphere_grid(*visual_grid)
    plot_background_wireframe(ax, xyz)

    for k in np.unique(labels):
        idx = labels == k
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            points[idx, 2],
            s=3,
            alpha=0.55,
            color=COMPONENT_COLORS[k % len(COMPONENT_COLORS)],
            depthshade=False,
        )

    style_3d_sphere(ax)


def plot_mixture_density_sphere(
    ax,
    model: str,
    component_params: list[dict[str, np.ndarray | float]],
    weights: np.ndarray,
    eval_grid: tuple[int, int] = (541, 1081),
    visual_grid: tuple[int, int] = VISUAL_GRID,
    background_alpha: float = 0.025,
    max_alpha: float = 0.88,
    alpha_gamma: float = 0.35,
):
    """
    Render a fitted mixture density on S^2.

    Important: component likelihoods are evaluated and normalized on eval_grid,
    which is intentionally much finer than visual_grid. The normalized values
    are then interpolated onto the coarse display mesh.

    Isotropic components are treated as a faint background. Non-isotropic
    components control the visible color and opacity.
    """
    theta_f, phi_f, density_f, isotropic, weights = (
        evaluate_mixture_on_fine_grid(
            model=model,
            component_params=component_params,
            weights=weights,
            eval_grid=eval_grid,
        )
    )

    _, _, theta_v, phi_v, xyz_v = sphere_grid(*visual_grid)

    density_v = np.stack(
        [
            interpolate_fine_to_visual_grid(
                theta_f,
                phi_f,
                density_f[k],
                theta_v,
                phi_v,
            )
            for k in range(density_f.shape[0])
        ],
        axis=0,
    )

    contributions = weights[:, None, None] * density_v
    foreground_mask = ~isotropic

    if np.any(foreground_mask):
        foreground = contributions[foreground_mask].sum(axis=0)
        scale = np.quantile(foreground, 0.995)
        if scale <= 0:
            strength = np.zeros_like(foreground)
        else:
            strength = np.clip(foreground / scale, 0.0, 1.0) ** alpha_gamma
    else:
        foreground = np.zeros(theta_v.shape)
        strength = np.zeros(theta_v.shape)

    background_rgb = np.array([0.88, 0.90, 0.92])

    # Blend colors by non-isotropic component responsibility.
    if np.any(foreground_mask):
        fg_indices = np.flatnonzero(foreground_mask)
        fg_contrib = contributions[foreground_mask]
        denom = fg_contrib.sum(axis=0)

        rgb_numerator = np.zeros(theta_v.shape + (3,), dtype=float)
        for local_index, component_index in enumerate(fg_indices):
            rgb = np.asarray(
                to_rgba(COMPONENT_COLORS[component_index % len(COMPONENT_COLORS)])
            )[:3]
            rgb_numerator += fg_contrib[local_index][..., None] * rgb

        foreground_rgb = rgb_numerator / np.maximum(denom[..., None], 1e-15)
    else:
        foreground_rgb = np.broadcast_to(background_rgb, theta_v.shape + (3,))

    rgb = (
        (1.0 - strength[..., None]) * background_rgb
        + strength[..., None] * foreground_rgb
    )
    alpha = background_alpha + (max_alpha - background_alpha) * strength
    rgba = np.dstack((rgb, alpha))

    ax.plot_surface(
        xyz_v[..., 0],
        xyz_v[..., 1],
        xyz_v[..., 2],
        facecolors=rgba,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    plot_background_wireframe(ax, xyz_v)

    style_3d_sphere(ax)


# ---------------------------------------------------------------------
# End-to-end example
# ---------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    # Well-separated antipodal axes make the two populations easy to see.
    mu_isotropic = unit(np.array([-0, -0.61, 0.33]))
    mu_anisotropic = unit(np.array([0.58, -0.23, 0.78]))
    R_isotropic = tangent_frame(mu_isotropic)
    R_anisotropic = tangent_frame(mu_anisotropic)

    # Component 0: concentrated and isotropic around its axis. Equal
    # tangent-plane eigenvalues give circular (rather than elliptical) contours.
    A_isotropic = R_isotropic @ np.diag([-14.0, -14.0, 0.0]) @ R_isotropic.T

    # Component 1: anisotropic Bingham.
    # The last eigenvector is mu_anisotropic and has eigenvalue 0, hence modes
    # at +/-mu_anisotropic.
    # Unequal negative tangent-plane eigenvalues produce elliptical contours.
    A_anisotropic = (
        R_anisotropic @ np.diag([-70.0, -8.0, 0.0]) @ R_anisotropic.T
    )

    true_weights = np.array([0.50, 0.50])
    true_A = [A_isotropic, A_anisotropic]

    points, labels = sample_bingham_mixture(
        n=1600,
        weights=true_weights,
        A_components=true_A,
        rng=rng,
    )

    fits = fit_pcmm_models(
        points,
        reference_directions=[mu_isotropic, mu_anisotropic],
    )

    fig = plt.figure(figsize=(10.0, 8.0))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    plot_sample_sphere(ax, points, labels)

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    plot_mixture_density_sphere(
        ax,
        model="watson",
        component_params=fits["watson"][0],
        weights=fits["watson"][1],
        eval_grid=(541, 1081),   # fine likelihood grid
        visual_grid=VISUAL_GRID,  # coarser sphere mesh
    )

    ax = fig.add_subplot(2, 2, 3, projection="3d")
    plot_mixture_density_sphere(
        ax,
        model="acg",
        component_params=fits["acg"][0],
        weights=fits["acg"][1],
        eval_grid=(541, 1081),
        visual_grid=VISUAL_GRID,
    )

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    plot_mixture_density_sphere(
        ax,
        model="bingham",
        component_params=fits["bingham"][0],
        weights=fits["bingham"][1],
        eval_grid=(541, 1081),
        visual_grid=VISUAL_GRID,
    )

    # Figure-level labels avoid the oversized title margins that Matplotlib
    # reserves independently for each 3D subplot.
    title_style = dict(ha="center", va="top", fontsize=plt.rcParams["axes.titlesize"])
    fig.text(0.305, 0.985, "Samples: isotropic concentrated + anisotropic", **title_style)
    fig.text(0.695, 0.985, "Watson mixture", **title_style)
    fig.text(0.305, 0.505, "Angular central Gaussian mixture", **title_style)
    fig.text(0.695, 0.505, "Bingham mixture", **title_style)

    fig.subplots_adjust(
        left=-0.03,
        right=1.03,
        bottom=0.01,
        top=0.97,
        wspace=-0.42,
        hspace=-0.08,
    )

    output = "overview_paper/spherical_mixture_demo.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
