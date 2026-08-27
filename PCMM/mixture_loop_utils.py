"""Internal helpers shared by the mixture-model optimization loops."""

import numpy as np


_TRACE_FREE_QUADRATIC_DISTRIBUTIONS = {
    'Bingham_lowrank',
    'Complex_Bingham_lowrank',
    'FisherBingham_lowrank',
    'ACG_lowrank',
    'Complex_ACG_lowrank',
    'MACG_lowrank',
    'MatrixBingham_lowrank',
    'MatrixFisherBingham_lowrank',
}


def _convergence_scale(data, convergence_normalization, intrinsic_dimension):
    """Return the divisor used only by the convergence diagnostic."""
    if convergence_normalization in (None, False, 'none'):
        return 1.0
    if convergence_normalization in (True, 'observations_and_dimension'):
        if intrinsic_dimension is None or intrinsic_dimension <= 0:
            raise ValueError(
                'A positive intrinsic_dimension is required when '
                "convergence_normalization is 'observations_and_dimension'."
            )
        return float(data.shape[0]) * float(intrinsic_dimension)
    if convergence_normalization == 'observations':
        return float(data.shape[0])
    raise ValueError(
        "convergence_normalization must be None, 'none', 'observations', or "
        "'observations_and_dimension'."
    )


def _identifiable_quadratic_parameter(model, natural):
    """Remove the identity direction when it is a distributional gauge."""
    if model.distribution not in _TRACE_FREE_QUADRATIC_DISTRIBUTIONS:
        return natural
    import torch

    dimension = natural.shape[-1]
    identity = torch.eye(dimension, dtype=natural.dtype, device=natural.device)
    trace = natural.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / dimension
    return natural - trace.to(natural.dtype)[:, None, None] * identity


def _standardize_isotropic_factor_spectra(model):
    """Keep seeded orientations but remove random singular-value conditioning."""
    import torch

    with torch.no_grad():
        if hasattr(model, 'M'):
            rank = model.M.shape[-1]
            fixed_factors = torch.empty_like(model.M)
            for component in range(model.M.shape[0]):
                frame, _ = torch.linalg.qr(model.M[component], mode='reduced')
                if rank > 1:
                    # A bounded, deterministic, non-repeated profile avoids
                    # singular SVD/eigendecomposition derivatives. It also
                    # keeps a full-rank covariance perturbation identifiable;
                    # an equal profile would be a pure identity shift for
                    # several directional families.
                    eigenvalues = torch.linspace(
                        0.5,
                        1.5,
                        rank,
                        dtype=model.M.real.dtype,
                        device=model.M.device,
                    )
                else:
                    eigenvalues = torch.ones(
                        rank,
                        dtype=model.M.real.dtype,
                        device=model.M.device,
                    )
                fixed_factors[component] = (
                    frame * torch.sqrt(eigenvalues)[None, :]
                )
            model.M.copy_(fixed_factors)

        if hasattr(model, 'F'):
            fixed_linear = torch.empty_like(model.F)
            for component in range(model.F.shape[0]):
                left, _, right_h = torch.linalg.svd(
                    model.F[component], full_matrices=False
                )
                rank = min(model.F.shape[-2:])
                singular_values = model.F.real.new_full(
                    (rank,), 1.0 / np.sqrt(rank)
                )
                fixed_linear[component] = (
                    left[:, :rank] * singular_values[None, :]
                ) @ right_h[:rank]
            model.F.copy_(fixed_linear)
        elif hasattr(model, 'F_left') and hasattr(model, 'F_right'):
            for component in range(model.F_left.shape[0]):
                product = model.F_left[component] @ model.F_right[component].mT
                left, _, right_h = torch.linalg.svd(product, full_matrices=False)
                rank = model.F_left.shape[-1]
                singular_values = product.real.new_full(
                    (rank,), 1.0 / np.sqrt(rank)
                )
                roots = torch.sqrt(singular_values)
                model.F_left[component].copy_(left[:, :rank] * roots[None, :])
                model.F_right[component].copy_(
                    right_h.mT[:, :rank] * roots[None, :]
                )


def _rescale_isotropic_parameters(model, natural_scale, wrapped_variance=None):
    """Put supported models at a common small perturbation from isotropy.

    Seeds determine Haar orientation, while a bounded deterministic factor
    spectrum fixes conditioning. Quadratic gauge directions are removed
    before measuring their Frobenius norm, and combined models divide the
    requested total norm equally among natural-parameter blocks. This
    operation is opt-in so historical PCMM callers retain their original
    initialization exactly.
    """
    if natural_scale is None:
        return
    if natural_scale <= 0:
        raise ValueError(
            'isotropic_natural_scale must be positive when provided.'
        )
    import torch

    scale = float(natural_scale)
    block_count = sum(
        (
            int(hasattr(model, 'M')),
            int(
                hasattr(model, 'F')
                or (hasattr(model, 'F_left') and hasattr(model, 'F_right'))
            ),
            int(hasattr(model, 'kappa')),
        )
    )
    # Combined Fisher--Bingham models should not receive a stronger start just
    # because they contain two natural-parameter blocks. Give each orthogonal
    # block an equal share of the requested total Euclidean/Frobenius norm.
    block_scale = scale / np.sqrt(max(block_count, 1))
    with torch.no_grad():
        _standardize_isotropic_factor_spectra(model)
        if hasattr(model, 'M'):
            natural = model.M @ model.M.mH
            identifiable = _identifiable_quadratic_parameter(model, natural)
            norms = torch.linalg.vector_norm(
                identifiable.reshape(identifiable.shape[0], -1), dim=1
            )
            if torch.any(norms == 0):
                raise RuntimeError(
                    'Isotropic initialization produced a zero quadratic factor.'
                )
            multiplier = torch.sqrt(model.M.new_tensor(block_scale) / norms)
            model.M.mul_(
                multiplier.reshape((-1,) + (1,) * (model.M.ndim - 1))
            )

        if hasattr(model, 'F'):
            flattened = model.F.reshape(model.F.shape[0], -1)
            norms = torch.linalg.vector_norm(flattened, dim=1)
            if torch.any(norms == 0):
                raise RuntimeError(
                    'Isotropic initialization produced a zero matrix linear term.'
                )
            model.F.mul_((block_scale / norms).reshape((-1, 1, 1)))
        elif hasattr(model, 'F_left') and hasattr(model, 'F_right'):
            natural = model.F_left @ model.F_right.mT
            norms = torch.linalg.vector_norm(
                natural.reshape(natural.shape[0], -1), dim=1
            )
            if torch.any(norms == 0):
                raise RuntimeError(
                    'Isotropic initialization produced zero matrix linear factors.'
                )
            multiplier = torch.sqrt(block_scale / norms)
            model.F_left.mul_(multiplier[:, None, None])
            model.F_right.mul_(multiplier[:, None, None])

        if hasattr(model, 'kappa'):
            target_scale = block_scale
            if model.distribution in {'Watson', 'Complex_Watson'}:
                # The identifiable Watson parameter is
                # kappa*(mu mu.H - I/p), whose Frobenius norm is
                # |kappa|*sqrt(1-1/p).
                target_scale /= np.sqrt(1.0 - 1.0 / model.p)
            target = model.kappa.new_full(model.kappa.shape, target_scale)
            if model.distribution in {'VonMisesFisher', 'FisherBingham_lowrank'}:
                target = target + torch.log(-torch.expm1(-target))
            model.kappa.copy_(target)

        if (
            wrapped_variance is not None
            and model.distribution == 'WrappedNormal_lowrank'
        ):
            variance = model.gamma.new_full(model.gamma.shape, wrapped_variance)
            raw_variance = variance + torch.log(-torch.expm1(-variance))
            model.gamma.copy_(raw_variance)


def _initial_natural_parameter_norms(model):
    """Summarize the seeded perturbation in model natural-parameter blocks."""
    import torch

    blocks = {}
    with torch.no_grad():
        if hasattr(model, 'M'):
            quadratic = model.M @ model.M.mH
            identifiable_quadratic = _identifiable_quadratic_parameter(
                model, quadratic
            )
            blocks['quadratic'] = torch.linalg.vector_norm(
                identifiable_quadratic.reshape(
                    identifiable_quadratic.shape[0], -1
                ),
                dim=1,
            )
            raw_quadratic_norms = torch.linalg.vector_norm(
                quadratic.reshape(quadratic.shape[0], -1), dim=1
            )
            factor_singular_values = torch.linalg.svdvals(model.M)
            quadratic_positive_eigenvalue_conditions = (
                factor_singular_values[:, 0] / factor_singular_values[:, -1]
            ).square()

        if hasattr(model, 'F'):
            linear_matrix = model.F
            positive_linear_rank = min(linear_matrix.shape[-2:])
            blocks['matrix_linear'] = torch.linalg.vector_norm(
                linear_matrix.reshape(linear_matrix.shape[0], -1), dim=1
            )
        elif hasattr(model, 'F_left') and hasattr(model, 'F_right'):
            linear_matrix = model.F_left @ model.F_right.mT
            positive_linear_rank = model.F_left.shape[-1]
            blocks['matrix_linear'] = torch.linalg.vector_norm(
                linear_matrix.reshape(linear_matrix.shape[0], -1), dim=1
            )
        if 'matrix_linear' in blocks:
            linear_singular_values = torch.linalg.svdvals(linear_matrix)
            matrix_linear_singular_conditions = (
                linear_singular_values[:, 0]
                / linear_singular_values[:, positive_linear_rank - 1]
            )

        if hasattr(model, 'kappa'):
            kappa = model.kappa
            if model.distribution in {'VonMisesFisher', 'FisherBingham_lowrank'}:
                kappa = torch.nn.functional.softplus(kappa)
            raw_kappa_norms = kappa.abs().reshape(kappa.shape[0], -1).norm(dim=1)
            if model.distribution in {'Watson', 'Complex_Watson'}:
                blocks['quadratic'] = raw_kappa_norms * np.sqrt(
                    1.0 - 1.0 / model.p
                )
            else:
                blocks['kappa'] = raw_kappa_norms

        if not blocks:
            return {
                'initial_natural_parameter_norm_median': float('nan'),
                'initial_natural_parameter_norm_max': float('nan'),
            }

        total_square = None
        for values in blocks.values():
            total_square = (
                values.square()
                if total_square is None
                else total_square + values.square()
            )
        totals = torch.sqrt(total_square)
        summary = {
            'initial_natural_parameter_norm_median': float(totals.median().cpu()),
            'initial_natural_parameter_norm_max': float(totals.max().cpu()),
        }
        for name, values in blocks.items():
            summary[f'initial_{name}_natural_norm_median'] = float(
                values.median().cpu()
            )
        if hasattr(model, 'M'):
            summary['initial_quadratic_raw_norm_median'] = float(
                raw_quadratic_norms.median().cpu()
            )
            summary[
                'initial_quadratic_positive_eigenvalue_condition_median'
            ] = float(quadratic_positive_eigenvalue_conditions.median().cpu())
        if 'matrix_linear' in blocks:
            summary['initial_matrix_linear_singular_condition_median'] = float(
                matrix_linear_singular_conditions.median().cpu()
            )
        if hasattr(model, 'kappa'):
            summary['initial_kappa_raw_norm_median'] = float(
                raw_kappa_norms.median().cpu()
            )
        return summary


def _synchronize(data):
    """Wait for pending CUDA work before recording a timing boundary."""
    if data.is_cuda:
        import torch

        torch.cuda.synchronize(data.device)
