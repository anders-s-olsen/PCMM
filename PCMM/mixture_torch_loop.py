import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from PCMM.PCMMnumpyBaseModel import init_M_svd_given_M_init
from PCMM.mixture_loop_utils import (
    _convergence_scale,
    _initial_natural_parameter_norms,
    _rescale_isotropic_parameters,
    _synchronize,
)


def mixture_torch_loop(model, data, tol=1e-8, max_iter=100000, num_repl=1, init=None, LR=0.1, suppress_output=False, threads=8,
    decrease_lr_on_plateau=False, num_comparison=50, convergence_normalization=None, intrinsic_dimension=None, return_diagnostics=False,
    timing_warmup_iterations=5, initialization_tol=None, max_optimization_seconds=None, max_learning_rate_reductions=0,
    learning_rate_reduction_factor=0.1, isotropic_natural_scale=None, isotropic_wrapped_variance=None):
    """Fit a torch model and restore the best observed parameter values.

    The model objective remains the summed log likelihood.  Optionally, only
    the improvement used for early stopping is expressed per observation and
    per intrinsic manifold dimension.  The default therefore retains the
    historical, unnormalised scale for existing callers.
    """
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    best_loglik = -np.inf

    if max_iter < 1:
        raise ValueError('max_iter must be positive.')
    if num_comparison < 1:
        raise ValueError('num_comparison must be positive.')
    if tol < 0:
        raise ValueError('tol must be non-negative.')
    if timing_warmup_iterations < 0:
        raise ValueError('timing_warmup_iterations must be non-negative.')
    if initialization_tol is not None and initialization_tol <= 0:
        raise ValueError('initialization_tol must be positive when provided.')
    if max_optimization_seconds is not None and max_optimization_seconds <= 0:
        raise ValueError('max_optimization_seconds must be positive when provided.')
    if max_learning_rate_reductions < 0:
        raise ValueError('max_learning_rate_reductions must be non-negative.')
    if not 0 < learning_rate_reduction_factor < 1:
        raise ValueError('learning_rate_reduction_factor must lie strictly between zero and one.')
    if isotropic_natural_scale is not None and isotropic_natural_scale <= 0:
        raise ValueError('isotropic_natural_scale must be positive when provided.')
    if isotropic_wrapped_variance is not None and isotropic_wrapped_variance <= 0:
        raise ValueError('isotropic_wrapped_variance must be positive when provided.')

    if 'lowrank' in model.distribution:
        assert model.r is not None, 'Model rank must be set'
        assert model.r != 0, 'Model rank must be non-zero'

    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)

    convergence_scale = _convergence_scale(data, convergence_normalization, intrinsic_dimension)

    if 'Complex' in model.distribution:
        if not data.is_complex():
            raise ValueError('Data must be complex for complex models')
    elif data.is_complex():
        raise ValueError('Data must be real for real models')

    param_names = [name for name, _ in model.named_parameters()]
    if init == 'no' and 'pi' not in param_names:
        raise ValueError('Model not initialized, please provide an initialization method or a set of parameters')

    for repl in range(num_repl):
        learning_rate_reductions = 0
        allowed_learning_rate_reductions = max(int(max_learning_rate_reductions), 1 if decrease_lr_on_plateau else 0)
        _synchronize(data)
        replication_start = time.perf_counter()
        initialization_start = time.perf_counter()
        if init != 'no':
            # Replications are independent and receive fresh parameters.
            model.initialize(X=data, init_method=init, tol=tol if initialization_tol is None else initialization_tol)
            if init == 'isotropic':
                _rescale_isotropic_parameters(model, isotropic_natural_scale, wrapped_variance=isotropic_wrapped_variance)
        model.to(device=data.device)
        param_names = [name for name, _ in model.named_parameters()]

        if 'lowrank' in model.distribution and model.M.shape[-1] != model.r:
            model2 = deepcopy(model)
            model2.r = model2.M.shape[-1]
            beta = model2.posterior(X=data)
            if model.distribution in ['Bingham_lowrank', 'Complex_Bingham_lowrank', 'ACG_lowrank', 'Complex_ACG_lowrank', 'MACG_lowrank',]:
                gamma = None
            elif model.distribution in ['SingularWishart_lowrank', 'Normal_lowrank', 'Complex_Normal_lowrank', 'WrappedNormal_lowrank',]:
                gamma = torch.nn.functional.softplus(model.gamma).detach().cpu().numpy()
            M = init_M_svd_given_M_init(X=data.detach().cpu().numpy(), K=model.K, r=model.r,
                                        M_init=model.M.detach().cpu().numpy(), beta=beta, gamma=gamma, distribution=model.distribution)
            model.M = torch.nn.Parameter(torch.as_tensor(M, dtype=data.dtype, device=data.device))

        initial_natural_parameter_norms = _initial_natural_parameter_norms(model)

        if model.HMM:
            # Do not retain the transition matrix from an earlier replication.
            if init != 'no' or 'T' not in param_names:
                if init in ['unif', 'uniform', 'isotropic']:
                    model.T = torch.nn.Parameter(torch.zeros((model.K, model.K), dtype=data.real.dtype, device=data.device))
                else:
                    T, delta = model.initialize_transition_matrix_hmm(X=data)
                    model.T = torch.nn.Parameter(model._probabilities_to_logits(T, dim=1))
                    model.pi = torch.nn.Parameter(model._probabilities_to_logits(delta, dim=0))
        _synchronize(data)
        initialization_seconds = time.perf_counter() - initialization_start

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        optimization_start = time.perf_counter()
        loglik = []
        iteration_seconds = []
        iteration_cpu_seconds = []
        objective_forward_seconds = []
        objective_backward_seconds = []
        parameter_snapshot_seconds = []
        optimizer_step_seconds = []
        best_epoch_loglik = -np.inf
        best_convergence_score = -np.inf
        last_significant_improvement = 0
        stopping_reason = 'max_iter'
        done = False
        if not suppress_output:
            tqdm.write(f'Beginning numerical optimization loop ' f'(replication {repl + 1}/{num_repl})')

        pbar = tqdm(total=max_iter, disable=suppress_output)
        pbar.set_description('In the initial phase')
        pbar.update(0)

        for epoch in range(max_iter):
            _synchronize(data)
            iteration_start = time.perf_counter()
            iteration_cpu_start = time.process_time()
            forward_start = time.perf_counter()
            epoch_nll = -model(data)
            _synchronize(data)
            objective_forward_seconds.append(time.perf_counter() - forward_start)

            if not torch.isfinite(epoch_nll):
                raise ValueError('The objective became non-finite.')

            optimizer.zero_grad(set_to_none=True)
            backward_start = time.perf_counter()
            epoch_nll.backward()
            _synchronize(data)
            objective_backward_seconds.append(time.perf_counter() - backward_start)
            epoch_loglik = -epoch_nll.item()
            loglik.append(epoch_loglik)

            snapshot_start = time.perf_counter()
            with torch.no_grad():
                if epoch_loglik > best_epoch_loglik:
                    best_model_params = deepcopy(model.get_params())
                    best_epoch_loglik = epoch_loglik
            parameter_snapshot_seconds.append(time.perf_counter() - snapshot_start)

            optimizer_step_start = time.perf_counter()
            optimizer.step()
            _synchronize(data)
            optimizer_step_seconds.append(time.perf_counter() - optimizer_step_start)
            iteration_seconds.append(time.perf_counter() - iteration_start)
            iteration_cpu_seconds.append(time.process_time() - iteration_cpu_start)

            convergence_score = epoch_loglik / convergence_scale
            diagnostic_gain = convergence_score - best_convergence_score
            if diagnostic_gain > tol:
                best_convergence_score = convergence_score
                last_significant_improvement = epoch

            epochs_without_improvement = epoch - last_significant_improvement
            pbar.set_description('Loglik: %.2f, diagnostic gain: %.2e' % (epoch_loglik, max(diagnostic_gain, 0.0)))
            pbar.update(1)

            if max_optimization_seconds is not None and time.perf_counter() - optimization_start >= max_optimization_seconds:
                stopping_reason = 'max_optimization_seconds'
                done = True
            elif epochs_without_improvement >= num_comparison:
                if learning_rate_reductions < allowed_learning_rate_reductions:
                    new_learning_rate = (optimizer.param_groups[0]['lr'] * learning_rate_reduction_factor)
                    # Restart from the best observed state.  Continuing from an
                    # oscillating plateau with stale Adam moments can make a
                    # smaller learning rate look converged without refining the
                    # best solution.
                    model.set_params(best_model_params)
                    model.to(device=data.device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=new_learning_rate)
                    if not suppress_output:
                        tqdm.write('Learning rate reduced to: %s after %d iterations ' '(restart from best state)' % (new_learning_rate, epoch + 1))
                    learning_rate_reductions += 1
                    best_convergence_score = best_epoch_loglik / convergence_scale
                    last_significant_improvement = epoch
                else:
                    stopping_reason = 'objective_plateau'
                    done = True
            if done:
                break
        pbar.close()

        # ``epoch_loglik`` is evaluated immediately before ``optimizer.step``.
        # Evaluate the state produced by the final step once so that a fit
        # stopped by patience, iteration count, or the soft wall-time budget
        # cannot silently discard a beneficial last update.  This is a fitting
        # finalization evaluation, not another optimizer iteration.
        _synchronize(data)
        post_update_evaluation_start = time.perf_counter()
        with torch.no_grad():
            post_update_loglik = float(model(data).detach().cpu())
            if np.isfinite(post_update_loglik) and post_update_loglik > best_epoch_loglik:
                best_model_params = deepcopy(model.get_params())
                best_epoch_loglik = post_update_loglik
        _synchronize(data)
        post_update_evaluation_seconds = (time.perf_counter() - post_update_evaluation_start)

        _synchronize(data)
        replication_seconds = time.perf_counter() - replication_start
        gradient_squares = [parameter.grad.detach().abs().square().sum() for parameter in model.parameters() if parameter.grad is not None]
        last_iteration_gradient_norm = (float(torch.sqrt(torch.stack(gradient_squares).sum()).detach().cpu()) if gradient_squares else float('nan'))

        warmup = min(timing_warmup_iterations, max(len(iteration_seconds) - 1, 0))
        timed_iterations = iteration_seconds[warmup:]
        timed_iteration_cpu = iteration_cpu_seconds[warmup:]
        timed_forward = objective_forward_seconds[warmup:]
        timed_backward = objective_backward_seconds[warmup:]
        timed_snapshot = parameter_snapshot_seconds[warmup:]
        timed_optimizer_step = optimizer_step_seconds[warmup:]
        repl_diagnostics = {
            'initialization_seconds': initialization_seconds,
            'optimization_seconds': float(np.sum(iteration_seconds)),
            'post_update_evaluation_seconds': post_update_evaluation_seconds,
            'post_update_log_likelihood': post_update_loglik,
            'best_log_likelihood': best_epoch_loglik,
            'total_seconds': replication_seconds,
            'unattributed_overhead_seconds': max(
                replication_seconds
                - initialization_seconds
                - float(np.sum(iteration_seconds))
                - post_update_evaluation_seconds,
                0.0,
            ),
            'iteration_seconds': iteration_seconds,
            'median_iteration_seconds': float(np.median(timed_iterations)),
            'median_iteration_cpu_seconds': float(np.median(timed_iteration_cpu)),
            'median_objective_forward_seconds': float(np.median(timed_forward)),
            'median_objective_backward_seconds': float(np.median(timed_backward)),
            'median_parameter_snapshot_seconds': float(np.median(timed_snapshot)),
            'median_optimizer_step_seconds': float(np.median(timed_optimizer_step)),
            'timing_warmup_iterations': warmup,
            'iterations': len(loglik),
            'stopping_reason': stopping_reason,
            # This is an operational completion rule, not a proof that the
            # restored parameters satisfy a stationarity condition.  Callers
            # can evaluate the final gradient separately as quality control.
            'converged': stopping_reason == 'objective_plateau',
            'learning_rate_reductions': learning_rate_reductions,
            'final_learning_rate': optimizer.param_groups[0]['lr'],
            'isotropic_natural_scale': isotropic_natural_scale,
            'isotropic_wrapped_variance': isotropic_wrapped_variance,
            'convergence_normalization': convergence_normalization or 'none',
            'convergence_scale': convergence_scale,
            'intrinsic_dimension': intrinsic_dimension,
            'max_optimization_seconds': max_optimization_seconds,
            'last_iteration_gradient_norm': last_iteration_gradient_norm,
            **initial_natural_parameter_norms,
        }

        if best_epoch_loglik > best_loglik:
            best_loglik = best_epoch_loglik
            params_final = best_model_params
            loglik_final = loglik
            diagnostics_final = repl_diagnostics

    _synchronize(data)
    finalization_start = time.perf_counter()
    with torch.no_grad():
        model.set_params(params_final)
        beta_final = model.posterior(X=data)
    _synchronize(data)
    diagnostics_final['finalization_seconds'] = (time.perf_counter() - finalization_start)

    result = (params_final, beta_final, loglik_final)
    if return_diagnostics:
        return (*result, diagnostics_final)
    return result
