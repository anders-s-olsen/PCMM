import time
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from PCMM.PCMMnumpyBaseModel import init_M_svd_given_M_init
from PCMM.mixture_loop_utils import _convergence_scale


def mixture_EM_loop(model, data, tol=1e-8, max_iter=10000, num_repl=1, init=None, suppress_output=False, num_comparison=10,
    convergence_normalization=None, intrinsic_dimension=None, return_diagnostics=False, timing_warmup_iterations=5):
    """Fit an EM model, retaining its summed log-likelihood objective."""
    if max_iter < 1:
        raise ValueError('max_iter must be positive.')
    if num_comparison < 1:
        raise ValueError('num_comparison must be positive.')
    if tol < 0:
        raise ValueError('tol must be non-negative.')
    if timing_warmup_iterations < 0:
        raise ValueError('timing_warmup_iterations must be non-negative.')

    convergence_scale = _convergence_scale(data, convergence_normalization, intrinsic_dimension)
    best_loglik = -np.inf
    if 'Complex' in model.distribution:
        if not np.iscomplexobj(data):
            raise ValueError('Data must be complex for complex models')
    elif np.iscomplexobj(data):
        raise ValueError('Data must be real for real models')

    if init == 'no' and 'pi' not in model.__dict__:
        raise ValueError('Model not initialized, please provide an initialization method or a set of parameters')

    for repl in range(num_repl):
        replication_start = time.perf_counter()
        initialization_start = time.perf_counter()
        if init != 'no':
            model.initialize(X=data, init_method=init, tol=tol)

        if 'lowrank' in model.distribution and model.M.shape[-1] != model.r:
            model2 = deepcopy(model)
            model2.r = model2.M.shape[-1]
            beta = model2.posterior(X=data)
            if model.distribution in ['ACG_lowrank', 'Complex_ACG_lowrank', 'MACG_lowrank']:
                model.M = init_M_svd_given_M_init(X=data, K=model.K, r=model.r, M_init=model2.M, beta=beta, gamma=None, distribution=model.distribution)
            elif model.distribution in ['SingularWishart_lowrank', 'Normal_lowrank', 'Complex_Normal_lowrank']:
                model.M = init_M_svd_given_M_init(X=data, K=model.K, r=model.r, M_init=model2.M, beta=beta, gamma=model2.gamma, distribution=model.distribution)
        initialization_seconds = time.perf_counter() - initialization_start

        loglik = []
        iteration_seconds = []
        best_epoch_loglik = -np.inf
        best_convergence_score = -np.inf
        last_significant_improvement = 0
        stopping_reason = 'max_iter'
        if not suppress_output:
            tqdm.write(f'Beginning EM loop (replication {repl + 1}/{num_repl})')
        pbar = tqdm(total=max_iter, disable=suppress_output)
        pbar.set_description('In the initial phase')
        pbar.update(0)

        for epoch in range(max_iter):
            iteration_start = time.perf_counter()
            epoch_loglik = model.log_likelihood(X=data)
            loglik.append(epoch_loglik)
            if not np.isfinite(epoch_loglik):
                raise ValueError('The objective became non-finite. Check the initialization, data, and model.')

            if epoch_loglik > best_epoch_loglik:
                best_model_params = deepcopy(model.get_params())
                best_epoch_loglik = epoch_loglik

            convergence_score = epoch_loglik / convergence_scale
            if convergence_score > best_convergence_score + tol:
                best_convergence_score = convergence_score
                last_significant_improvement = epoch

            # Complete the EM update before recording the iteration duration.
            model.M_step(X=data)
            iteration_seconds.append(time.perf_counter() - iteration_start)

            epochs_without_improvement = epoch - last_significant_improvement
            pbar.set_description('Loglik: %.2f, epochs without improvement: %d' % (epoch_loglik, epochs_without_improvement))
            pbar.update(1)
            if epochs_without_improvement >= num_comparison:
                stopping_reason = 'no_significant_improvement'
                break
        pbar.close()
        replication_seconds = time.perf_counter() - replication_start

        warmup = min(timing_warmup_iterations, max(len(iteration_seconds) - 1, 0))
        timed_iterations = iteration_seconds[warmup:]
        repl_diagnostics = {
            'initialization_seconds': initialization_seconds,
            'optimization_seconds': float(np.sum(iteration_seconds)),
            'total_seconds': replication_seconds,
            'unattributed_overhead_seconds': max(replication_seconds - initialization_seconds - float(np.sum(iteration_seconds)), 0.0,),
            'iteration_seconds': iteration_seconds,
            'median_iteration_seconds': float(np.median(timed_iterations)),
            'timing_warmup_iterations': warmup,
            'iterations': len(loglik),
            'stopping_reason': stopping_reason,
            'convergence_normalization': convergence_normalization or 'none',
            'convergence_scale': convergence_scale,
            'intrinsic_dimension': intrinsic_dimension,
        }

        if best_epoch_loglik > best_loglik:
            best_loglik = best_epoch_loglik
            params_final = best_model_params
            loglik_final = loglik
            diagnostics_final = repl_diagnostics

    model.set_params(params_final)
    beta_final = model.posterior(X=data)

    result = (params_final, beta_final, loglik_final)
    if return_diagnostics:
        return (*result, diagnostics_final)
    return result
