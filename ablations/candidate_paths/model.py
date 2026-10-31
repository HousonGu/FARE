"""Candidate-path controls: learned B+S, learned B+F, and constant alpha and beta."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from io_utils import VARIANTS

OPTIONS = dict(maxiter=250, ftol=1e-9, gtol=1e-6)
PENALTY = .001


def parameter_count(variant, k):
    if variant not in VARIANTS:
        raise ValueError('Existing/unrequested variant cannot be trained: ' + variant)
    return 2 if variant == 'both_constant' else k


def predict(theta, x, parts, variant):
    parameter_count(variant, x.shape[1])
    b, s, f = parts.T
    if variant == 'both_constant':
        beta = np.full(len(parts), expit(theta[0])); alpha = np.full(len(parts), expit(theta[1]))
    else:
        beta = expit(x @ theta)
        alpha = np.full(len(parts), 0. if variant == 'bs_path' else 1.)
    reference = s + alpha * (f - s)
    return np.maximum(b + beta * (reference - b), 0.), alpha, beta


def objective(theta, x, parts, y, variant, batch_rows=65536):
    """Same smooth-AE/sum(y) and non-intercept L2; constant logits unpenalized."""
    parameter_count(variant, x.shape[1])
    denominator = float(np.sum(y))
    if denominator <= 0 or not np.isfinite(y).all():
        raise ValueError('Invalid development labels')
    value = 0.; gradient = np.zeros_like(theta)
    for start in range(0, len(y), batch_rows):
        stop = min(start + batch_rows, len(y)); xx = x[start:stop]; pp = parts[start:stop]
        pred, alpha, beta = predict(theta, xx, pp, variant)
        b, s, f = pp.T; ref = s + alpha * (f - s)
        residual = pred - y[start:stop]; smooth = np.sqrt(residual ** 2 + 1.)
        g = residual / smooth / denominator
        value += float(smooth.sum()) / denominator
        gb = g * beta * (1 - beta) * (ref - b)
        if variant == 'both_constant':
            gradient += [gb.sum(), (g * beta * alpha * (1 - alpha) * (f - s)).sum()]
        else:
            gradient += xx.T @ gb
    if variant != 'both_constant':
        value += PENALTY * np.square(theta[1:]).sum()
        gradient[1:] += 2 * PENALTY * theta[1:]
    return float(value), gradient


def fit(x, parts, y, variant, threads=4):
    count = parameter_count(variant, x.shape[1])
    with threadpool_limits(limits=threads):
        result = minimize(objective, np.zeros(count), args=(x, parts, y, variant),
                          jac=True, method='L-BFGS-B', options=OPTIONS)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'{variant}: optimization failed: {result.message}')
    return result.x, dict(variant=variant, parameters=result.x.tolist(), active_parameters=count,
                         objective=float(result.fun), iterations=int(result.nit), converged=True,
                         initialization='all zero logits', penalty=PENALTY, options=OPTIONS,
                         control='all original gate features retained; only forecast output path constrained')
