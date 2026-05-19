from __future__ import annotations

import numpy as np
from scipy.special import eval_jacobi, gammaln


def _validate_alpha_beta(alpha: float, beta: float) -> None:
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("alpha and beta must both be > -1.")


def jacobi_norm_sq(n: int, alpha: float, beta: float) -> float:
    if n < 0:
        raise ValueError("n must be >= 0")

    _validate_alpha_beta(alpha, beta)

    log_num = (
        (alpha + beta + 1.0) * np.log(2.0)
        + gammaln(n + alpha + 1.0)
        + gammaln(n + beta + 1.0)
    )
    log_den = (
        np.log(2.0 * n + alpha + beta + 1.0)
        + gammaln(n + 1.0)
        + gammaln(n + alpha + beta + 1.0)
    )

    return float(np.exp(log_num - log_den))


def jacobi_classical(n: int, alpha: float, beta: float, x) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be >= 0")

    _validate_alpha_beta(alpha, beta)
    x = np.asarray(x, dtype=float)

    return np.asarray(eval_jacobi(n, alpha, beta, x), dtype=float)


def jacobi_orthonormal(n: int, alpha: float, beta: float, x) -> np.ndarray:
    p = jacobi_classical(n, alpha, beta, x)
    h = jacobi_norm_sq(n, alpha, beta)
    return p / np.sqrt(h)


def grad_jacobi_classical(n: int, alpha: float, beta: float, x) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be >= 0")

    _validate_alpha_beta(alpha, beta)
    x = np.asarray(x, dtype=float)

    if n == 0:
        return np.zeros_like(x, dtype=float)

    factor = 0.5 * (n + alpha + beta + 1.0)
    return factor * jacobi_classical(n - 1, alpha + 1.0, beta + 1.0, x)


def grad_jacobi_orthonormal(n: int, alpha: float, beta: float, x) -> np.ndarray:
    dp = grad_jacobi_classical(n, alpha, beta, x)
    h = jacobi_norm_sq(n, alpha, beta)
    return dp / np.sqrt(h)