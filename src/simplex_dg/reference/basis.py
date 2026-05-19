from __future__ import annotations

import numpy as np

from simplex_dg.reference.indexing import mode_indices_2d, num_modes_2d
from simplex_dg.reference.jacobi import jacobi_orthonormal, grad_jacobi_orthonormal


def rstoab(r, s) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    if r.shape != s.shape:
        raise ValueError("r and s must have the same shape.")

    a = np.empty_like(r, dtype=float)
    mask = np.abs(1.0 - s) > 1e-14

    a[mask] = 2.0 * (1.0 + r[mask]) / (1.0 - s[mask]) - 1.0
    a[~mask] = -1.0

    b = s.copy()

    return a, b


def simplex2d_mode(i: int, j: int, r, s) -> np.ndarray:
    if i < 0 or j < 0:
        raise ValueError("i and j must be >= 0")

    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    a, b = rstoab(r, s)

    fa = jacobi_orthonormal(i, 0.0, 0.0, a)
    gb = jacobi_orthonormal(j, 2.0 * i + 1.0, 0.0, b)

    return np.sqrt(2.0) * fa * gb * (1.0 - b) ** i


def grad_simplex2d_mode(i: int, j: int, r, s) -> tuple[np.ndarray, np.ndarray]:
    if i < 0 or j < 0:
        raise ValueError("i and j must be >= 0")

    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    a, b = rstoab(r, s)

    fa = jacobi_orthonormal(i, 0.0, 0.0, a)
    dfa = grad_jacobi_orthonormal(i, 0.0, 0.0, a)

    gb = jacobi_orthonormal(j, 2.0 * i + 1.0, 0.0, b)
    dgb = grad_jacobi_orthonormal(j, 2.0 * i + 1.0, 0.0, b)

    one_minus_b = 1.0 - b

    if i == 0:
        h = np.ones_like(b, dtype=float)
        dh_db = np.zeros_like(b, dtype=float)
    else:
        h = one_minus_b**i
        dh_db = -i * one_minus_b ** (i - 1)

    eps = 1e-14
    denom = np.where(np.abs(one_minus_b) > eps, one_minus_b, eps)

    da_dr = 2.0 / denom
    da_ds = (1.0 + a) / denom

    pref = np.sqrt(2.0)

    dpsi_da = pref * dfa * gb * h
    dpsi_db = pref * fa * (dgb * h + gb * dh_db)

    dpsi_dr = dpsi_da * da_dr
    dpsi_ds = dpsi_da * da_ds + dpsi_db

    return dpsi_dr, dpsi_ds


def vandermonde2d(order: int, r, s) -> np.ndarray:
    if order < 0:
        raise ValueError("order must be >= 0")

    r = np.asarray(r, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)

    if r.shape != s.shape:
        raise ValueError("r and s must have the same shape.")

    V = np.zeros((r.size, num_modes_2d(order)), dtype=float)

    for k, (i, j) in enumerate(mode_indices_2d(order)):
        V[:, k] = simplex2d_mode(i, j, r, s)

    return V


def grad_vandermonde2d(order: int, r, s) -> tuple[np.ndarray, np.ndarray]:
    if order < 0:
        raise ValueError("order must be >= 0")

    r = np.asarray(r, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)

    if r.shape != s.shape:
        raise ValueError("r and s must have the same shape.")

    nmodes = num_modes_2d(order)
    Vr = np.zeros((r.size, nmodes), dtype=float)
    Vs = np.zeros((r.size, nmodes), dtype=float)

    for k, (i, j) in enumerate(mode_indices_2d(order)):
        dr, ds = grad_simplex2d_mode(i, j, r, s)
        Vr[:, k] = dr
        Vs[:, k] = ds

    return Vr, Vs