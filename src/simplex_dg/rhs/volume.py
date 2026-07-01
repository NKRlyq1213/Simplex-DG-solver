from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np

from simplex_dg.geometry import GeometryCache
from simplex_dg.reference import ReferenceCache


try:
    _numba = importlib.import_module("numba")
    njit = _numba.njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class VolumeRHSCache:
    n_elements: int
    n_points: int

    Dr: np.ndarray
    Ds: np.ndarray

    sqrt_g: np.ndarray

    velocity: np.ndarray
    speed: np.ndarray
    max_speed: float

    alpha: np.ndarray
    beta: np.ndarray
    Dr_alpha: np.ndarray
    Ds_beta: np.ndarray
    div_velocity: np.ndarray


def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


def apply_reference_operator(D: np.ndarray, u: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    u = np.asarray(u, dtype=float)

    if u.ndim == 1:
        return D @ u

    if u.ndim == 2:
        return u @ D.T

    raise ValueError("u must have shape (Np,) or (K, Np).")


def project_to_tangent(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    normal = np.asarray(normal, dtype=float)

    if v.shape != normal.shape:
        raise ValueError("v and normal must have the same shape.")

    vn = np.sum(v * normal, axis=-1, keepdims=True)

    return v - vn * normal


def solid_body_rotation_velocity(
    X: np.ndarray,
    omega: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    omega = np.asarray(omega, dtype=float).reshape(3)

    return np.cross(omega[None, None, :], X)


def build_volume_rhs_cache(
    ref: ReferenceCache,
    geom: GeometryCache,
    velocity: np.ndarray | None = None,
    omega: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    project_velocity: bool = True,
    validate: bool = True,
) -> VolumeRHSCache:
    X = np.asarray(geom.X, dtype=float)

    K, Np, dim = X.shape

    if dim != 3:
        raise ValueError("geom.X must have shape (K, Np, 3).")

    if velocity is None:
        u = solid_body_rotation_velocity(X, omega=omega)
    else:
        u = np.asarray(velocity, dtype=float)

    if u.shape != X.shape:
        raise ValueError("velocity must have shape (K, Np, 3).")

    if project_velocity:
        u = project_to_tangent(u, geom.normal)

    speed = np.linalg.norm(u, axis=2)
    max_speed = float(np.max(speed))

    sqrt_g = np.asarray(geom.sqrt_g, dtype=float)

    alpha = sqrt_g * np.sum(u * geom.grad_r, axis=2)
    beta = sqrt_g * np.sum(u * geom.grad_s, axis=2)

    Dr = np.asarray(ref.Dr, dtype=float)
    Ds = np.asarray(ref.Ds, dtype=float)

    Dr_alpha = apply_reference_operator(Dr, alpha)
    Ds_beta = apply_reference_operator(Ds, beta)

    div_velocity = (Dr_alpha + Ds_beta) / sqrt_g

    cache = VolumeRHSCache(
        n_elements=K,
        n_points=Np,
        Dr=Dr,
        Ds=Ds,
        sqrt_g=sqrt_g,
        velocity=u,
        speed=speed,
        max_speed=max_speed,
        alpha=alpha,
        beta=beta,
        Dr_alpha=Dr_alpha,
        Ds_beta=Ds_beta,
        div_velocity=div_velocity,
    )

    if validate:
        validate_volume_rhs_cache(cache, geom)

    return cache


def validate_volume_rhs_cache(
    cache: VolumeRHSCache,
    geom: GeometryCache,
    tol: float = 1e-10,
) -> None:
    K = cache.n_elements
    Np = cache.n_points

    if cache.Dr.shape != (Np, Np):
        raise ValueError("Dr must have shape (Np, Np).")

    if cache.Ds.shape != (Np, Np):
        raise ValueError("Ds must have shape (Np, Np).")

    for name in ("sqrt_g", "speed", "alpha", "beta", "Dr_alpha", "Ds_beta", "div_velocity"):
        arr = getattr(cache, name)
        if arr.shape != (K, Np):
            raise ValueError(f"{name} must have shape (K, Np).")

    if cache.velocity.shape != (K, Np, 3):
        raise ValueError("velocity must have shape (K, Np, 3).")

    if np.any(cache.sqrt_g <= 0.0):
        raise ValueError("sqrt_g must be positive.")

    tangent_error = np.max(np.abs(np.sum(cache.velocity * geom.normal, axis=2)))

    #if tangent_error > tol:
    #   raise ValueError(f"velocity is not tangent to the manifold: max error = {tangent_error}.")


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _volume_divergence_split_kernel(q, Dr, Ds, sqrt_g, alpha, beta, Dr_alpha, Ds_beta, out):
        K = q.shape[0]
        Np = q.shape[1]

        for k in range(K):
            for i in range(Np):
                qr = 0.0
                qs = 0.0
                Dr_alpha_q = 0.0
                Ds_beta_q = 0.0

                for j in range(Np):
                    dr = Dr[i, j]
                    ds = Ds[i, j]
                    qj = q[k, j]

                    qr += dr * qj
                    qs += ds * qj

                    Dr_alpha_q += dr * alpha[k, j] * qj
                    Ds_beta_q += ds * beta[k, j] * qj

                split_r = 0.5 * (
                    Dr_alpha_q
                    + alpha[k, i] * qr
                    + q[k, i] * Dr_alpha[k, i]
                )

                split_s = 0.5 * (
                    Ds_beta_q
                    + beta[k, i] * qs
                    + q[k, i] * Ds_beta[k, i]
                )

                out[k, i] = (split_r + split_s) / sqrt_g[k, i]


    @njit(cache=True)
    def _volume_divergence_conservative_kernel(q, Dr, Ds, sqrt_g, alpha, beta, out):
        K = q.shape[0]
        Np = q.shape[1]

        for k in range(K):
            for i in range(Np):
                Dr_alpha_q = 0.0
                Ds_beta_q = 0.0

                for j in range(Np):
                    Dr_alpha_q += Dr[i, j] * alpha[k, j] * q[k, j]
                    Ds_beta_q += Ds[i, j] * beta[k, j] * q[k, j]

                out[k, i] = (Dr_alpha_q + Ds_beta_q) / sqrt_g[k, i]

else:
    _volume_divergence_split_kernel = None
    _volume_divergence_conservative_kernel = None


def volume_divergence_split(
    q: np.ndarray,
    cache: VolumeRHSCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    q = np.asarray(q, dtype=float)

    expected = (cache.n_elements, cache.n_points)

    if q.shape != expected:
        raise ValueError(f"q must have shape {expected}.")

    if out is None:
        div = np.empty_like(q)
    else:
        div = np.asarray(out, dtype=float)
        if div.shape != expected:
            raise ValueError("out has wrong shape.")

    if _should_use_numba(use_numba):
        _volume_divergence_split_kernel(
            q,
            cache.Dr,
            cache.Ds,
            cache.sqrt_g,
            cache.alpha,
            cache.beta,
            cache.Dr_alpha,
            cache.Ds_beta,
            div,
        )
        return div

    qr = apply_reference_operator(cache.Dr, q)
    qs = apply_reference_operator(cache.Ds, q)

    Dr_alpha_q = apply_reference_operator(cache.Dr, cache.alpha * q)
    Ds_beta_q = apply_reference_operator(cache.Ds, cache.beta * q)

    split_r = 0.5 * (
        Dr_alpha_q
        + cache.alpha * qr
        + q * cache.Dr_alpha
    )

    split_s = 0.5 * (
        Ds_beta_q
        + cache.beta * qs
        + q * cache.Ds_beta
    )

    div[:, :] = (split_r + split_s) / cache.sqrt_g

    return div


def volume_divergence_conservative(
    q: np.ndarray,
    cache: VolumeRHSCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    q = np.asarray(q, dtype=float)

    expected = (cache.n_elements, cache.n_points)

    if q.shape != expected:
        raise ValueError(f"q must have shape {expected}.")

    if out is None:
        div = np.empty_like(q)
    else:
        div = np.asarray(out, dtype=float)
        if div.shape != expected:
            raise ValueError("out has wrong shape.")

    if _should_use_numba(use_numba):
        _volume_divergence_conservative_kernel(
            q,
            cache.Dr,
            cache.Ds,
            cache.sqrt_g,
            cache.alpha,
            cache.beta,
            div,
        )
        return div

    Dr_alpha_q = apply_reference_operator(cache.Dr, cache.alpha * q)
    Ds_beta_q = apply_reference_operator(cache.Ds, cache.beta * q)

    div[:, :] = (Dr_alpha_q + Ds_beta_q) / cache.sqrt_g

    return div


def volume_rhs_split(
    q: np.ndarray,
    cache: VolumeRHSCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    div = volume_divergence_split(q, cache, out=out, use_numba=use_numba)
    div *= -1.0
    return div


def volume_rhs_conservative(
    q: np.ndarray,
    cache: VolumeRHSCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    div = volume_divergence_conservative(q, cache, out=out, use_numba=use_numba)
    div *= -1.0
    return div