from __future__ import annotations

import numpy as np


def normalize_vector(v: np.ndarray, radius: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    nrm = np.linalg.norm(v)

    if nrm <= 0.0:
        raise ValueError("Cannot normalize zero vector.")

    return radius * v / nrm


def rodrigues_rotate(
    v: np.ndarray,
    omega: np.ndarray | tuple[float, float, float],
    t: float,
) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    omega = np.asarray(omega, dtype=float).reshape(3)

    omega_norm = np.linalg.norm(omega)

    if omega_norm <= 0.0:
        return v.copy()

    axis = omega / omega_norm
    theta = omega_norm * float(t)

    c = np.cos(theta)
    s = np.sin(theta)

    return (
        c * v
        + s * np.cross(axis, v)
        + (1.0 - c) * np.dot(axis, v) * axis
    )


def sphere_geodesic_distance(
    X: np.ndarray,
    center: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    center = normalize_vector(center, radius=radius)

    dot = np.sum(X * center.reshape(1, 1, 3), axis=-1) / (radius * radius)
    dot = np.clip(dot, -1.0, 1.0)

    return radius * np.arccos(dot)


def gaussian_on_sphere(
    X: np.ndarray,
    center: np.ndarray | None = None,
    radius: float = 1.0,
    sigma: float = 0.35,
    amplitude: float = 1.0,
) -> np.ndarray:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if center is None:
        center = np.array([radius, 0.0, 0.0], dtype=float)

    X = np.asarray(X, dtype=float)
    d = sphere_geodesic_distance(X, center=center, radius=radius)

    return float(amplitude) * np.exp(-0.5 * (d / sigma) ** 2)


def gaussian_center_solid_body(
    t: float,
    radius: float = 1.0,
    center0: np.ndarray | tuple[float, float, float] | None = None,
    omega: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    if center0 is None:
        center0 = np.array([radius, 0.0, 0.0], dtype=float)

    c0 = normalize_vector(np.asarray(center0, dtype=float), radius=radius)
    return normalize_vector(rodrigues_rotate(c0, omega=omega, t=t), radius=radius)


def exact_gaussian_solid_body(
    X: np.ndarray,
    t: float,
    radius: float = 1.0,
    sigma: float = 0.35,
    amplitude: float = 1.0,
    center0: np.ndarray | tuple[float, float, float] | None = None,
    omega: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    center_t = gaussian_center_solid_body(
        t=t,
        radius=radius,
        center0=center0,
        omega=omega,
    )

    return gaussian_on_sphere(
        X=X,
        center=center_t,
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
    )