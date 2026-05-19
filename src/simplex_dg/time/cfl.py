from __future__ import annotations

import numpy as np

from simplex_dg.geometry import GeometryCache
from simplex_dg.reference import ReferenceCache


def face_lengths_from_geometry(
    ref: ReferenceCache,
    geom: GeometryCache,
) -> np.ndarray:
    K = geom.face_jacobian.shape[0]
    out = np.zeros((K, 3), dtype=float)

    for face_id in (1, 2, 3):
        f = face_id - 1
        w = ref.edge_rules[face_id].weights
        out[:, f] = geom.face_jacobian[:, f, :] @ w

    return out


def minimum_face_length(
    ref: ReferenceCache,
    geom: GeometryCache,
) -> float:
    lengths = face_lengths_from_geometry(ref, geom)
    return float(np.min(lengths))


def cfl_dt(
    cfl: float,
    h: float,
    order: int,
    max_speed: float,
    *,
    power: int = 2,
) -> float:
    if cfl <= 0.0:
        raise ValueError("cfl must be positive.")

    if h <= 0.0:
        raise ValueError("h must be positive.")

    if order <= 0:
        raise ValueError("order must be positive.")

    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive.")

    if power <= 0:
        raise ValueError("power must be positive.")

    return float(cfl * h / ((order**power) * max_speed))


def cfl_dt_from_geometry(
    ref: ReferenceCache,
    geom: GeometryCache,
    max_speed: float,
    cfl: float = 0.25,
    *,
    power: int = 2,
) -> float:
    hmin = minimum_face_length(ref, geom)

    return cfl_dt(
        cfl=cfl,
        h=hmin,
        order=max(ref.order, 1),
        max_speed=max_speed,
        power=power,
    )