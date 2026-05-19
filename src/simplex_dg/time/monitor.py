from __future__ import annotations

import numpy as np

from simplex_dg.geometry import GeometryCache
from simplex_dg.reference import ReferenceCache
from simplex_dg.reference.quadrature import REFERENCE_AREA


def manifold_integral(
    q: np.ndarray,
    ref: ReferenceCache,
    geom: GeometryCache,
) -> float:
    q = np.asarray(q, dtype=float)

    expected = geom.sqrt_g.shape

    if q.shape != expected:
        raise ValueError(f"q must have shape {expected}.")

    weighted = q * geom.sqrt_g

    return float(REFERENCE_AREA * np.sum(weighted * ref.weights[None, :]))


def manifold_l2_norm(
    q: np.ndarray,
    ref: ReferenceCache,
    geom: GeometryCache,
) -> float:
    q = np.asarray(q, dtype=float)

    val = manifold_integral(q * q, ref, geom)

    return float(np.sqrt(max(val, 0.0)))


def mass_history_entry(
    t: float,
    q: np.ndarray,
    ref: ReferenceCache,
    geom: GeometryCache,
) -> dict[str, float]:
    return {
        "t": float(t),
        "mass": manifold_integral(q, ref, geom),
        "l2": manifold_l2_norm(q, ref, geom),
        "linf": float(np.max(np.abs(q))),
    }