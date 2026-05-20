from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplex_dg.geometry import GeometryCache
from simplex_dg.reference import ReferenceCache
from simplex_dg.time import manifold_integral, manifold_l2_norm


@dataclass(frozen=True)
class ErrorReport:
    l2_error: float
    relative_l2_error: float
    linf_error: float
    mass_error: float
    l2_norm_numerical: float
    l2_norm_exact: float


def l2_error(
    q_num: np.ndarray,
    q_exact: np.ndarray,
    ref: ReferenceCache,
    geom: GeometryCache,
) -> float:
    q_num = np.asarray(q_num, dtype=float)
    q_exact = np.asarray(q_exact, dtype=float)

    if q_num.shape != q_exact.shape:
        raise ValueError("q_num and q_exact must have the same shape.")

    return manifold_l2_norm(q_num - q_exact, ref, geom)


def relative_l2_error(
    q_num: np.ndarray,
    q_exact: np.ndarray,
    ref: ReferenceCache,
    geom: GeometryCache,
) -> float:
    denom = manifold_l2_norm(q_exact, ref, geom)

    if denom <= 0.0:
        return np.inf

    return l2_error(q_num, q_exact, ref, geom) / denom


def linf_error(
    q_num: np.ndarray,
    q_exact: np.ndarray,
) -> float:
    q_num = np.asarray(q_num, dtype=float)
    q_exact = np.asarray(q_exact, dtype=float)

    if q_num.shape != q_exact.shape:
        raise ValueError("q_num and q_exact must have the same shape.")

    return float(np.max(np.abs(q_num - q_exact)))


def error_report(
    q_num: np.ndarray,
    q_exact: np.ndarray,
    ref: ReferenceCache,
    geom: GeometryCache,
) -> ErrorReport:
    q_num = np.asarray(q_num, dtype=float)
    q_exact = np.asarray(q_exact, dtype=float)

    l2e = l2_error(q_num, q_exact, ref, geom)
    l2_exact = manifold_l2_norm(q_exact, ref, geom)

    if l2_exact > 0.0:
        rel = l2e / l2_exact
    else:
        rel = np.inf

    mass_num = manifold_integral(q_num, ref, geom)
    mass_exact = manifold_integral(q_exact, ref, geom)

    return ErrorReport(
        l2_error=float(l2e),
        relative_l2_error=float(rel),
        linf_error=linf_error(q_num, q_exact),
        mass_error=float(mass_num - mass_exact),
        l2_norm_numerical=manifold_l2_norm(q_num, ref, geom),
        l2_norm_exact=float(l2_exact),
    )