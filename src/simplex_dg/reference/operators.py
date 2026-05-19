from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplex_dg.backends import BackendStatus, backend_status
from simplex_dg.reference.basis import vandermonde2d, grad_vandermonde2d
from simplex_dg.reference.quadrature import (
    REFERENCE_AREA,
    EdgeRule,
    TriangleRule,
    edge_gl_rule,
    load_triangle_rule,
)


@dataclass(frozen=True)
class ReferenceCache:
    order: int
    table: str
    area: float

    rule: TriangleRule
    rs: np.ndarray
    weights: np.ndarray

    V: np.ndarray
    Vr: np.ndarray
    Vs: np.ndarray

    M: np.ndarray
    Minv: np.ndarray
    projection: np.ndarray

    Dr: np.ndarray
    Ds: np.ndarray

    edge_rules: dict[int, EdgeRule]
    face_interp: dict[int, np.ndarray]

    backend: BackendStatus


def mass_matrix_from_quadrature(
    V: np.ndarray,
    weights: np.ndarray,
    area: float = REFERENCE_AREA,
) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if V.ndim != 2:
        raise ValueError("V must be 2D.")

    if V.shape[0] != weights.size:
        raise ValueError("V.shape[0] must match weights.size.")

    return area * (V.T @ (weights[:, None] * V))


def weighted_projection_matrix(
    V: np.ndarray,
    weights: np.ndarray,
    area: float = REFERENCE_AREA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = mass_matrix_from_quadrature(V, weights, area=area)
    rhs = area * (V.T * weights[None, :])

    projection = np.linalg.solve(M, rhs)
    Minv = np.linalg.inv(M)

    return M, Minv, projection


def differentiation_matrices_weighted(
    V: np.ndarray,
    Vr: np.ndarray,
    Vs: np.ndarray,
    weights: np.ndarray,
    area: float = REFERENCE_AREA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M, Minv, projection = weighted_projection_matrix(V, weights, area=area)

    Dr = Vr @ projection
    Ds = Vs @ projection

    return Dr, Ds, M, Minv, projection


def build_reference_cache(
    order: int,
    table: str = "table1",
    n_face: int | None = None,
    area: float = REFERENCE_AREA,
    validate: bool = True,
) -> ReferenceCache:
    rule = load_triangle_rule(table=table, order=order)

    rs = rule.rs
    weights = rule.weights

    V = vandermonde2d(order, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(order, rs[:, 0], rs[:, 1])

    Dr, Ds, M, Minv, projection = differentiation_matrices_weighted(
        V=V,
        Vr=Vr,
        Vs=Vs,
        weights=weights,
        area=area,
    )

    if n_face is None:
        n_face = order + 1

    edge_rules: dict[int, EdgeRule] = {}
    face_interp: dict[int, np.ndarray] = {}

    for face_id in (1, 2, 3):
        edge = edge_gl_rule(face_id, n_face)
        V_face = vandermonde2d(order, edge.rs[:, 0], edge.rs[:, 1])
        E_face = V_face @ projection

        edge_rules[face_id] = edge
        face_interp[face_id] = E_face

    cache = ReferenceCache(
        order=order,
        table=rule.table,
        area=area,
        rule=rule,
        rs=rs,
        weights=weights,
        V=V,
        Vr=Vr,
        Vs=Vs,
        M=M,
        Minv=Minv,
        projection=projection,
        Dr=Dr,
        Ds=Ds,
        edge_rules=edge_rules,
        face_interp=face_interp,
        backend=backend_status(),
    )

    if validate:
        validate_reference_cache(cache)

    return cache


def validate_reference_cache(cache: ReferenceCache, tol: float = 1e-10) -> None:
    n_points = cache.rs.shape[0]

    if cache.rs.ndim != 2 or cache.rs.shape[1] != 2:
        raise ValueError("cache.rs must have shape (Np, 2).")

    if cache.weights.shape != (n_points,):
        raise ValueError("cache.weights must have shape (Np,).")

    if cache.V.shape[0] != n_points:
        raise ValueError("cache.V row count must match number of points.")

    if cache.Vr.shape != cache.V.shape or cache.Vs.shape != cache.V.shape:
        raise ValueError("cache.Vr/cache.Vs must match cache.V shape.")

    if cache.M.shape[0] != cache.M.shape[1]:
        raise ValueError("cache.M must be square.")

    if not np.allclose(cache.M, cache.M.T, atol=tol, rtol=tol):
        raise ValueError("cache.M must be symmetric.")

    ones = np.ones(n_points)

    if not np.allclose(cache.Dr @ ones, 0.0, atol=1e-8):
        raise ValueError("cache.Dr must differentiate constants to zero.")

    if not np.allclose(cache.Ds @ ones, 0.0, atol=1e-8):
        raise ValueError("cache.Ds must differentiate constants to zero.")

    for face_id in (1, 2, 3):
        E = cache.face_interp[face_id]

        if E.shape[1] != n_points:
            raise ValueError(f"face_interp[{face_id}] must have Np columns.")