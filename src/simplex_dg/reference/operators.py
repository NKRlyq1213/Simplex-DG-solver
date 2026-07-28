from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplex_dg.backends import BackendStatus, backend_status
from simplex_dg.reference.basis import grad_vandermonde2d, vandermonde2d
from simplex_dg.reference.quadrature import (
    REFERENCE_AREA,
    EdgeRule,
    TriangleRule,
    edge_gl_rule,
    load_triangle_rule,
)
from simplex_dg.reference.sbp_variants import (
    SBPVariant,
    boundary_representation_for_variant,
    full_sbp_construction_for_variant,
    is_full_sbp_variant,
    normalize_sbp_variant,
)
from simplex_dg.reference.table1_boundary import build_table1_direct_boundary_data
from simplex_dg.reference.table1_full_sbp import build_table1_full_sbp_operators


_FACE_DRDT = {
    1: -2.0,
    2: 0.0,
    3: 2.0,
}

_FACE_DSDT = {
    1: 2.0,
    2: -2.0,
    3: 0.0,
}


@dataclass(frozen=True)
class ReferenceCache:
    """Reference-element operators and SBP-compatible face data.

    ``projection`` stores the coefficient projection ``P_c`` defined by the
    current raw Vandermonde basis and the diagonal reference norm ``H``. It is
    not the nodal polynomial projection ``P = V P_c``.

    ``face_interp`` stores the variant-selected face trace operator:

    - ``projected``: ``V_f P_c``
    - ``full-*``: direct extraction ``E_ext``

    ``face_lift`` stores the lifting operator compatible with the selected
    ``face_interp`` and ``sbp_variant``:

    - ``projected``: ``V M^{-1} V_f^T W_b``
    - ``full-*``: ``H^{-1} E_ext^T W_b``

    ``sbp_variant`` atomically selects ``Dr``, ``Ds``, ``face_interp``, and
    ``face_lift``.
    """

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
    face_lift: dict[int, np.ndarray]

    Br: np.ndarray
    Bs: np.ndarray

    sbp_variant: SBPVariant
    boundary_representation: str
    face_volume_indices: dict[int, np.ndarray] | None

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


def _projected_face_interp_matrix(
    *,
    order: int,
    projection: np.ndarray,
    edge_rule: EdgeRule,
) -> np.ndarray:
    V_face = vandermonde2d(order, edge_rule.rs[:, 0], edge_rule.rs[:, 1])
    return np.ascontiguousarray(V_face @ projection, dtype=float)


def _projected_face_lift_matrix(
    *,
    order: int,
    V: np.ndarray,
    Minv: np.ndarray,
    edge_rule: EdgeRule,
) -> np.ndarray:
    V_face = vandermonde2d(order, edge_rule.rs[:, 0], edge_rule.rs[:, 1])
    return np.ascontiguousarray((V @ Minv @ V_face.T) * edge_rule.weights[None, :], dtype=float)


def _direct_face_lift_matrix(
    *,
    face_extract: np.ndarray,
    face_weights: np.ndarray,
    h_diag: np.ndarray,
) -> np.ndarray:
    return np.ascontiguousarray((face_extract.T * face_weights[None, :]) / h_diag[:, None], dtype=float)


def _face_boundary_matrices(
    *,
    n_points: int,
    face_interp: dict[int, np.ndarray],
    face_weights: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    Br = np.zeros((n_points, n_points), dtype=float)
    Bs = np.zeros((n_points, n_points), dtype=float)

    for face_id in (1, 2, 3):
        E = np.asarray(face_interp[face_id], dtype=float)
        weights = np.asarray(face_weights[face_id], dtype=float).reshape(-1)

        face_term = E.T @ (weights[:, None] * E)
        Br += _FACE_DSDT[face_id] * face_term
        Bs += (-_FACE_DRDT[face_id]) * face_term

    return Br, Bs


def build_reference_cache(
    order: int,
    table: str = "table1",
    n_face: int | None = None,
    area: float = REFERENCE_AREA,
    validate: bool = True,
    sbp_variant: SBPVariant = "projected",
) -> ReferenceCache:
    sbp_variant_norm = normalize_sbp_variant(sbp_variant)
    rule = load_triangle_rule(table=table, order=order)

    if n_face is None:
        n_face = order + 1

    if is_full_sbp_variant(sbp_variant_norm):
        if rule.table != "table1":
            raise ValueError(
                f"sbp_variant={sbp_variant_norm!r} only supports table1 rules; got {rule.table!r}."
            )
        if n_face != order + 1:
            raise ValueError(
                f"sbp_variant={sbp_variant_norm!r} requires n_face == order + 1 == {order + 1}; "
                f"got {n_face}."
            )

    rs = np.asarray(rule.rs, dtype=float)
    weights = np.asarray(rule.weights, dtype=float)

    V = vandermonde2d(order, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(order, rs[:, 0], rs[:, 1])

    Dr_projected, Ds_projected, M, Minv, projection = differentiation_matrices_weighted(
        V=V,
        Vr=Vr,
        Vs=Vs,
        weights=weights,
        area=area,
    )

    edge_rules: dict[int, EdgeRule] = {}
    for face_id in (1, 2, 3):
        edge_rules[face_id] = edge_gl_rule(face_id, n_face)

    face_interp: dict[int, np.ndarray] = {}
    face_lift: dict[int, np.ndarray] = {}
    face_weights: dict[int, np.ndarray] = {}
    face_volume_indices: dict[int, np.ndarray] | None = None
    Br: np.ndarray
    Bs: np.ndarray
    Dr = np.asarray(Dr_projected, dtype=float)
    Ds = np.asarray(Ds_projected, dtype=float)

    if sbp_variant_norm == "projected":
        for face_id in (1, 2, 3):
            edge_rule = edge_rules[face_id]
            face_interp[face_id] = _projected_face_interp_matrix(
                order=order,
                projection=projection,
                edge_rule=edge_rule,
            )
            face_lift[face_id] = _projected_face_lift_matrix(
                order=order,
                V=V,
                Minv=Minv,
                edge_rule=edge_rule,
            )
            face_weights[face_id] = np.ascontiguousarray(edge_rule.weights, dtype=float)

        Br, Bs = _face_boundary_matrices(
            n_points=rs.shape[0],
            face_interp=face_interp,
            face_weights=face_weights,
        )
    else:
        boundary = build_table1_direct_boundary_data(rule=rule)
        full_ops = build_table1_full_sbp_operators(
            rule=rule,
            V_raw=V,
            Vr_raw=Vr,
            Vs_raw=Vs,
            boundary=boundary,
            area=area,
            construction=full_sbp_construction_for_variant(sbp_variant_norm),
            validate=validate,
        )

        Dr = np.asarray(full_ops.Dr, dtype=float)
        Ds = np.asarray(full_ops.Ds, dtype=float)
        Br = np.asarray(boundary.Br, dtype=float)
        Bs = np.asarray(boundary.Bs, dtype=float)
        face_volume_indices = {
            face_id: np.asarray(boundary.face_indices[face_id], dtype=int)
            for face_id in (1, 2, 3)
        }

        for face_id in (1, 2, 3):
            face_interp[face_id] = np.ascontiguousarray(boundary.face_extract[face_id], dtype=float)
            face_weights[face_id] = np.ascontiguousarray(boundary.face_weights[face_id], dtype=float)
            face_lift[face_id] = _direct_face_lift_matrix(
                face_extract=face_interp[face_id],
                face_weights=face_weights[face_id],
                h_diag=full_ops.h_diag,
            )

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
        face_lift=face_lift,
        Br=np.asarray(Br, dtype=float),
        Bs=np.asarray(Bs, dtype=float),
        sbp_variant=sbp_variant_norm,
        boundary_representation=boundary_representation_for_variant(sbp_variant_norm),
        face_volume_indices=face_volume_indices,
        backend=backend_status(),
    )

    if validate:
        validate_reference_cache(cache)

    return cache


def validate_reference_cache(cache: ReferenceCache, tol: float = 1e-10) -> None:
    n_points = cache.rs.shape[0]
    sbp_variant = normalize_sbp_variant(cache.sbp_variant)

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

    if cache.Minv.shape != cache.M.shape:
        raise ValueError("cache.Minv must have the same shape as cache.M.")

    if cache.projection.shape != (cache.V.shape[1], n_points):
        raise ValueError("cache.projection must have shape (Nmodes, Np).")

    if cache.Dr.shape != (n_points, n_points) or cache.Ds.shape != (n_points, n_points):
        raise ValueError("cache.Dr/cache.Ds must have shape (Np, Np).")

    if cache.Br.shape != (n_points, n_points) or cache.Bs.shape != (n_points, n_points):
        raise ValueError("cache.Br/cache.Bs must have shape (Np, Np).")

    if not np.all(np.isfinite(cache.Br)) or not np.all(np.isfinite(cache.Bs)):
        raise ValueError("cache.Br/cache.Bs must be finite.")

    if not np.allclose(cache.M, cache.M.T, atol=tol, rtol=tol):
        raise ValueError("cache.M must be symmetric.")

    if cache.boundary_representation not in ("projected", "direct"):
        raise ValueError("cache.boundary_representation must be 'projected' or 'direct'.")

    ones = np.ones(n_points)

    if not np.allclose(cache.Dr @ ones, 0.0, atol=1e-8):
        raise ValueError("cache.Dr must differentiate constants to zero.")

    if not np.allclose(cache.Ds @ ones, 0.0, atol=1e-8):
        raise ValueError("cache.Ds must differentiate constants to zero.")

    for face_id in (1, 2, 3):
        edge_rule = cache.edge_rules[face_id]
        E = np.asarray(cache.face_interp[face_id], dtype=float)
        L = np.asarray(cache.face_lift[face_id], dtype=float)

        if E.shape != (edge_rule.n_points, n_points):
            raise ValueError(
                f"face_interp[{face_id}] must have shape ({edge_rule.n_points}, {n_points})."
            )

        if L.shape != (n_points, edge_rule.n_points):
            raise ValueError(
                f"face_lift[{face_id}] must have shape ({n_points}, {edge_rule.n_points})."
            )

        if not np.all(np.isfinite(E)) or not np.all(np.isfinite(L)):
            raise ValueError(f"face_interp[{face_id}] and face_lift[{face_id}] must be finite.")

    if cache.boundary_representation == "direct":
        if cache.face_volume_indices is None:
            raise ValueError("direct boundary representation requires face_volume_indices.")

        for face_id in (1, 2, 3):
            indices = np.asarray(cache.face_volume_indices[face_id], dtype=int).reshape(-1)
            expected = cache.edge_rules[face_id].n_points

            if indices.shape != (expected,):
                raise ValueError(
                    f"face_volume_indices[{face_id}] must have shape ({expected},); got {indices.shape}."
                )

            if np.any(indices < 0) or np.any(indices >= n_points):
                raise ValueError(f"face_volume_indices[{face_id}] must lie in [0, {n_points}).")
    else:
        if cache.face_volume_indices is not None:
            raise ValueError("projected boundary representation must not define face_volume_indices.")

    if sbp_variant == "projected" and cache.boundary_representation != "projected":
        raise ValueError("projected sbp_variant must use projected boundary representation.")

    if is_full_sbp_variant(sbp_variant) and cache.boundary_representation != "direct":
        raise ValueError("full sbp variants must use direct boundary representation.")
