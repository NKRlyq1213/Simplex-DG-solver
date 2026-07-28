from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplex_dg.reference.quadrature import TriangleRule, edge_gl_rule


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

_DEFAULT_ATOL = 5e-13


@dataclass(frozen=True)
class DirectBoundaryData:
    """Direct Table 1 boundary data keyed by one-based face ids 1, 2, 3."""

    face_indices: dict[int, np.ndarray]
    face_extract: dict[int, np.ndarray]
    face_weights: dict[int, np.ndarray]
    Br: np.ndarray
    Bs: np.ndarray


def _face_parameter(edge_id: int, rs: np.ndarray) -> np.ndarray:
    rs = np.asarray(rs, dtype=float)

    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs must have shape (N, 2).")

    r = rs[:, 0]
    s = rs[:, 1]

    if edge_id == 1:
        return 0.5 * (s + 1.0)
    if edge_id == 2:
        return 0.5 * (1.0 - s)
    if edge_id == 3:
        return 0.5 * (r + 1.0)

    raise ValueError("edge_id must be 1, 2, or 3.")


def _face_mask(edge_id: int, rs: np.ndarray, *, atol: float) -> np.ndarray:
    rs = np.asarray(rs, dtype=float)

    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs must have shape (N, 2).")

    r = rs[:, 0]
    s = rs[:, 1]

    if edge_id == 1:
        residual = r + s
        target = 0.0
    elif edge_id == 2:
        residual = r
        target = -1.0
    elif edge_id == 3:
        residual = s
        target = -1.0
    else:
        raise ValueError("edge_id must be 1, 2, or 3.")

    return np.isclose(residual, target, atol=atol, rtol=0.0)


def _build_face_extract(indices: np.ndarray, n_volume: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int).reshape(-1)

    if np.any(indices < 0) or np.any(indices >= n_volume):
        raise ValueError("face indices must lie in [0, Nq).")

    if np.unique(indices).size != indices.size:
        raise ValueError("face indices must be unique within each face.")

    extract = np.zeros((indices.size, n_volume), dtype=float)
    extract[np.arange(indices.size), indices] = 1.0
    return extract


def _sorted_face_data(
    *,
    rule: TriangleRule,
    edge_id: int,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_face_expected = rule.order + 1
    mask = _face_mask(edge_id, rule.rs, atol=atol)
    indices = np.flatnonzero(mask)

    if indices.size != n_face_expected:
        raise ValueError(
            f"face {edge_id}: expected {n_face_expected} boundary nodes, got {indices.size}."
        )

    rs_face = np.asarray(rule.rs[indices], dtype=float)
    t_face = _face_parameter(edge_id, rs_face)
    order_idx = np.argsort(t_face)

    indices = indices[order_idx]
    rs_face = rs_face[order_idx]
    t_face = t_face[order_idx]

    if np.any(np.diff(t_face) <= 0.0):
        min_delta = float(np.min(np.diff(t_face)))
        raise ValueError(
            f"face {edge_id}: face parameter ordering is not strictly increasing; "
            f"minimum delta is {min_delta:.3e}."
        )

    edge_rule = edge_gl_rule(edge_id, n_face_expected)
    coord_error = float(np.max(np.abs(rs_face - edge_rule.rs)))

    if coord_error > atol:
        raise ValueError(
            f"face {edge_id}: coordinate mismatch magnitude {coord_error:.3e} exceeds "
            f"atol={atol:.3e}."
        )

    if rule.edge_weights is None:
        raise ValueError("Table 1 direct boundary data requires rule.edge_weights.")

    weights_face = np.asarray(rule.edge_weights[indices], dtype=float)

    if weights_face.shape != (n_face_expected,):
        raise ValueError(
            f"face {edge_id}: expected {n_face_expected} boundary weights, got "
            f"{weights_face.shape[0]}."
        )

    if not np.all(np.isfinite(weights_face)):
        raise ValueError(f"face {edge_id}: boundary weights must be finite.")

    if np.any(weights_face <= 0.0):
        min_weight = float(np.min(weights_face))
        raise ValueError(
            f"face {edge_id}: boundary weights must be positive; minimum value is "
            f"{min_weight:.3e}."
        )

    weight_error = float(np.max(np.abs(weights_face - edge_rule.weights)))

    if weight_error > atol:
        raise ValueError(
            f"face {edge_id}: boundary-weight mismatch magnitude {weight_error:.3e} exceeds "
            f"atol={atol:.3e}."
        )

    return indices, weights_face, rs_face


def build_table1_direct_boundary_data(
    *,
    rule: TriangleRule,
    atol: float = _DEFAULT_ATOL,
) -> DirectBoundaryData:
    """Build raw direct-extraction boundary data for a Table 1 triangle rule.

    The returned dictionaries use the existing one-based face ids 1, 2, 3 that
    match ``reference_edge_nodes`` and ``edge_gl_rule``.
    """

    if atol <= 0.0:
        raise ValueError("atol must be positive.")

    if rule.table != "table1":
        raise ValueError(
            f"Table 1 direct boundary data only supports table1 rules; got {rule.table!r}."
        )

    rs = np.asarray(rule.rs, dtype=float)

    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rule.rs must have shape (Nq, 2).")

    if rule.edge_weights is None:
        raise ValueError("Table 1 direct boundary data requires rule.edge_weights.")

    edge_weights = np.asarray(rule.edge_weights, dtype=float).reshape(-1)
    n_volume = rs.shape[0]

    if edge_weights.shape != (n_volume,):
        raise ValueError("rule.edge_weights must have shape (Nq,).")

    face_indices: dict[int, np.ndarray] = {}
    face_extract: dict[int, np.ndarray] = {}
    face_weights: dict[int, np.ndarray] = {}

    for edge_id in (1, 2, 3):
        indices, weights_face, _ = _sorted_face_data(
            rule=rule,
            edge_id=edge_id,
            atol=atol,
        )
        face_indices[edge_id] = indices
        face_extract[edge_id] = _build_face_extract(indices, n_volume)
        face_weights[edge_id] = weights_face

    Br = np.zeros((n_volume, n_volume), dtype=float)
    Bs = np.zeros((n_volume, n_volume), dtype=float)

    for edge_id in (1, 2, 3):
        extract = face_extract[edge_id]
        weights = face_weights[edge_id]
        weighted_face = extract.T @ (weights[:, None] * extract)

        Br += _FACE_DSDT[edge_id] * weighted_face
        Bs += (-_FACE_DRDT[edge_id]) * weighted_face

    return DirectBoundaryData(
        face_indices=face_indices,
        face_extract=face_extract,
        face_weights=face_weights,
        Br=Br,
        Bs=Bs,
    )
