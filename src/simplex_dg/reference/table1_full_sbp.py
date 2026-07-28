from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from simplex_dg.reference.quadrature import TriangleRule
from simplex_dg.reference.table1_boundary import DirectBoundaryData


FullSBPConstruction = Literal["raw", "orthogonalized"]

_ALLOWED_CONSTRUCTIONS = ("raw", "orthogonalized")


@dataclass(frozen=True)
class FullSBPOperatorData:
    """Standalone Table 1 full-SBP differentiation operators.

    ``coefficient_projection`` always denotes the raw-basis coefficient
    projection ``P_c = M_raw^{-1} V_raw^T H``. The returned
    ``polynomial_projection`` / ``complement_projection`` / volume operators
    follow the selected construction.
    """

    construction: FullSBPConstruction
    order: int
    area: float

    h_diag: np.ndarray
    modal_mass: np.ndarray

    coefficient_projection: np.ndarray
    polynomial_projection: np.ndarray
    complement_projection: np.ndarray

    Dr_volume: np.ndarray
    Ds_volume: np.ndarray

    delta_Dr: np.ndarray
    delta_Ds: np.ndarray

    Dr: np.ndarray
    Ds: np.ndarray

    V_orth: np.ndarray | None = None
    Vr_orth: np.ndarray | None = None
    Vs_orth: np.ndarray | None = None
    cholesky_factor: np.ndarray | None = None


def _normalize_construction(construction: str) -> FullSBPConstruction:
    if not isinstance(construction, str):
        raise ValueError("construction must be 'raw' or 'orthogonalized'.")

    normalized = construction.lower().strip()

    if normalized not in _ALLOWED_CONSTRUCTIONS:
        raise ValueError("construction must be 'raw' or 'orthogonalized'.")

    return normalized  # type: ignore[return-value]


def _as_matrix(name: str, value: np.ndarray, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D; got ndim={array.ndim}.")

    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _as_vector(name: str, value: np.ndarray, *, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)

    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},); got {array.shape}.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _matrix_infinity_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, ord=np.inf))


def _modal_mass_symmetry_tolerance(modal_mass: np.ndarray) -> float:
    n_modes = modal_mass.shape[0]
    scale = max(1.0, _matrix_infinity_norm(modal_mass))
    return 128.0 * np.finfo(float).eps * max(1, n_modes) * scale


def _symmetrize_modal_mass(modal_mass: np.ndarray) -> np.ndarray:
    residual = _matrix_infinity_norm(modal_mass - modal_mass.T)
    tol = _modal_mass_symmetry_tolerance(modal_mass)

    if residual > tol:
        raise ValueError(
            "modal mass matrix is not symmetric enough for Cholesky: "
            f"residual={residual:.3e}, tol={tol:.3e}."
        )

    return 0.5 * (modal_mass + modal_mass.T)


def _validate_boundary_data(boundary: DirectBoundaryData, *, n_volume: int, order: int) -> None:
    if not isinstance(boundary, DirectBoundaryData):
        raise ValueError("boundary must be a DirectBoundaryData instance.")

    expected_face_size = order + 1

    for face_id in (1, 2, 3):
        if face_id not in boundary.face_indices:
            raise ValueError(f"boundary.face_indices is missing face {face_id}.")
        if face_id not in boundary.face_extract:
            raise ValueError(f"boundary.face_extract is missing face {face_id}.")
        if face_id not in boundary.face_weights:
            raise ValueError(f"boundary.face_weights is missing face {face_id}.")

        indices = np.asarray(boundary.face_indices[face_id], dtype=int).reshape(-1)
        extract = _as_matrix(
            f"boundary.face_extract[{face_id}]",
            boundary.face_extract[face_id],
            shape=(expected_face_size, n_volume),
        )
        weights = _as_vector(
            f"boundary.face_weights[{face_id}]",
            boundary.face_weights[face_id],
            size=expected_face_size,
        )

        if indices.shape != (expected_face_size,):
            raise ValueError(
                f"boundary.face_indices[{face_id}] must have shape ({expected_face_size},); "
                f"got {indices.shape}."
            )

        if np.any(indices < 0) or np.any(indices >= n_volume):
            raise ValueError(
                f"boundary.face_indices[{face_id}] must lie in [0, {n_volume}); "
                f"got min={indices.min()}, max={indices.max()}."
            )

        if np.unique(indices).size != indices.size:
            raise ValueError(f"boundary.face_indices[{face_id}] must be unique.")

        if np.any(weights <= 0.0):
            min_weight = float(np.min(weights))
            raise ValueError(
                f"boundary.face_weights[{face_id}] must be positive; minimum={min_weight:.3e}."
            )

        row_sums = np.sum(extract, axis=1)
        nnz = np.count_nonzero(extract, axis=1)

        if not np.all(np.isin(extract, (0.0, 1.0))):
            raise ValueError(f"boundary.face_extract[{face_id}] must be one-hot.")

        if not np.all(row_sums == 1.0) or not np.all(nnz == 1):
            raise ValueError(f"boundary.face_extract[{face_id}] must be one-hot by row.")

    _as_matrix("boundary.Br", boundary.Br, shape=(n_volume, n_volume))
    _as_matrix("boundary.Bs", boundary.Bs, shape=(n_volume, n_volume))


def _raw_modal_mass(V_raw: np.ndarray, h_diag: np.ndarray) -> np.ndarray:
    return V_raw.T @ (h_diag[:, None] * V_raw)


def _raw_coefficient_projection(modal_mass: np.ndarray, V_raw: np.ndarray, h_diag: np.ndarray) -> np.ndarray:
    rhs = V_raw.T * h_diag[None, :]
    try:
        return np.linalg.solve(modal_mass, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Coefficient projection solve failed for the modal mass matrix.") from exc


def _orthogonalized_basis(
    V_raw: np.ndarray,
    Vr_raw: np.ndarray,
    Vs_raw: np.ndarray,
    modal_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    modal_mass_cholesky = _symmetrize_modal_mass(modal_mass)

    try:
        L = np.linalg.cholesky(modal_mass_cholesky)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Cholesky factorization failed for the modal mass matrix.") from exc

    V_orth = np.linalg.solve(L, V_raw.T).T
    Vr_orth = np.linalg.solve(L, Vr_raw.T).T
    Vs_orth = np.linalg.solve(L, Vs_raw.T).T

    return V_orth, Vr_orth, Vs_orth, L


def _build_raw_construction(
    *,
    V_raw: np.ndarray,
    Vr_raw: np.ndarray,
    Vs_raw: np.ndarray,
    h_diag: np.ndarray,
    coefficient_projection: np.ndarray,
    polynomial_projection: np.ndarray,
    complement_projection: np.ndarray,
    boundary: DirectBoundaryData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_volume = V_raw.shape[0]
    identity = np.eye(n_volume, dtype=float)

    Dr_volume = Vr_raw @ coefficient_projection
    Ds_volume = Vs_raw @ coefficient_projection

    correction_rhs_r = (identity + polynomial_projection.T) @ boundary.Br @ complement_projection
    correction_rhs_s = (identity + polynomial_projection.T) @ boundary.Bs @ complement_projection

    delta_Dr = 0.5 * (correction_rhs_r / h_diag[:, None])
    delta_Ds = 0.5 * (correction_rhs_s / h_diag[:, None])

    return Dr_volume, Ds_volume, delta_Dr, delta_Ds, Dr_volume + delta_Dr, Ds_volume + delta_Ds


def _build_orthogonalized_construction(
    *,
    V_raw: np.ndarray,
    Vr_raw: np.ndarray,
    Vs_raw: np.ndarray,
    h_diag: np.ndarray,
    modal_mass: np.ndarray,
    boundary: DirectBoundaryData,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_volume = V_raw.shape[0]
    identity = np.eye(n_volume, dtype=float)
    V_orth, Vr_orth, Vs_orth, cholesky_factor = _orthogonalized_basis(
        V_raw,
        Vr_raw,
        Vs_raw,
        modal_mass,
    )

    weighted_orth_basis_transpose = V_orth.T * h_diag[None, :]
    polynomial_projection = V_orth @ weighted_orth_basis_transpose
    complement_projection = identity - polynomial_projection

    Dr_volume = Vr_orth @ weighted_orth_basis_transpose
    Ds_volume = Vs_orth @ weighted_orth_basis_transpose

    boundary_residual_r = boundary.Br @ complement_projection
    boundary_residual_s = boundary.Bs @ complement_projection
    orth_metric_inverse = V_orth @ V_orth.T

    delta_Dr = 0.5 * (
        boundary_residual_r / h_diag[:, None]
        + orth_metric_inverse @ boundary_residual_r
    )
    delta_Ds = 0.5 * (
        boundary_residual_s / h_diag[:, None]
        + orth_metric_inverse @ boundary_residual_s
    )

    return (
        polynomial_projection,
        complement_projection,
        Dr_volume,
        Ds_volume,
        delta_Dr,
        delta_Ds,
        Dr_volume + delta_Dr,
        Ds_volume + delta_Ds,
        V_orth,
        Vr_orth,
        Vs_orth,
        cholesky_factor,
    )


def _validate_full_sbp_operator_data(
    *,
    data: FullSBPOperatorData,
    V_raw: np.ndarray,
    Vr_raw: np.ndarray,
    Vs_raw: np.ndarray,
    boundary: DirectBoundaryData,
) -> None:
    n_volume, n_modes = V_raw.shape
    h_diag = _as_vector("data.h_diag", data.h_diag, size=n_volume)
    H = np.diag(h_diag)
    tol_scale = max(1.0, np.linalg.cond(data.modal_mass))
    tol = 512.0 * np.finfo(float).eps * max(n_volume, n_modes) * tol_scale

    for name, matrix, shape in (
        ("data.modal_mass", data.modal_mass, (n_modes, n_modes)),
        ("data.coefficient_projection", data.coefficient_projection, (n_modes, n_volume)),
        ("data.polynomial_projection", data.polynomial_projection, (n_volume, n_volume)),
        ("data.complement_projection", data.complement_projection, (n_volume, n_volume)),
        ("data.Dr_volume", data.Dr_volume, (n_volume, n_volume)),
        ("data.Ds_volume", data.Ds_volume, (n_volume, n_volume)),
        ("data.delta_Dr", data.delta_Dr, (n_volume, n_volume)),
        ("data.delta_Ds", data.delta_Ds, (n_volume, n_volume)),
        ("data.Dr", data.Dr, (n_volume, n_volume)),
        ("data.Ds", data.Ds, (n_volume, n_volume)),
    ):
        _as_matrix(name, matrix, shape=shape)

    P = data.polynomial_projection
    Q = data.complement_projection

    np.testing.assert_allclose(P @ P, P, atol=tol, rtol=tol)
    np.testing.assert_allclose(Q @ Q, Q, atol=tol, rtol=tol)
    np.testing.assert_allclose(P @ Q, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(Q @ P, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(P @ V_raw, V_raw, atol=tol, rtol=tol)
    np.testing.assert_allclose(Q @ V_raw, 0.0, atol=tol, rtol=tol)

    np.testing.assert_allclose(P.T @ H, H @ P, atol=tol, rtol=tol)
    np.testing.assert_allclose(Q.T @ H, H @ Q, atol=tol, rtol=tol)

    np.testing.assert_allclose(data.Dr_volume @ V_raw, Vr_raw, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.Ds_volume @ V_raw, Vs_raw, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.delta_Dr @ V_raw, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.delta_Ds @ V_raw, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.Dr @ V_raw, Vr_raw, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.Ds @ V_raw, Vs_raw, atol=tol, rtol=tol)

    correction_target_r = boundary.Br - P.T @ boundary.Br @ P
    correction_target_s = boundary.Bs - P.T @ boundary.Bs @ P
    np.testing.assert_allclose(
        H @ data.delta_Dr + data.delta_Dr.T @ H,
        correction_target_r,
        atol=tol,
        rtol=tol,
    )
    np.testing.assert_allclose(
        H @ data.delta_Ds + data.delta_Ds.T @ H,
        correction_target_s,
        atol=tol,
        rtol=tol,
    )

    np.testing.assert_allclose(H @ data.Dr + data.Dr.T @ H, boundary.Br, atol=tol, rtol=tol)
    np.testing.assert_allclose(H @ data.Ds + data.Ds.T @ H, boundary.Bs, atol=tol, rtol=tol)

    ones = np.ones(n_volume, dtype=float)
    np.testing.assert_allclose(data.Dr @ ones, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.Ds @ ones, 0.0, atol=tol, rtol=tol)

    if data.construction == "orthogonalized":
        if data.V_orth is None or data.Vr_orth is None or data.Vs_orth is None or data.cholesky_factor is None:
            raise ValueError(
                "orthogonalized construction must retain V_orth, Vr_orth, Vs_orth, and cholesky_factor."
            )

        _as_matrix("data.V_orth", data.V_orth, shape=(n_volume, n_modes))
        _as_matrix("data.Vr_orth", data.Vr_orth, shape=(n_volume, n_modes))
        _as_matrix("data.Vs_orth", data.Vs_orth, shape=(n_volume, n_modes))
        _as_matrix("data.cholesky_factor", data.cholesky_factor, shape=(n_modes, n_modes))
        np.testing.assert_allclose(
            data.V_orth.T @ H @ data.V_orth,
            np.eye(n_modes, dtype=float),
            atol=tol,
            rtol=tol,
        )
    else:
        if any(value is not None for value in (data.V_orth, data.Vr_orth, data.Vs_orth, data.cholesky_factor)):
            raise ValueError("raw construction must not populate orthogonalized diagnostics.")


def build_table1_full_sbp_operators(
    *,
    rule: TriangleRule,
    V_raw: np.ndarray,
    Vr_raw: np.ndarray,
    Vs_raw: np.ndarray,
    boundary: DirectBoundaryData,
    area: float,
    construction: FullSBPConstruction,
    validate: bool = True,
) -> FullSBPOperatorData:
    """Build standalone Table 1 full-SBP differentiation operators."""

    if not isinstance(rule, TriangleRule):
        raise ValueError("rule must be a TriangleRule instance.")

    if rule.table != "table1":
        raise ValueError(f"build_table1_full_sbp_operators only supports table1; got {rule.table!r}.")

    area = float(area)

    if not np.isfinite(area) or area <= 0.0:
        raise ValueError(f"area must be positive and finite; got {area!r}.")

    weights = _as_vector("rule.weights", rule.weights)
    n_volume = weights.size
    V_raw = _as_matrix("V_raw", V_raw)

    if V_raw.shape[0] != n_volume:
        raise ValueError(
            f"V_raw row count must match rule weights: expected {n_volume}, got {V_raw.shape[0]}."
        )

    Vr_raw = _as_matrix("Vr_raw", Vr_raw, shape=V_raw.shape)
    Vs_raw = _as_matrix("Vs_raw", Vs_raw, shape=V_raw.shape)

    _validate_boundary_data(boundary, n_volume=n_volume, order=rule.order)

    h_diag = area * weights

    if not np.all(np.isfinite(h_diag)):
        raise ValueError("h_diag must be finite.")

    if np.any(h_diag <= 0.0):
        min_h = float(np.min(h_diag))
        raise ValueError(f"h_diag must be positive; minimum value is {min_h:.3e}.")

    construction_norm = _normalize_construction(construction)
    modal_mass = _raw_modal_mass(V_raw, h_diag)
    modal_mass = _as_matrix("modal_mass", modal_mass, shape=(V_raw.shape[1], V_raw.shape[1]))
    modal_mass = _symmetrize_modal_mass(modal_mass)

    coefficient_projection = _raw_coefficient_projection(modal_mass, V_raw, h_diag)
    polynomial_projection_raw = V_raw @ coefficient_projection
    complement_projection_raw = np.eye(n_volume, dtype=float) - polynomial_projection_raw

    if construction_norm == "raw":
        Dr_volume, Ds_volume, delta_Dr, delta_Ds, Dr, Ds = _build_raw_construction(
            V_raw=V_raw,
            Vr_raw=Vr_raw,
            Vs_raw=Vs_raw,
            h_diag=h_diag,
            coefficient_projection=coefficient_projection,
            polynomial_projection=polynomial_projection_raw,
            complement_projection=complement_projection_raw,
            boundary=boundary,
        )
        data = FullSBPOperatorData(
            construction=construction_norm,
            order=rule.order,
            area=area,
            h_diag=h_diag,
            modal_mass=modal_mass,
            coefficient_projection=coefficient_projection,
            polynomial_projection=polynomial_projection_raw,
            complement_projection=complement_projection_raw,
            Dr_volume=Dr_volume,
            Ds_volume=Ds_volume,
            delta_Dr=delta_Dr,
            delta_Ds=delta_Ds,
            Dr=Dr,
            Ds=Ds,
        )
    else:
        (
            polynomial_projection,
            complement_projection,
            Dr_volume,
            Ds_volume,
            delta_Dr,
            delta_Ds,
            Dr,
            Ds,
            V_orth,
            Vr_orth,
            Vs_orth,
            cholesky_factor,
        ) = _build_orthogonalized_construction(
            V_raw=V_raw,
            Vr_raw=Vr_raw,
            Vs_raw=Vs_raw,
            h_diag=h_diag,
            modal_mass=modal_mass,
            boundary=boundary,
        )
        data = FullSBPOperatorData(
            construction=construction_norm,
            order=rule.order,
            area=area,
            h_diag=h_diag,
            modal_mass=modal_mass,
            coefficient_projection=coefficient_projection,
            polynomial_projection=polynomial_projection,
            complement_projection=complement_projection,
            Dr_volume=Dr_volume,
            Ds_volume=Ds_volume,
            delta_Dr=delta_Dr,
            delta_Ds=delta_Ds,
            Dr=Dr,
            Ds=Ds,
            V_orth=V_orth,
            Vr_orth=Vr_orth,
            Vs_orth=Vs_orth,
            cholesky_factor=cholesky_factor,
        )

    if validate:
        _validate_full_sbp_operator_data(
            data=data,
            V_raw=V_raw,
            Vr_raw=Vr_raw,
            Vs_raw=Vs_raw,
            boundary=boundary,
        )

    return data
