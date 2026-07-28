from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.reference import (
    DirectBoundaryData,
    build_reference_cache,
    build_table1_direct_boundary_data,
    build_table1_full_sbp_operators,
)
from simplex_dg.reference.quadrature import TABLE1_RAW, TriangleRule


TABLE1_ORDERS = sorted(TABLE1_RAW)
CONSTRUCTIONS = ("raw", "orthogonalized")


def _build_case(order: int, construction: str):
    ref = build_reference_cache(order=order, table="table1", validate=True)
    boundary = build_table1_direct_boundary_data(rule=ref.rule)
    data = build_table1_full_sbp_operators(
        rule=ref.rule,
        V_raw=ref.V,
        Vr_raw=ref.Vr,
        Vs_raw=ref.Vs,
        boundary=boundary,
        area=ref.area,
        construction=construction,
        validate=True,
    )
    return ref, boundary, data


def _tol(data, factor: float = 512.0) -> float:
    eps = np.finfo(float).eps
    dim = max(data.Dr.shape[0], data.modal_mass.shape[0])
    cond_scale = max(1.0, float(np.linalg.cond(data.modal_mass)))
    return factor * eps * dim * cond_scale


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_shapes_and_finiteness(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    n_volume = ref.rs.shape[0]
    n_modes = ref.V.shape[1]

    assert data.construction == construction
    assert data.order == order
    assert data.h_diag.shape == (n_volume,)
    assert data.modal_mass.shape == (n_modes, n_modes)
    assert data.coefficient_projection.shape == (n_modes, n_volume)
    assert data.polynomial_projection.shape == (n_volume, n_volume)
    assert data.complement_projection.shape == (n_volume, n_volume)
    assert data.Dr_volume.shape == (n_volume, n_volume)
    assert data.Ds_volume.shape == (n_volume, n_volume)
    assert data.delta_Dr.shape == (n_volume, n_volume)
    assert data.delta_Ds.shape == (n_volume, n_volume)
    assert data.Dr.shape == (n_volume, n_volume)
    assert data.Ds.shape == (n_volume, n_volume)

    for array in (
        data.h_diag,
        data.modal_mass,
        data.coefficient_projection,
        data.polynomial_projection,
        data.complement_projection,
        data.Dr_volume,
        data.Ds_volume,
        data.delta_Dr,
        data.delta_Ds,
        data.Dr,
        data.Ds,
        boundary.Br,
        boundary.Bs,
    ):
        assert np.all(np.isfinite(array))

    if construction == "raw":
        assert data.V_orth is None
        assert data.Vr_orth is None
        assert data.Vs_orth is None
        assert data.cholesky_factor is None
    else:
        assert data.V_orth is not None
        assert data.Vr_orth is not None
        assert data.Vs_orth is not None
        assert data.cholesky_factor is not None
        assert data.V_orth.shape == (n_volume, n_modes)
        assert data.Vr_orth.shape == (n_volume, n_modes)
        assert data.Vs_orth.shape == (n_volume, n_modes)
        assert data.cholesky_factor.shape == (n_modes, n_modes)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_projection_identities(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)
    P = data.polynomial_projection
    Q = data.complement_projection

    np.testing.assert_allclose(P @ P, P, atol=tol, rtol=tol)
    np.testing.assert_allclose(Q @ Q, Q, atol=tol, rtol=tol)
    np.testing.assert_allclose(P @ Q, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(Q @ P, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(P @ ref.V, ref.V, atol=tol, rtol=tol)
    np.testing.assert_allclose(Q @ ref.V, 0.0, atol=tol, rtol=tol)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_weighted_self_adjointness(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)
    H = np.diag(data.h_diag)
    P = data.polynomial_projection
    Q = data.complement_projection

    np.testing.assert_allclose(P.T @ H, H @ P, atol=tol, rtol=tol)
    np.testing.assert_allclose(Q.T @ H, H @ Q, atol=tol, rtol=tol)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
def test_table1_full_sbp_discrete_orthonormality(order: int):
    ref, boundary, data = _build_case(order, "orthogonalized")
    tol = _tol(data)
    H = np.diag(data.h_diag)

    np.testing.assert_allclose(
        data.V_orth.T @ H @ data.V_orth,
        np.eye(ref.V.shape[1]),
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_volume_operator_exactness(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)

    np.testing.assert_allclose(data.Dr_volume @ ref.V, ref.Vr, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.Ds_volume @ ref.V, ref.Vs, atol=tol, rtol=tol)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_corrections_annihilate_polynomial_space(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)

    np.testing.assert_allclose(data.delta_Dr @ ref.V, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.delta_Ds @ ref.V, 0.0, atol=tol, rtol=tol)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_corrected_polynomial_exactness(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)

    np.testing.assert_allclose(data.Dr @ ref.V, ref.Vr, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.Ds @ ref.V, ref.Vs, atol=tol, rtol=tol)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_correction_equation(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)
    H = np.diag(data.h_diag)
    P = data.polynomial_projection

    np.testing.assert_allclose(
        H @ data.delta_Dr + data.delta_Dr.T @ H,
        boundary.Br - P.T @ boundary.Br @ P,
        atol=tol,
        rtol=tol,
    )
    np.testing.assert_allclose(
        H @ data.delta_Ds + data.delta_Ds.T @ H,
        boundary.Bs - P.T @ boundary.Bs @ P,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_identities(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)
    H = np.diag(data.h_diag)

    np.testing.assert_allclose(H @ data.Dr + data.Dr.T @ H, boundary.Br, atol=tol, rtol=tol)
    np.testing.assert_allclose(H @ data.Ds + data.Ds.T @ H, boundary.Bs, atol=tol, rtol=tol)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
def test_table1_full_sbp_raw_and_orthogonalized_constructions_match(order: int):
    ref, boundary, raw = _build_case(order, "raw")
    _, _, orth = _build_case(order, "orthogonalized")
    tol = max(_tol(raw), _tol(orth))

    np.testing.assert_allclose(orth.polynomial_projection, raw.polynomial_projection, atol=tol, rtol=tol)
    np.testing.assert_allclose(orth.complement_projection, raw.complement_projection, atol=tol, rtol=tol)
    np.testing.assert_allclose(orth.Dr_volume, raw.Dr_volume, atol=tol, rtol=tol)
    np.testing.assert_allclose(orth.Ds_volume, raw.Ds_volume, atol=tol, rtol=tol)
    np.testing.assert_allclose(orth.delta_Dr, raw.delta_Dr, atol=tol, rtol=tol)
    np.testing.assert_allclose(orth.delta_Ds, raw.delta_Ds, atol=tol, rtol=tol)
    np.testing.assert_allclose(orth.Dr, raw.Dr, atol=tol, rtol=tol)
    np.testing.assert_allclose(orth.Ds, raw.Ds, atol=tol, rtol=tol)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_matches_current_projected_volume_operators(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)

    np.testing.assert_allclose(data.Dr_volume, ref.Dr, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.Ds_volume, ref.Ds, atol=tol, rtol=tol)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_table1_full_sbp_constant_differentiation(order: int, construction: str):
    ref, boundary, data = _build_case(order, construction)
    tol = _tol(data)
    ones = np.ones(ref.rs.shape[0], dtype=float)

    np.testing.assert_allclose(data.Dr @ ones, 0.0, atol=tol, rtol=tol)
    np.testing.assert_allclose(data.Ds @ ones, 0.0, atol=tol, rtol=tol)


def test_table1_full_sbp_rejects_table2():
    ref = build_reference_cache(order=4, table="table2", validate=True)
    boundary = build_table1_direct_boundary_data(
        rule=build_reference_cache(order=4, table="table1", validate=True).rule
    )

    with pytest.raises(ValueError, match="only supports table1"):
        build_table1_full_sbp_operators(
            rule=ref.rule,
            V_raw=ref.V,
            Vr_raw=ref.Vr,
            Vs_raw=ref.Vs,
            boundary=boundary,
            area=ref.area,
            construction="raw",
        )


def test_table1_full_sbp_rejects_invalid_construction():
    ref = build_reference_cache(order=4, table="table1", validate=True)
    boundary = build_table1_direct_boundary_data(rule=ref.rule)

    with pytest.raises(ValueError, match="construction must be"):
        build_table1_full_sbp_operators(
            rule=ref.rule,
            V_raw=ref.V,
            Vr_raw=ref.Vr,
            Vs_raw=ref.Vs,
            boundary=boundary,
            area=ref.area,
            construction="full-raw",
        )


def test_table1_full_sbp_rejects_nonpositive_area():
    ref = build_reference_cache(order=4, table="table1", validate=True)
    boundary = build_table1_direct_boundary_data(rule=ref.rule)

    with pytest.raises(ValueError, match="area must be positive"):
        build_table1_full_sbp_operators(
            rule=ref.rule,
            V_raw=ref.V,
            Vr_raw=ref.Vr,
            Vs_raw=ref.Vs,
            boundary=boundary,
            area=0.0,
            construction="raw",
        )


def test_table1_full_sbp_rejects_nonfinite_weights():
    ref = build_reference_cache(order=4, table="table1", validate=True)
    boundary = build_table1_direct_boundary_data(rule=ref.rule)
    bad_weights = ref.rule.weights.copy()
    bad_weights[0] = np.nan
    bad_rule = TriangleRule(
        table=ref.rule.table,
        order=ref.rule.order,
        bary_raw=ref.rule.bary_raw,
        rs=ref.rule.rs,
        weights=bad_weights,
        edge_weights=ref.rule.edge_weights,
    )

    with pytest.raises(ValueError, match="rule.weights"):
        build_table1_full_sbp_operators(
            rule=bad_rule,
            V_raw=ref.V,
            Vr_raw=ref.Vr,
            Vs_raw=ref.Vs,
            boundary=boundary,
            area=ref.area,
            construction="raw",
        )


def test_table1_full_sbp_rejects_mismatched_vandermonde_shapes():
    ref = build_reference_cache(order=4, table="table1", validate=True)
    boundary = build_table1_direct_boundary_data(rule=ref.rule)

    with pytest.raises(ValueError, match="V_raw row count"):
        build_table1_full_sbp_operators(
            rule=ref.rule,
            V_raw=ref.V[:-1],
            Vr_raw=ref.Vr[:-1],
            Vs_raw=ref.Vs[:-1],
            boundary=boundary,
            area=ref.area,
            construction="raw",
        )

    with pytest.raises(ValueError, match="Vr_raw must have shape"):
        build_table1_full_sbp_operators(
            rule=ref.rule,
            V_raw=ref.V,
            Vr_raw=ref.Vr[:, :-1],
            Vs_raw=ref.Vs,
            boundary=boundary,
            area=ref.area,
            construction="raw",
        )


def test_table1_full_sbp_rejects_mismatched_boundary_shapes():
    ref = build_reference_cache(order=4, table="table1", validate=True)
    boundary = build_table1_direct_boundary_data(rule=ref.rule)
    bad_boundary = DirectBoundaryData(
        face_indices=boundary.face_indices,
        face_extract={
            1: boundary.face_extract[1][:, :-1],
            2: boundary.face_extract[2],
            3: boundary.face_extract[3],
        },
        face_weights=boundary.face_weights,
        Br=boundary.Br,
        Bs=boundary.Bs,
    )

    with pytest.raises(ValueError, match="boundary.face_extract\\[1\\]"):
        build_table1_full_sbp_operators(
            rule=ref.rule,
            V_raw=ref.V,
            Vr_raw=ref.Vr,
            Vs_raw=ref.Vs,
            boundary=bad_boundary,
            area=ref.area,
            construction="raw",
        )


def test_table1_full_sbp_rejects_forced_cholesky_failure(monkeypatch: pytest.MonkeyPatch):
    ref = build_reference_cache(order=4, table="table1", validate=True)
    boundary = build_table1_direct_boundary_data(rule=ref.rule)

    def _raise_cholesky_failure(_matrix):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "cholesky", _raise_cholesky_failure)

    with pytest.raises(ValueError, match="Cholesky factorization failed"):
        build_table1_full_sbp_operators(
            rule=ref.rule,
            V_raw=ref.V,
            Vr_raw=ref.Vr,
            Vs_raw=ref.Vs,
            boundary=boundary,
            area=ref.area,
            construction="orthogonalized",
        )
