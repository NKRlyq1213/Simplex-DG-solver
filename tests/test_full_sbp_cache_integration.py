from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import (
    build_reference_cache,
    build_table1_direct_boundary_data,
    build_table1_full_sbp_operators,
    vandermonde2d,
)
from simplex_dg.trace import TraceCache, build_trace_cache, gather_neighbor_traces, pair_face_traces


TABLE1_ORDERS = (1, 2, 3, 4)


def _projected_face_interp_formula(ref, face_id: int) -> np.ndarray:
    edge = ref.edge_rules[face_id]
    v_face = vandermonde2d(ref.order, edge.rs[:, 0], edge.rs[:, 1])
    return v_face @ ref.projection


def _projected_face_lift_formula(ref, face_id: int) -> np.ndarray:
    edge = ref.edge_rules[face_id]
    v_face = vandermonde2d(ref.order, edge.rs[:, 0], edge.rs[:, 1])
    return (ref.V @ ref.Minv @ v_face.T) * edge.weights[None, :]


def _direct_face_lift_formula(ref, face_id: int) -> np.ndarray:
    E = ref.face_interp[face_id]
    wb = ref.edge_rules[face_id].weights
    h_diag = ref.area * ref.weights
    return (E.T * wb[None, :]) / h_diag[:, None]


def _gather_exact_face_geometry(geom, trace):
    Xf = geom.X_face
    XP = np.empty_like(Xf)

    K, n_faces, Nf, _ = Xf.shape

    for k in range(K):
        for f in range(n_faces):
            if trace.is_boundary[k, f]:
                XP[k, f, :, :] = np.nan
                continue

            nbr = trace.neighbor_elements[k, f]
            nbr_f = trace.neighbor_faces[k, f]

            if trace.face_flip[k, f]:
                XP[k, f, :, :] = Xf[nbr, nbr_f, ::-1, :]
            else:
                XP[k, f, :, :] = Xf[nbr, nbr_f, :, :]

    return XP


@pytest.mark.parametrize(("table", "order"), [("table1", 4), ("table2", 4)])
def test_projected_reference_cache_default_and_explicit_projected_match_previous_face_ops(table: str, order: int):
    default = build_reference_cache(order=order, table=table, validate=True)
    projected = build_reference_cache(order=order, table=table, validate=True, sbp_variant="projected")

    np.testing.assert_allclose(projected.Dr, default.Dr, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(projected.Ds, default.Ds, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(projected.M, default.M, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(projected.projection, default.projection, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(projected.weights, default.weights, atol=0.0, rtol=0.0)

    assert projected.sbp_variant == "projected"
    assert projected.boundary_representation == "projected"
    assert projected.face_volume_indices is None

    for face_id in (1, 2, 3):
        np.testing.assert_allclose(projected.face_interp[face_id], default.face_interp[face_id], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(projected.face_lift[face_id], default.face_lift[face_id], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            projected.face_interp[face_id],
            _projected_face_interp_formula(projected, face_id),
            atol=2e-14,
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            projected.face_lift[face_id],
            _projected_face_lift_formula(projected, face_id),
            atol=2e-14,
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            projected.edge_rules[face_id].weights,
            default.edge_rules[face_id].weights,
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("sbp_variant", ["full-raw", "full-orth"])
def test_full_reference_cache_matches_task1_and_task2_builders(order: int, sbp_variant: str):
    ref = build_reference_cache(order=order, table="table1", validate=True, sbp_variant=sbp_variant)
    boundary = build_table1_direct_boundary_data(rule=ref.rule)
    full_ops = build_table1_full_sbp_operators(
        rule=ref.rule,
        V_raw=ref.V,
        Vr_raw=ref.Vr,
        Vs_raw=ref.Vs,
        boundary=boundary,
        area=ref.area,
        construction="raw" if sbp_variant == "full-raw" else "orthogonalized",
        validate=True,
    )

    assert ref.sbp_variant == sbp_variant
    assert ref.boundary_representation == "direct"

    np.testing.assert_allclose(ref.Dr, full_ops.Dr, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(ref.Ds, full_ops.Ds, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(ref.Br, boundary.Br, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(ref.Bs, boundary.Bs, atol=0.0, rtol=0.0)

    for face_id in (1, 2, 3):
        np.testing.assert_allclose(ref.face_interp[face_id], boundary.face_extract[face_id], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            ref.face_lift[face_id],
            _direct_face_lift_formula(ref, face_id),
            atol=2e-14,
            rtol=2e-14,
        )
        np.testing.assert_array_equal(ref.face_volume_indices[face_id], boundary.face_indices[face_id])


@pytest.mark.parametrize("order", TABLE1_ORDERS)
@pytest.mark.parametrize("sbp_variant", ["full-raw", "full-orth"])
def test_full_reference_cache_direct_trace_random_extraction_identity(order: int, sbp_variant: str):
    ref = build_reference_cache(order=order, table="table1", validate=True, sbp_variant=sbp_variant)
    rng = np.random.default_rng(20260728 + 10 * order)
    q = rng.standard_normal(ref.rs.shape[0])

    for face_id in (1, 2, 3):
        expected = q[ref.face_volume_indices[face_id]]
        actual = ref.face_interp[face_id] @ q
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("table", "sbp_variant"),
    [
        ("table1", "projected"),
        ("table2", "projected"),
        ("table1", "full-raw"),
        ("table1", "full-orth"),
    ],
)
def test_reference_cache_lift_duality_across_variants(table: str, sbp_variant: str):
    ref = build_reference_cache(order=4, table=table, validate=True, sbp_variant=sbp_variant)
    rng = np.random.default_rng(20260728)
    q = rng.standard_normal(ref.rs.shape[0])
    h_diag = ref.area * ref.weights

    for face_id in (1, 2, 3):
        p = rng.standard_normal(ref.edge_rules[face_id].n_points)
        E = ref.face_interp[face_id]
        L = ref.face_lift[face_id]
        wb = ref.edge_rules[face_id].weights

        lhs = np.dot(q, h_diag * (L @ p))
        rhs = np.dot(E @ q, wb * p)
        np.testing.assert_allclose(lhs, rhs, atol=5e-13, rtol=5e-13)


@pytest.mark.parametrize(
    ("table", "sbp_variant"),
    [
        ("table1", "projected"),
        ("table2", "projected"),
        ("table1", "full-raw"),
        ("table1", "full-orth"),
    ],
)
def test_trace_cache_copies_variant_face_interp(table: str, sbp_variant: str):
    ref = build_reference_cache(order=4, table=table, validate=True, sbp_variant=sbp_variant)
    mesh = build_octa_sphere_mesh(ndivs=1, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)

    for face_id in (1, 2, 3):
        np.testing.assert_allclose(trace.face_interp[face_id - 1], ref.face_interp[face_id], atol=0.0, rtol=0.0)


def test_full_trace_neighbor_orientation_with_direct_extraction():
    ref = build_reference_cache(order=4, table="table1", validate=True, sbp_variant="full-raw")
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)

    interior = ~trace.is_boundary
    assert np.any(interior & trace.face_flip)

    XP = _gather_exact_face_geometry(geom, trace)

    for comp in range(3):
        traces = pair_face_traces(geom.X[:, :, comp], trace, use_numba=False)
        np.testing.assert_allclose(traces.qM, geom.X_face[:, :, :, comp], atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(traces.qP[interior], XP[:, :, :, comp][interior], atol=2e-14, rtol=2e-14)


def test_gather_neighbor_traces_respects_flip_false_and_true_branches():
    trace = TraceCache(
        n_elements=2,
        n_points=4,
        n_faces=3,
        n_face_points=3,
        face_interp=np.zeros((3, 3, 4), dtype=float),
        face_weights=np.ones((3, 3), dtype=float) / 3.0,
        neighbor_elements=np.array([[1, 1, -1], [0, 0, -1]], dtype=int),
        neighbor_faces=np.array([[0, 1, -1], [0, 1, -1]], dtype=int),
        is_boundary=np.array([[False, False, True], [False, False, True]], dtype=bool),
        face_flip=np.array([[False, True, False], [False, True, False]], dtype=bool),
    )

    qM = np.zeros((2, 3, 3), dtype=float)
    qM[0, 0] = np.array([1.0, 2.0, 3.0])
    qM[1, 0] = np.array([10.0, 20.0, 30.0])
    qM[0, 1] = np.array([4.0, 5.0, 6.0])
    qM[1, 1] = np.array([40.0, 50.0, 60.0])

    qP = gather_neighbor_traces(qM, trace, use_numba=False)

    np.testing.assert_array_equal(qP[0, 0], qM[1, 0])
    np.testing.assert_array_equal(qP[1, 0], qM[0, 0])
    np.testing.assert_array_equal(qP[0, 1], qM[1, 1, ::-1])
    np.testing.assert_array_equal(qP[1, 1], qM[0, 1, ::-1])


def test_build_reference_cache_rejects_invalid_full_variant_combinations():
    with pytest.raises(ValueError, match="only supports table1"):
        build_reference_cache(order=4, table="table2", validate=True, sbp_variant="full-raw")

    with pytest.raises(ValueError, match="only supports table1"):
        build_reference_cache(order=4, table="table2", validate=True, sbp_variant="full-orth")

    with pytest.raises(ValueError, match="requires n_face == order \\+ 1"):
        build_reference_cache(order=4, table="table1", n_face=6, validate=True, sbp_variant="full-raw")

    with pytest.raises(ValueError, match="sbp_variant must be"):
        build_reference_cache(order=4, table="table1", validate=True, sbp_variant="bogus")
