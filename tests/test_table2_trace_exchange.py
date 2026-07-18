from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.trace import TraceCache, build_trace_cache, evaluate_face_traces, gather_neighbor_traces, pair_face_traces


@pytest.fixture(scope="module")
def table2_exchange_fixture():
    ref = build_reference_cache(
        order=4,
        table="table2",
        n_face=5,
        validate=True,
    )
    mesh = build_octa_sphere_mesh(ndivs=1, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref=ref, conn=conn, validate=True)
    return mesh, ref, conn, geom, trace


@pytest.fixture(scope="module")
def table2_exchange_geom_fixture():
    ref = build_reference_cache(
        order=4,
        table="table2",
        n_face=5,
        validate=True,
    )
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref=ref, conn=conn, validate=True)
    return mesh, ref, conn, geom, trace


def _iter_unique_interior_faces(trace):
    for k in range(trace.n_elements):
        for f in range(trace.n_faces):
            if trace.is_boundary[k, f]:
                continue
            nbr = int(trace.neighbor_elements[k, f])
            nbr_f = int(trace.neighbor_faces[k, f])
            if k < nbr:
                yield k, f, nbr, nbr_f


def test_table2_connectivity_is_reciprocal(table2_exchange_fixture):
    mesh, ref, conn, geom, trace = table2_exchange_fixture

    failures = []

    for k, f, nbr, nbr_f in _iter_unique_interior_faces(trace):
        if trace.neighbor_elements[nbr, nbr_f] != k:
            failures.append((k, f, nbr, nbr_f, "element"))
        if trace.neighbor_faces[nbr, nbr_f] != f:
            failures.append((k, f, nbr, nbr_f, "face"))
        if trace.face_flip[nbr, nbr_f] != trace.face_flip[k, f]:
            failures.append((k, f, nbr, nbr_f, "flip"))

    assert not failures, failures


def test_table2_neighbor_gathering_uses_face_and_flip(table2_exchange_fixture):
    mesh, ref, conn, geom, trace = table2_exchange_fixture

    qM = np.zeros((trace.n_elements, trace.n_faces, trace.n_face_points))
    for k in range(trace.n_elements):
        for f in range(trace.n_faces):
            qM[k, f, :] = 1000.0 * k + 100.0 * f + np.arange(trace.n_face_points, dtype=float)

    qP = gather_neighbor_traces(qM, trace, use_numba=False)

    for k, f, nbr, nbr_f in _iter_unique_interior_faces(trace):
        expected = qM[nbr, nbr_f, ::-1] if trace.face_flip[k, f] else qM[nbr, nbr_f, :]
        np.testing.assert_array_equal(qP[k, f, :], expected)

        expected_back = qM[k, f, ::-1] if trace.face_flip[nbr, nbr_f] else qM[k, f, :]
        np.testing.assert_array_equal(qP[nbr, nbr_f, :], expected_back)


def test_table2_asymmetric_data_detects_face_flip(table2_exchange_fixture):
    mesh, ref, conn, geom, trace = table2_exchange_fixture

    assert np.any(trace.face_flip)

    qM = np.zeros((trace.n_elements, trace.n_faces, trace.n_face_points))
    pattern = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    for k in range(trace.n_elements):
        for f in range(trace.n_faces):
            qM[k, f, :] = pattern + (100.0 * k + 10.0 * f)

    qP = gather_neighbor_traces(qM, trace, use_numba=False)

    found_flipped = False

    for k, f, nbr, nbr_f in _iter_unique_interior_faces(trace):
        if trace.face_flip[k, f]:
            found_flipped = True
            np.testing.assert_array_equal(qP[k, f, :], qM[nbr, nbr_f, ::-1])

    assert found_flipped


def test_table2_neighbor_gathering_identity_path_with_mock_cache():
    trace = TraceCache(
        n_elements=2,
        n_points=16,
        n_faces=3,
        n_face_points=5,
        face_interp=np.zeros((3, 5, 16)),
        face_weights=np.ones((3, 5)) / 5.0,
        neighbor_elements=np.array([[1, -1, -1], [-1, -1, 0]], dtype=int),
        neighbor_faces=np.array([[2, -1, -1], [-1, -1, 0]], dtype=int),
        is_boundary=np.array([[False, True, True], [True, True, False]], dtype=bool),
        face_flip=np.array([[False, False, False], [False, False, False]], dtype=bool),
    )

    qM = np.full((2, 3, 5), np.nan)
    qM[0, 0, :] = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    qM[1, 2, :] = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    qM[0, 2, :] = np.array([7.0, 14.0, 21.0, 28.0, 35.0])
    qM[1, 0, :] = np.array([3.0, 6.0, 9.0, 12.0, 15.0])

    qP = gather_neighbor_traces(qM, trace, use_numba=False)

    np.testing.assert_array_equal(qP[0, 0, :], qM[1, 2, :])
    np.testing.assert_array_equal(qP[1, 2, :], qM[0, 0, :])


def test_table2_shared_face_coordinates_match(table2_exchange_geom_fixture):
    mesh, ref, conn, geom, trace = table2_exchange_geom_fixture

    max_err = 0.0

    for k, f, nbr, nbr_f in _iter_unique_interior_faces(trace):
        x_minus = geom.X_face[k, f]
        x_plus_raw = geom.X_face[nbr, nbr_f]
        x_plus = x_plus_raw[::-1] if trace.face_flip[k, f] else x_plus_raw

        np.testing.assert_allclose(x_minus, x_plus, atol=2e-12, rtol=2e-12)
        max_err = max(max_err, float(np.max(np.linalg.norm(x_minus - x_plus, axis=1))))

    assert max_err < 2e-12


def test_table2_global_field_is_continuous_on_shared_face_points(table2_exchange_geom_fixture):
    mesh, ref, conn, geom, trace = table2_exchange_geom_fixture

    q = 1.0 + 0.3 * geom.X[..., 0] - 0.2 * geom.X[..., 1] + 0.1 * geom.X[..., 2]
    traces = pair_face_traces(q, trace, use_numba=False)

    max_exact_shared_face_error = 0.0
    max_projected_trace_mismatch = 0.0

    for k, f, nbr, nbr_f in _iter_unique_interior_faces(trace):
        x_minus = geom.X_face[k, f]
        x_plus_raw = geom.X_face[nbr, nbr_f]
        x_plus = x_plus_raw[::-1] if trace.face_flip[k, f] else x_plus_raw

        q_face_exact_minus = 1.0 + 0.3 * x_minus[:, 0] - 0.2 * x_minus[:, 1] + 0.1 * x_minus[:, 2]
        q_face_exact_plus = 1.0 + 0.3 * x_plus[:, 0] - 0.2 * x_plus[:, 1] + 0.1 * x_plus[:, 2]

        np.testing.assert_allclose(q_face_exact_minus, q_face_exact_plus, atol=2e-13, rtol=2e-13)
        max_exact_shared_face_error = max(
            max_exact_shared_face_error,
            float(np.max(np.abs(q_face_exact_minus - q_face_exact_plus))),
        )

        max_projected_trace_mismatch = max(
            max_projected_trace_mismatch,
            float(np.max(np.abs(traces.qM[k, f, :] - traces.qP[k, f, :]))),
        )

    assert max_exact_shared_face_error < 2e-13
    assert max_projected_trace_mismatch < 5e-4


def test_table2_numpy_numba_trace_parity(table2_exchange_fixture):
    pytest.importorskip("numba")

    mesh, ref, conn, geom, trace = table2_exchange_fixture

    rng = np.random.default_rng(20260718)
    q = rng.standard_normal((trace.n_elements, trace.n_points))

    qM_numpy = evaluate_face_traces(q, trace, use_numba=False)
    qP_numpy = gather_neighbor_traces(qM_numpy, trace, use_numba=False)

    qM_numba = evaluate_face_traces(q, trace, use_numba=True)
    qP_numba = gather_neighbor_traces(qM_numba, trace, use_numba=True)

    np.testing.assert_allclose(qM_numba, qM_numpy, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(qP_numba, qP_numpy, atol=1e-14, rtol=1e-14)
