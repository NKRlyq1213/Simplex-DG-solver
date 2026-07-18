from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.trace import build_trace_cache, evaluate_face_traces, gather_neighbor_traces, pair_face_traces


@pytest.fixture(scope="module")
def table2_trace_fixture():
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


def test_table2_trace_cache_shapes(table2_trace_fixture):
    mesh, ref, conn, geom, trace = table2_trace_fixture

    k_elements = mesh.elements.shape[0]

    assert not np.any(trace.is_boundary)
    assert trace.n_points == 16
    assert trace.n_face_points == 5
    assert trace.n_faces == 3

    assert trace.face_interp.shape == (3, 5, 16)
    assert trace.face_weights.shape == (3, 5)
    assert trace.neighbor_elements.shape == (k_elements, 3)
    assert trace.neighbor_faces.shape == (k_elements, 3)
    assert trace.face_flip.shape == (k_elements, 3)
    assert trace.is_boundary.shape == (k_elements, 3)

    assert np.all(np.isfinite(trace.face_interp))
    assert np.all(np.isfinite(trace.face_weights))
    assert np.all(trace.face_weights > 0.0)

    np.testing.assert_allclose(trace.face_weights.sum(axis=1), 1.0, atol=2e-14, rtol=2e-14)


def test_table2_face_evaluation_matches_matrix_product(table2_trace_fixture):
    mesh, ref, conn, geom, trace = table2_trace_fixture

    rng = np.random.default_rng(12345)
    q = rng.standard_normal((trace.n_elements, trace.n_points))
    qM = evaluate_face_traces(q, trace, use_numba=False)

    assert qM.shape == (trace.n_elements, 3, 5)

    for f in range(trace.n_faces):
        expected = q @ trace.face_interp[f].T
        np.testing.assert_allclose(qM[:, f, :], expected, atol=1e-13, rtol=1e-13)


def test_table2_constant_trace(table2_trace_fixture):
    mesh, ref, conn, geom, trace = table2_trace_fixture

    constant = 2.75
    q = np.full((trace.n_elements, trace.n_points), constant)

    qM = evaluate_face_traces(q, trace, use_numba=False)
    qP = gather_neighbor_traces(qM, trace, use_numba=False)
    traces = pair_face_traces(q, trace, use_numba=False)

    np.testing.assert_allclose(qM, constant, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(traces.qM, constant, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(qP, constant, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(traces.qP, constant, atol=2e-13, rtol=2e-13)
