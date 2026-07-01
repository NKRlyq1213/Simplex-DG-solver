import numpy as np

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.trace import (
    build_trace_cache,
    check_constant_trace_consistency,
    evaluate_face_traces,
    gather_neighbor_traces,
    max_interior_trace_mismatch,
    pair_face_traces,
)


def _build_case(ndivs=4, order=4, table="table1"):
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    return mesh, ref, conn, geom, trace


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


def test_trace_cache_shapes():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4)

    K = mesh.elements.shape[0]
    Np = ref.rs.shape[0]
    Nf = ref.edge_rules[1].n_points

    assert trace.face_interp.shape == (3, Nf, Np)
    assert trace.face_weights.shape == (3, Nf)
    assert trace.neighbor_elements.shape == (K, 3)
    assert trace.neighbor_faces.shape == (K, 3)
    assert trace.face_flip.shape == (K, 3)


def test_evaluate_face_traces_shape():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4)

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    qM = evaluate_face_traces(q, trace)

    assert qM.shape == (mesh.elements.shape[0], 3, ref.edge_rules[1].n_points)


def test_gather_neighbor_traces_shape():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4)

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    qM = evaluate_face_traces(q, trace)
    qP = gather_neighbor_traces(qM, trace)

    assert qP.shape == qM.shape


def test_pair_face_traces_shape():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4)

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    traces = pair_face_traces(q, trace)

    assert traces.qM.shape == (mesh.elements.shape[0], 3, ref.edge_rules[1].n_points)
    assert traces.qP.shape == traces.qM.shape


def test_constant_trace_consistency_table1():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4, table="table1")

    mismatch = check_constant_trace_consistency(trace, value=3.25)

    assert mismatch < 1e-10


def test_constant_trace_consistency_table2():
    mesh, ref, conn, geom, trace = _build_case(ndivs=2, order=4, table="table2")

    mismatch = check_constant_trace_consistency(trace, value=-2.0)

    assert mismatch < 1e-10


def test_exact_face_geometry_continuity_after_neighbor_gather():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4, table="table1")

    XP = _gather_exact_face_geometry(geom, trace)
    interior = ~trace.is_boundary

    mismatch = np.max(np.linalg.norm(geom.X_face[interior] - XP[interior], axis=1))

    assert mismatch < 1e-12


def test_projected_coordinate_field_x_component_reasonable():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4, table="table1")

    q = geom.X[:, :, 0]
    traces = pair_face_traces(q, trace)
    mismatch = max_interior_trace_mismatch(traces, trace)

    # Sphere coordinates are not polynomial functions in reference coordinates.
    # Left/right traces are independent polynomial projections, so this is not
    # expected to be machine precision.
    assert mismatch < 1e-3


def test_projected_coordinate_field_y_component_reasonable():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4, table="table1")

    q = geom.X[:, :, 1]
    traces = pair_face_traces(q, trace)
    mismatch = max_interior_trace_mismatch(traces, trace)

    assert mismatch < 1e-3


def test_projected_coordinate_field_z_component_reasonable():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4, table="table1")

    q = geom.X[:, :, 2]
    traces = pair_face_traces(q, trace)
    mismatch = max_interior_trace_mismatch(traces, trace)

    assert mismatch < 1e-3


def test_numba_and_numpy_trace_paths_agree():
    mesh, ref, conn, geom, trace = _build_case(ndivs=4, order=4, table="table1")

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    traces_np = pair_face_traces(q, trace, use_numba=False)
    traces_nb = pair_face_traces(q, trace, use_numba=True)

    assert np.allclose(traces_np.qM, traces_nb.qM, atol=1e-12, rtol=1e-12)
    assert np.allclose(traces_np.qP, traces_nb.qP, atol=1e-12, rtol=1e-12)
