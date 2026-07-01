import numpy as np

from simplex_dg.geometry import build_geometry_cache, dual_basis_residuals
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache


def _build_case(ndivs=4, order=4, table="table1"):
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    geom = build_geometry_cache(mesh, ref)

    return mesh, ref, geom


def test_geometry_cache_shapes():
    mesh, ref, geom = _build_case(ndivs=4, order=4)

    K = mesh.elements.shape[0]
    Np = ref.rs.shape[0]
    Nf = ref.edge_rules[1].n_points

    assert geom.X.shape == (K, Np, 3)
    assert geom.Xr.shape == (K, Np, 3)
    assert geom.Xs.shape == (K, Np, 3)

    assert geom.normal.shape == (K, Np, 3)
    assert geom.sqrt_g.shape == (K, Np)

    assert geom.X_face.shape == (K, 3, Nf, 3)
    assert geom.face_tangent.shape == (K, 3, Nf, 3)
    assert geom.face_jacobian.shape == (K, 3, Nf)
    assert geom.face_normal.shape == (K, 3, Nf, 3)
    assert geom.face_conormal.shape == (K, 3, Nf, 3)


def test_geometry_nodes_are_on_sphere():
    mesh, ref, geom = _build_case(ndivs=4, order=4)

    rv = np.linalg.norm(geom.X, axis=2)
    rf = np.linalg.norm(geom.X_face, axis=3)

    assert np.allclose(rv, mesh.radius, atol=1e-10, rtol=1e-10)
    assert np.allclose(rf, mesh.radius, atol=1e-10, rtol=1e-10)


def test_surface_metric_positive():
    mesh, ref, geom = _build_case(ndivs=4, order=4)

    assert np.all(geom.sqrt_g > 0.0)
    assert np.all(geom.gdet > 0.0)
    assert np.all(geom.face_jacobian > 0.0)


def test_normals_are_unit_and_tangent_orthogonal():
    mesh, ref, geom = _build_case(ndivs=4, order=4)

    normal_norm = np.linalg.norm(geom.normal, axis=2)

    assert np.allclose(normal_norm, 1.0, atol=1e-10, rtol=1e-10)

    assert np.max(np.abs(np.sum(geom.normal * geom.Xr, axis=2))) < 1e-9
    assert np.max(np.abs(np.sum(geom.normal * geom.Xs, axis=2))) < 1e-9


def test_dual_basis_identity():
    mesh, ref, geom = _build_case(ndivs=4, order=4)

    res = dual_basis_residuals(geom)

    assert res["max_abs_grad_r_dot_Xr_minus_1"] < 1e-9
    assert res["max_abs_grad_r_dot_Xs"] < 1e-9
    assert res["max_abs_grad_s_dot_Xr"] < 1e-9
    assert res["max_abs_grad_s_dot_Xs_minus_1"] < 1e-9


def test_face_geometry_orthogonality():
    mesh, ref, geom = _build_case(ndivs=4, order=4)

    tf = geom.face_tangent / geom.face_jacobian[:, :, :, None]

    assert np.allclose(np.linalg.norm(tf, axis=3), 1.0, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.linalg.norm(geom.face_normal, axis=3), 1.0, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.linalg.norm(geom.face_conormal, axis=3), 1.0, atol=1e-10, rtol=1e-10)

    assert np.max(np.abs(np.sum(tf * geom.face_normal, axis=3))) < 1e-9
    assert np.max(np.abs(np.sum(tf * geom.face_conormal, axis=3))) < 1e-9
    assert np.max(np.abs(np.sum(geom.face_normal * geom.face_conormal, axis=3))) < 1e-9


def test_table2_geometry_cache_also_builds():
    ref = build_reference_cache(order=4, table="table2")
    mesh = build_octa_sphere_mesh(ndivs=2, radius=1.0)
    geom = build_geometry_cache(mesh, ref)

    assert geom.X.shape[0] == mesh.elements.shape[0]
    assert geom.X.shape[1] == ref.rs.shape[0]
