from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import (
    build_geometry_cache,
    dual_basis_residuals,
    map_reference_to_sphere_element,
)
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache


@pytest.fixture(scope="module")
def table2_geometry_case():
    ref = build_reference_cache(
        order=4,
        table="table2",
        n_face=5,
        validate=True,
    )
    mesh = build_octa_sphere_mesh(
        ndivs=4,
        radius=1.0,
    )
    geom = build_geometry_cache(
        mesh=mesh,
        ref=ref,
        validate=True,
    )
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    return mesh, ref, geom, conn


def test_table2_geometry_dimensions_and_finiteness(table2_geometry_case):
    mesh, ref, geom, conn = table2_geometry_case

    k_elements = mesh.elements.shape[0]

    assert ref.rs.shape == (16, 2)
    assert ref.edge_rules[1].n_points == 5
    assert geom.element_vertices.shape == (k_elements, 3, 3)
    assert geom.X.shape == (k_elements, 16, 3)
    assert geom.Xr.shape == (k_elements, 16, 3)
    assert geom.Xs.shape == (k_elements, 16, 3)
    assert geom.normal.shape == (k_elements, 16, 3)
    assert geom.sqrt_g.shape == (k_elements, 16)
    assert geom.g11.shape == (k_elements, 16)
    assert geom.g12.shape == (k_elements, 16)
    assert geom.g22.shape == (k_elements, 16)
    assert geom.gdet.shape == (k_elements, 16)
    assert geom.ginv11.shape == (k_elements, 16)
    assert geom.ginv12.shape == (k_elements, 16)
    assert geom.ginv22.shape == (k_elements, 16)
    assert geom.grad_r.shape == (k_elements, 16, 3)
    assert geom.grad_s.shape == (k_elements, 16, 3)
    assert geom.X_face.shape == (k_elements, 3, 5, 3)
    assert geom.face_tangent.shape == (k_elements, 3, 5, 3)
    assert geom.face_jacobian.shape == (k_elements, 3, 5)
    assert geom.face_normal.shape == (k_elements, 3, 5, 3)
    assert geom.face_conormal.shape == (k_elements, 3, 5, 3)
    assert conn.interior_faces.shape[1] == 4

    arrays = (
        geom.element_vertices,
        geom.X,
        geom.Xr,
        geom.Xs,
        geom.normal,
        geom.sqrt_g,
        geom.g11,
        geom.g12,
        geom.g22,
        geom.gdet,
        geom.ginv11,
        geom.ginv12,
        geom.ginv22,
        geom.grad_r,
        geom.grad_s,
        geom.X_face,
        geom.face_tangent,
        geom.face_jacobian,
        geom.face_normal,
        geom.face_conormal,
    )

    for array in arrays:
        assert np.all(np.isfinite(array))


def test_table2_geometry_points_lie_on_sphere(table2_geometry_case):
    mesh, ref, geom, conn = table2_geometry_case

    volume_radius_error = np.max(np.abs(np.linalg.norm(geom.X, axis=2) - mesh.radius))
    face_radius_error = np.max(np.abs(np.linalg.norm(geom.X_face, axis=3) - mesh.radius))

    assert volume_radius_error < 5e-14
    assert face_radius_error < 5e-14


def test_table2_geometry_normals_match_radial_direction(table2_geometry_case):
    mesh, ref, geom, conn = table2_geometry_case

    radial_volume = geom.X / mesh.radius
    radial_face = geom.X_face / mesh.radius

    np.testing.assert_allclose(geom.normal, radial_volume, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(geom.face_normal, radial_face, atol=2e-13, rtol=2e-13)

    assert np.min(np.sum(geom.normal * radial_volume, axis=2)) > 0.0
    assert np.min(np.sum(geom.face_normal * radial_face, axis=3)) > 0.0


def test_table2_geometry_analytic_derivatives_match_finite_difference(table2_geometry_case):
    mesh, ref, geom, conn = table2_geometry_case

    rs_test = np.array(
        [
            [-1.0 / 3.0, -1.0 / 3.0],
            [-0.60, -0.20],
            [-0.20, -0.60],
            [-0.45, -0.35],
        ]
    )
    element_ids = [0, geom.element_vertices.shape[0] // 4, geom.element_vertices.shape[0] // 2, geom.element_vertices.shape[0] - 1]
    h = 1e-6

    max_xr_fd_error = 0.0
    max_xs_fd_error = 0.0

    for elem_id in element_ids:
        vertices = geom.element_vertices[elem_id]
        _, xr, xs = map_reference_to_sphere_element(
            rs=rs_test,
            vertices=vertices,
            radius=mesh.radius,
        )

        rs_r_plus = rs_test.copy()
        rs_r_minus = rs_test.copy()
        rs_s_plus = rs_test.copy()
        rs_s_minus = rs_test.copy()

        rs_r_plus[:, 0] += h
        rs_r_minus[:, 0] -= h
        rs_s_plus[:, 1] += h
        rs_s_minus[:, 1] -= h

        x_r_plus, _, _ = map_reference_to_sphere_element(rs_r_plus, vertices, mesh.radius)
        x_r_minus, _, _ = map_reference_to_sphere_element(rs_r_minus, vertices, mesh.radius)
        x_s_plus, _, _ = map_reference_to_sphere_element(rs_s_plus, vertices, mesh.radius)
        x_s_minus, _, _ = map_reference_to_sphere_element(rs_s_minus, vertices, mesh.radius)

        xr_fd = (x_r_plus - x_r_minus) / (2.0 * h)
        xs_fd = (x_s_plus - x_s_minus) / (2.0 * h)

        max_xr_fd_error = max(max_xr_fd_error, float(np.max(np.abs(xr - xr_fd))))
        max_xs_fd_error = max(max_xs_fd_error, float(np.max(np.abs(xs - xs_fd))))

        np.testing.assert_allclose(xr, xr_fd, atol=5e-9, rtol=5e-9)
        np.testing.assert_allclose(xs, xs_fd, atol=5e-9, rtol=5e-9)

    assert max_xr_fd_error < 5e-9
    assert max_xs_fd_error < 5e-9


def test_table2_geometry_surface_jacobian_and_metric_consistency(table2_geometry_case):
    mesh, ref, geom, conn = table2_geometry_case

    sqrt_g_cross = np.linalg.norm(np.cross(geom.Xr, geom.Xs), axis=2)
    gdet_from_metric = geom.g11 * geom.g22 - geom.g12**2

    np.testing.assert_allclose(geom.sqrt_g, sqrt_g_cross, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(geom.gdet, gdet_from_metric, atol=5e-13, rtol=5e-13)
    np.testing.assert_allclose(geom.gdet, geom.sqrt_g**2, atol=5e-13, rtol=5e-13)

    assert np.all(geom.sqrt_g > 0.0)
    assert np.all(geom.g11 > 0.0)
    assert np.all(geom.g22 > 0.0)
    assert np.all(geom.gdet > 0.0)


def test_table2_geometry_inverse_metric_and_dual_basis_identities(table2_geometry_case):
    mesh, ref, geom, conn = table2_geometry_case

    residuals = [
        geom.ginv11 * geom.g11 + geom.ginv12 * geom.g12 - 1.0,
        geom.ginv11 * geom.g12 + geom.ginv12 * geom.g22,
        geom.ginv12 * geom.g11 + geom.ginv22 * geom.g12,
        geom.ginv12 * geom.g12 + geom.ginv22 * geom.g22 - 1.0,
    ]
    max_inverse_metric_error = max(float(np.max(np.abs(residual))) for residual in residuals)

    dual = dual_basis_residuals(geom)
    max_dual_basis_error = max(dual.values())
    max_dual_basis_normal_component = max(
        float(np.max(np.abs(np.sum(geom.grad_r * geom.normal, axis=2)))),
        float(np.max(np.abs(np.sum(geom.grad_s * geom.normal, axis=2)))),
    )

    assert max_inverse_metric_error < 2e-12
    assert max_dual_basis_error < 2e-12
    assert max_dual_basis_normal_component < 2e-12


def test_table2_geometry_face_jacobian_and_orthonormal_frame(table2_geometry_case):
    mesh, ref, geom, conn = table2_geometry_case

    face_jacobian_from_tangent = np.linalg.norm(geom.face_tangent, axis=3)
    unit_tangent = geom.face_tangent / geom.face_jacobian[..., None]
    conormal_from_definition = np.cross(unit_tangent, geom.face_normal)

    np.testing.assert_allclose(geom.face_jacobian, face_jacobian_from_tangent, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(geom.face_conormal, conormal_from_definition, atol=2e-13, rtol=2e-13)

    tangent_norm_error = float(np.max(np.abs(np.linalg.norm(unit_tangent, axis=3) - 1.0)))
    face_normal_norm_error = float(np.max(np.abs(np.linalg.norm(geom.face_normal, axis=3) - 1.0)))
    conormal_norm_error = float(np.max(np.abs(np.linalg.norm(geom.face_conormal, axis=3) - 1.0)))
    tangent_normal_dot = float(np.max(np.abs(np.sum(unit_tangent * geom.face_normal, axis=3))))
    tangent_conormal_dot = float(np.max(np.abs(np.sum(unit_tangent * geom.face_conormal, axis=3))))
    normal_conormal_dot = float(np.max(np.abs(np.sum(geom.face_normal * geom.face_conormal, axis=3))))

    assert np.all(geom.face_jacobian > 0.0)
    assert tangent_norm_error < 2e-13
    assert face_normal_norm_error < 2e-13
    assert conormal_norm_error < 2e-13
    assert tangent_normal_dot < 2e-13
    assert tangent_conormal_dot < 2e-13
    assert normal_conormal_dot < 2e-13


def test_table2_geometry_conormal_points_outward(table2_geometry_case):
    mesh, ref, geom, conn = table2_geometry_case

    rs_centroid = np.array([[-1.0 / 3.0, -1.0 / 3.0]])
    max_outward_dot = -np.inf

    for k in range(geom.element_vertices.shape[0]):
        x_centroid, _, _ = map_reference_to_sphere_element(
            rs=rs_centroid,
            vertices=geom.element_vertices[k],
            radius=mesh.radius,
        )
        x_centroid = x_centroid[0]

        for f in range(3):
            d_in = x_centroid[None, :] - geom.X_face[k, f]
            d_in_tangent = d_in - np.sum(d_in * geom.face_normal[k, f], axis=1, keepdims=True) * geom.face_normal[k, f]
            inward_norm = np.linalg.norm(d_in_tangent, axis=1)
            mask = inward_norm > 1e-12

            if not np.any(mask):
                continue

            inward_tangent_unit = d_in_tangent[mask] / inward_norm[mask, None]
            dots = np.sum(geom.face_conormal[k, f][mask] * inward_tangent_unit, axis=1)
            max_outward_dot = max(max_outward_dot, float(np.max(dots)))

    assert max_outward_dot < -1e-6


def test_table2_sphere_area_diagnostic_improves_with_refinement():
    errors = []

    for ndivs in (1, 2, 4, 8):
        ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)
        mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
        geom = build_geometry_cache(mesh, ref, validate=True)

        area_discrete = float(np.sum(ref.area * ref.weights[None, :] * geom.sqrt_g))
        area_exact = 4.0 * np.pi * mesh.radius**2
        relative_error = abs(area_discrete - area_exact) / area_exact

        assert np.isfinite(area_discrete)
        assert area_discrete > 0.0
        errors.append(relative_error)

    assert errors[-1] < errors[0]

    small_slack = 5e-2
    for prev, curr in zip(errors, errors[1:]):
        assert curr <= prev * (1.0 + small_slack)
