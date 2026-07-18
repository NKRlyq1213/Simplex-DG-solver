from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.reference.basis import vandermonde2d
from simplex_dg.rhs import (
    build_surface_rhs_cache,
    build_volume_rhs_cache,
    compute_face_velocity,
    flux_id_from_name,
)
from simplex_dg.trace import build_trace_cache


OMEGA = np.array([0.3, -0.2, 0.7])


@pytest.fixture(scope="module")
def table2_surface_case():
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
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)
    volume = build_volume_rhs_cache(
        ref=ref,
        geom=geom,
        omega=OMEGA,
        project_velocity=True,
        validate=True,
    )
    surface = build_surface_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=OMEGA,
        flux_type="upwind",
        lf_alpha=1.0,
        project_velocity=True,
        validate=True,
    )
    return ref, mesh, conn, geom, trace, volume, surface


def test_table2_surface_cache_dimensions_and_finiteness(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    k_elements = mesh.elements.shape[0]

    assert surface.n_elements == k_elements
    assert surface.n_points == 16
    assert surface.n_faces == 3
    assert surface.n_face_points == 5
    assert surface.lift.shape == (3, 16, 5)
    assert surface.face_interp.shape == (3, 5, 16)
    assert surface.sqrt_g.shape == (k_elements, 16)
    assert surface.face_jacobian.shape == (k_elements, 3, 5)
    assert surface.face_velocity.shape == (k_elements, 3, 5, 3)
    assert surface.normal_velocity.shape == (k_elements, 3, 5)

    for array in (
        surface.lift,
        surface.face_interp,
        surface.sqrt_g,
        surface.face_jacobian,
        surface.face_velocity,
        surface.normal_velocity,
    ):
        assert np.all(np.isfinite(array))

    np.testing.assert_allclose(surface.face_interp, trace.face_interp, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(surface.sqrt_g, geom.sqrt_g, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(surface.face_jacobian, geom.face_jacobian, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("central", 0),
        ("upwind", 1),
        ("lf", 2),
        ("lax_friedrichs", 2),
        ("lax-friedrichs", 2),
    ],
)
def test_flux_id_from_name_aliases(name, expected):
    assert flux_id_from_name(name) == expected


def test_flux_id_from_name_rejects_invalid_name():
    with pytest.raises(ValueError, match="flux_type"):
        flux_id_from_name("bogus")


def test_build_surface_rhs_cache_rejects_negative_lf_alpha(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    with pytest.raises(ValueError, match="lf_alpha must be non-negative"):
        build_surface_rhs_cache(
            ref=ref,
            geom=geom,
            trace=trace,
            omega=OMEGA,
            flux_type="lf",
            lf_alpha=-1.0,
            validate=True,
        )


def test_table2_lift_matrices_match_direct_definition(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    for face_id in (1, 2, 3):
        f = face_id - 1
        edge = ref.edge_rules[face_id]
        v_face = vandermonde2d(ref.order, edge.rs[:, 0], edge.rs[:, 1])
        expected = (ref.V @ ref.Minv @ v_face.T) * edge.weights[None, :]

        np.testing.assert_allclose(surface.lift[f], expected, atol=2e-14, rtol=2e-14)


def test_table2_lift_weighted_adjoint_identity(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case
    rng = np.random.default_rng(20260718)
    q_volume = rng.standard_normal(ref.rs.shape[0])

    for face_id in (1, 2, 3):
        f = face_id - 1
        edge = ref.edge_rules[face_id]
        p_face = rng.standard_normal(edge.n_points)

        lhs = ref.area * np.dot(ref.weights * q_volume, surface.lift[f] @ p_face)
        rhs = np.dot(edge.weights * (surface.face_interp[f] @ q_volume), p_face)

        np.testing.assert_allclose(lhs, rhs, atol=5e-13, rtol=5e-13)


def test_table2_face_velocity_matches_solid_body_definition(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    expected = np.cross(np.broadcast_to(OMEGA, geom.X_face.shape), geom.X_face)

    np.testing.assert_allclose(surface.face_velocity, expected, atol=2e-14, rtol=2e-14)

    tangent_error = float(np.max(np.abs(np.sum(surface.face_velocity * geom.face_normal, axis=3))))
    assert tangent_error < 2e-14


def test_table2_face_velocity_projection_removes_radial_component(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    tangent_velocity = np.cross(np.broadcast_to(OMEGA, geom.X_face.shape), geom.X_face)
    raw_velocity = tangent_velocity + 0.65 * geom.face_normal

    projected = build_surface_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        velocity_face=raw_velocity,
        flux_type="upwind",
        lf_alpha=1.0,
        project_velocity=True,
        validate=True,
    )
    unprojected = build_surface_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        velocity_face=raw_velocity,
        flux_type="upwind",
        lf_alpha=1.0,
        project_velocity=False,
        validate=False,
    )

    np.testing.assert_allclose(projected.face_velocity, tangent_velocity, atol=2e-14, rtol=2e-14)

    tangent_error = float(np.max(np.abs(np.sum(projected.face_velocity * geom.face_normal, axis=3))))
    unprojected_radial = float(np.max(np.abs(np.sum(unprojected.face_velocity * geom.face_normal, axis=3))))

    assert tangent_error < 2e-14
    assert unprojected_radial > 0.1


def test_table2_normal_velocity_matches_conormal_dot_product(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    expected = np.sum(surface.face_velocity * geom.face_conormal, axis=3)
    np.testing.assert_allclose(surface.normal_velocity, expected, atol=2e-15, rtol=2e-15)


def test_compute_face_velocity_shape_guard(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    with pytest.raises(ValueError, match="velocity_face must have shape"):
        compute_face_velocity(geom, velocity_face=np.zeros((1, 1, 1, 3)))
