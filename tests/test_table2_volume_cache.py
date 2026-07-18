from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import apply_reference_operator, build_volume_rhs_cache


OMEGA = np.array([0.3, -0.2, 0.7])


@pytest.fixture(scope="module")
def table2_volume_case():
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
        mesh,
        ref,
        validate=True,
    )
    volume = build_volume_rhs_cache(
        ref,
        geom,
        omega=OMEGA,
        project_velocity=True,
        validate=True,
    )
    return mesh, ref, geom, volume


def test_table2_volume_cache_dimensions_and_finiteness(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    k_elements = mesh.elements.shape[0]

    assert k_elements == 128
    assert volume.n_elements == k_elements
    assert volume.n_points == 16

    assert volume.Dr.shape == (16, 16)
    assert volume.Ds.shape == (16, 16)
    assert volume.sqrt_g.shape == (k_elements, 16)
    assert volume.velocity.shape == (k_elements, 16, 3)
    assert volume.speed.shape == (k_elements, 16)
    assert volume.alpha.shape == (k_elements, 16)
    assert volume.beta.shape == (k_elements, 16)
    assert volume.Dr_alpha.shape == (k_elements, 16)
    assert volume.Ds_beta.shape == (k_elements, 16)
    assert volume.div_velocity.shape == (k_elements, 16)

    for array in (
        volume.Dr,
        volume.Ds,
        volume.sqrt_g,
        volume.velocity,
        volume.speed,
        volume.alpha,
        volume.beta,
        volume.Dr_alpha,
        volume.Ds_beta,
        volume.div_velocity,
    ):
        assert np.all(np.isfinite(array))

    np.testing.assert_allclose(volume.sqrt_g, geom.sqrt_g, atol=0.0, rtol=0.0)

    assert np.isfinite(volume.max_speed)
    assert volume.max_speed > 0.0


def test_table2_volume_speed_and_max_speed_match_velocity_norm(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    speed_exact = np.linalg.norm(volume.velocity, axis=2)

    np.testing.assert_allclose(volume.speed, speed_exact, atol=2e-15, rtol=2e-15)
    assert volume.max_speed == pytest.approx(np.max(speed_exact), abs=2e-15, rel=2e-15)


def test_table2_volume_solid_body_velocity_definition(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    expected = np.cross(np.broadcast_to(OMEGA, geom.X.shape), geom.X)

    np.testing.assert_allclose(volume.velocity, expected, atol=2e-14, rtol=2e-14)


def test_table2_volume_tangential_projection_removes_radial_component(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    u_tangent = np.cross(np.broadcast_to(OMEGA, geom.X.shape), geom.X)
    raw_velocity = u_tangent + 0.65 * geom.normal

    projected_cache = build_volume_rhs_cache(
        ref,
        geom,
        velocity=raw_velocity,
        project_velocity=True,
    )
    unprojected_cache = build_volume_rhs_cache(
        ref,
        geom,
        velocity=raw_velocity,
        project_velocity=False,
        validate=False,
    )

    np.testing.assert_allclose(projected_cache.velocity, u_tangent, atol=2e-14, rtol=2e-14)

    tangent_error = float(np.max(np.abs(np.sum(projected_cache.velocity * geom.normal, axis=2))))
    unprojected_radial = float(np.max(np.abs(np.sum(unprojected_cache.velocity * geom.normal, axis=2))))

    assert tangent_error < 2e-14
    assert unprojected_radial > 0.1


def test_table2_volume_mapped_coefficients_match_definitions(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    alpha_expected = geom.sqrt_g * np.sum(volume.velocity * geom.grad_r, axis=2)
    beta_expected = geom.sqrt_g * np.sum(volume.velocity * geom.grad_s, axis=2)

    np.testing.assert_allclose(volume.alpha, alpha_expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(volume.beta, beta_expected, atol=2e-14, rtol=2e-14)


def test_table2_volume_contravariant_velocity_reconstruction(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    ur = volume.alpha / volume.sqrt_g
    us = volume.beta / volume.sqrt_g
    velocity_reconstructed = ur[..., None] * geom.Xr + us[..., None] * geom.Xs

    np.testing.assert_allclose(
        velocity_reconstructed,
        volume.velocity,
        atol=5e-13,
        rtol=5e-13,
    )


def test_table2_volume_derivative_cache_and_divergence_definitions(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    dr_alpha_expected = apply_reference_operator(volume.Dr, volume.alpha)
    ds_beta_expected = apply_reference_operator(volume.Ds, volume.beta)
    div_velocity_expected = (dr_alpha_expected + ds_beta_expected) / volume.sqrt_g

    np.testing.assert_allclose(volume.Dr_alpha, dr_alpha_expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(volume.Ds_beta, ds_beta_expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(volume.div_velocity, div_velocity_expected, atol=2e-14, rtol=2e-14)


def test_table2_volume_zero_velocity_produces_zero_cache_fields():
    ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    geom = build_geometry_cache(mesh, ref, validate=True)

    zero_velocity = np.zeros_like(geom.X)
    zero_cache = build_volume_rhs_cache(
        ref,
        geom,
        velocity=zero_velocity,
        project_velocity=True,
    )

    np.testing.assert_allclose(zero_cache.velocity, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zero_cache.speed, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zero_cache.alpha, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zero_cache.beta, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zero_cache.Dr_alpha, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zero_cache.Ds_beta, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zero_cache.div_velocity, 0.0, atol=0.0, rtol=0.0)
    assert zero_cache.max_speed == 0.0


def test_table2_volume_velocity_scaling_identities(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    scale = -1.75
    scaled = build_volume_rhs_cache(
        ref,
        geom,
        omega=scale * OMEGA,
        project_velocity=True,
        validate=True,
    )

    np.testing.assert_allclose(scaled.velocity, scale * volume.velocity, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(scaled.alpha, scale * volume.alpha, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(scaled.beta, scale * volume.beta, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(scaled.Dr_alpha, scale * volume.Dr_alpha, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(scaled.Ds_beta, scale * volume.Ds_beta, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(scaled.div_velocity, scale * volume.div_velocity, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(scaled.speed, abs(scale) * volume.speed, atol=2e-14, rtol=2e-14)
    assert scaled.max_speed == pytest.approx(abs(scale) * volume.max_speed, abs=2e-14, rel=2e-14)
