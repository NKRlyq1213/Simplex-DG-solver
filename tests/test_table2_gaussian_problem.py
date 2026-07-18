from __future__ import annotations

import numpy as np
import pytest

from examples import step9_gaussian_convergence as step9
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.problems import (
    exact_gaussian_solid_body,
    gaussian_center_solid_body,
    gaussian_on_sphere,
)
from simplex_dg.reference import build_reference_cache


@pytest.fixture(scope="module")
def table2_geom():
    ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)
    mesh = build_octa_sphere_mesh(ndivs=1, radius=1.0)
    geom = build_geometry_cache(mesh, ref, validate=True)
    return mesh, ref, geom


def test_table2_exact_solution_matches_initial_gaussian_at_t0(table2_geom):
    mesh, ref, geom = table2_geom
    center0 = (mesh.radius, 0.0, 0.0)

    q0 = gaussian_on_sphere(geom.X, center=center0, radius=mesh.radius, sigma=0.35, amplitude=1.0)
    qe = exact_gaussian_solid_body(
        geom.X,
        t=0.0,
        radius=mesh.radius,
        sigma=0.35,
        amplitude=1.0,
        center0=center0,
        omega=(0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(qe, q0, atol=2e-14, rtol=2e-14)


def test_table2_gaussian_center_remains_on_sphere():
    radius = 1.0
    alpha0 = -np.pi / 4.0
    u0 = 2.0 * np.pi / 10.0
    axis = step9.rotation_axis_from_alpha0(alpha0)
    omega = tuple(u0 * component for component in axis)

    for t in (0.0, 0.1, 1.0, 5.0, 10.0, 12.0):
        center = gaussian_center_solid_body(
            t=t,
            radius=radius,
            center0=(radius, 0.0, 0.0),
            omega=omega,
        )
        assert np.linalg.norm(center) == pytest.approx(radius, abs=2e-14, rel=2e-14)


def test_table2_full_period_center_and_exact_solution_close(table2_geom):
    mesh, ref, geom = table2_geom
    u0 = 2.0 * np.pi / 10.0
    alpha0 = -np.pi / 4.0
    axis = step9.rotation_axis_from_alpha0(alpha0)
    omega = tuple(u0 * component for component in axis)
    period = 2.0 * np.pi / np.linalg.norm(omega)
    center0 = np.array([mesh.radius, 0.0, 0.0], dtype=float)

    center_t = gaussian_center_solid_body(
        t=period,
        radius=mesh.radius,
        center0=center0,
        omega=omega,
    )
    q0 = gaussian_on_sphere(geom.X, center=center0, radius=mesh.radius, sigma=0.35, amplitude=1.0)
    q_period = exact_gaussian_solid_body(
        geom.X,
        t=period,
        radius=mesh.radius,
        sigma=0.35,
        amplitude=1.0,
        center0=center0,
        omega=omega,
    )

    np.testing.assert_allclose(center_t, center0, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(q_period, q0, atol=2e-13, rtol=2e-13)


def test_resolve_sigma_physical_supports_default_and_override():
    assert step9.resolve_sigma_physical(radius=2.0, sigma_angle=0.35, sigma_physical=None) == pytest.approx(0.7)
    assert step9.resolve_sigma_physical(radius=2.0, sigma_angle=0.35, sigma_physical=0.5) == pytest.approx(0.5)


def test_gaussian_problem_rejects_invalid_sigma_and_radius():
    X = np.array([[[1.0, 0.0, 0.0]]], dtype=float)

    with pytest.raises(ValueError, match="sigma must be positive"):
        gaussian_on_sphere(X, radius=1.0, sigma=0.0)

    with pytest.raises(ValueError, match="radius must be positive"):
        gaussian_on_sphere(X, radius=0.0, sigma=0.35)

    with pytest.raises(ValueError, match="radius must be positive"):
        step9.resolve_sigma_physical(radius=0.0, sigma_angle=0.35)

    with pytest.raises(ValueError, match="sigma_physical must be positive"):
        step9.resolve_sigma_physical(radius=1.0, sigma_angle=0.35, sigma_physical=0.0)
