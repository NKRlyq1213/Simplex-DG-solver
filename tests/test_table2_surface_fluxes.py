from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    build_surface_rhs_cache,
    build_volume_rhs_cache,
    common_projected_line_velocity,
    numerical_flux,
    projected_interior_line_flux,
    projected_line_velocity,
)
from simplex_dg.trace import build_trace_cache, gather_neighbor_traces, pair_face_traces


FACE_DRDT = np.array([-2.0, 0.0, 2.0], dtype=float)
FACE_DSDT = np.array([2.0, -2.0, 0.0], dtype=float)


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
        omega=(0.3, -0.2, 0.7),
        project_velocity=True,
        validate=True,
    )
    surface = build_surface_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=(0.3, -0.2, 0.7),
        flux_type="upwind",
        lf_alpha=1.0,
        project_velocity=True,
        validate=True,
    )
    return ref, mesh, conn, geom, trace, volume, surface


def _state(geom):
    return 1.0 + 0.4 * geom.X[..., 0] - 0.3 * geom.X[..., 1] + 0.2 * geom.X[..., 2]


def _align_face_values(values, trace, k, f, nbr, nbr_f):
    aligned = values[nbr, nbr_f]
    if trace.face_flip[k, f]:
        aligned = aligned[::-1]
    return aligned


def test_table2_projected_line_velocity_matches_direct_definition(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    actual = projected_line_velocity(volume, surface)
    expected = np.empty_like(actual)

    for f in range(surface.n_faces):
        alpha_face = volume.alpha @ surface.face_interp[f].T
        beta_face = volume.beta @ surface.face_interp[f].T
        expected[:, f, :] = FACE_DSDT[f] * alpha_face - FACE_DRDT[f] * beta_face

    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_table2_projected_line_velocity_refines_toward_exact_physical_line_velocity():
    errors = []

    for ndivs in (1, 2, 4, 8):
        ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)
        mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
        conn = build_connectivity_cache_from_mesh(mesh, validate=True)
        geom = build_geometry_cache(mesh, ref, validate=True)
        trace = build_trace_cache(ref, conn, validate=True)
        volume = build_volume_rhs_cache(
            ref=ref,
            geom=geom,
            omega=(0.3, -0.2, 0.7),
            project_velocity=True,
            validate=True,
        )
        surface = build_surface_rhs_cache(
            ref=ref,
            geom=geom,
            trace=trace,
            omega=(0.3, -0.2, 0.7),
            flux_type="upwind",
            lf_alpha=1.0,
            project_velocity=True,
            validate=True,
        )

        projected = projected_line_velocity(volume, surface)
        exact = surface.face_jacobian * surface.normal_velocity
        errors.append(float(np.max(np.abs(projected - exact))))

    assert errors[-1] < errors[0]


def test_table2_common_projected_line_velocity_matches_direct_definition(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    a_m = projected_line_velocity(volume, surface)
    a_p = gather_neighbor_traces(
        a_m,
        trace,
        boundary_value=np.nan,
        use_numba=False,
    )
    expected = 0.5 * (a_m - a_p)
    actual = common_projected_line_velocity(volume, surface, trace, use_numba=False)

    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_table2_common_line_velocity_interface_antisymmetry(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    common = common_projected_line_velocity(volume, surface, trace, use_numba=False)
    max_sum = 0.0

    for k, f, nbr, nbr_f in conn.interior_faces:
        a_plus = _align_face_values(common, trace, k, f, nbr, nbr_f)
        np.testing.assert_allclose(common[k, f], -a_plus, atol=2e-13, rtol=2e-13)
        max_sum = max(max_sum, float(np.max(np.abs(common[k, f] + a_plus))))

    assert max_sum < 2e-13


def test_table2_projected_interior_line_flux_matches_direct_definition(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    q = _state(geom)
    actual = projected_interior_line_flux(q, volume, surface)
    expected = np.empty_like(actual)

    alpha_q = volume.alpha * q
    beta_q = volume.beta * q

    for f in range(surface.n_faces):
        alpha_q_face = alpha_q @ surface.face_interp[f].T
        beta_q_face = beta_q @ surface.face_interp[f].T
        expected[:, f, :] = FACE_DSDT[f] * alpha_q_face - FACE_DRDT[f] * beta_q_face

    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_table2_projected_product_ordering_guard(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case
    rng = np.random.default_rng(314159)
    q = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))

    correct = projected_interior_line_flux(q, volume, surface)
    wrong = np.empty_like(correct)

    for f in range(surface.n_faces):
        alpha_face = volume.alpha @ surface.face_interp[f].T
        beta_face = volume.beta @ surface.face_interp[f].T
        q_face = q @ surface.face_interp[f].T
        wrong[:, f, :] = FACE_DSDT[f] * alpha_face * q_face - FACE_DRDT[f] * beta_face * q_face

    ordering_gap = float(np.max(np.abs(correct - wrong)))
    assert ordering_gap > 1e-8


def test_table2_constant_state_line_flux_identity(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    constant = 2.75
    q = np.full((mesh.elements.shape[0], ref.rs.shape[0]), constant)

    line_flux = projected_interior_line_flux(q, volume, surface)
    line_velocity = projected_line_velocity(volume, surface)

    np.testing.assert_allclose(line_flux, constant * line_velocity, atol=2e-12, rtol=2e-12)


def test_numerical_flux_formulas_and_consistency():
    q_m = np.array([[1.0, -2.0, 3.0, 4.0]])
    q_p = np.array([[0.5, 1.0, -1.0, 4.0]])
    a = np.array([[2.0, -3.0, 0.0, 1.5]])

    central = numerical_flux(q_m, q_p, a, flux_id=0)
    upwind = numerical_flux(q_m, q_p, a, flux_id=1)
    lf0 = numerical_flux(q_m, q_p, a, flux_id=2, lf_alpha=0.0)
    lf1 = numerical_flux(q_m, q_p, a, flux_id=2, lf_alpha=1.0)
    lf2 = numerical_flux(q_m, q_p, a, flux_id=2, lf_alpha=2.0)

    np.testing.assert_allclose(central, 0.5 * a * (q_m + q_p), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(upwind, np.where(a >= 0.0, a * q_m, a * q_p), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(lf1, 0.5 * a * (q_m + q_p) - 0.5 * np.abs(a) * (q_p - q_m), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(lf2, 0.5 * a * (q_m + q_p) - np.abs(a) * (q_p - q_m), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(lf0, central, atol=0.0, rtol=0.0)

    q = np.array([[1.25, -0.5, 2.0]])
    speed = np.array([[2.0, -1.5, 0.0]])
    for flux_id in (0, 1, 2):
        kwargs = {"lf_alpha": 2.0} if flux_id == 2 else {}
        flux = numerical_flux(q, q, speed, flux_id=flux_id, **kwargs)
        np.testing.assert_allclose(flux, speed * q, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="lf_alpha must be non-negative"):
        numerical_flux(q_m, q_p, a, flux_id=2, lf_alpha=-1.0)


@pytest.mark.parametrize(
    ("flux_type", "lf_alpha"),
    [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 2.0),
    ],
)
def test_table2_numerical_flux_interface_antisymmetry(table2_surface_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    q = _state(geom)
    traces = pair_face_traces(q, trace, use_numba=False)
    common = common_projected_line_velocity(volume, surface, trace, use_numba=False)
    flux_id = 2 if flux_type == "lf" else {"central": 0, "upwind": 1}[flux_type]
    flux = numerical_flux(traces.qM, traces.qP, common, flux_id=flux_id, lf_alpha=lf_alpha)

    max_sum = 0.0

    for k, f, nbr, nbr_f in conn.interior_faces:
        flux_plus = _align_face_values(flux, trace, k, f, nbr, nbr_f)
        np.testing.assert_allclose(flux[k, f], -flux_plus, atol=2e-13, rtol=2e-13)
        max_sum = max(max_sum, float(np.max(np.abs(flux[k, f] + flux_plus))))

    assert max_sum < 2e-13


def test_table2_surface_output_buffer_behavior(table2_surface_case):
    ref, mesh, conn, geom, trace, volume, surface = table2_surface_case

    q = _state(geom)
    face_out = np.empty((mesh.elements.shape[0], 3, ref.edge_rules[1].n_points))
    common_out = np.empty_like(face_out)

    returned_flux = projected_interior_line_flux(q, volume, surface, out=face_out)
    returned_vel = projected_line_velocity(volume, surface, out=np.empty_like(face_out))
    returned_common = common_projected_line_velocity(volume, surface, trace, out=common_out, use_numba=False)

    assert returned_flux is face_out
    assert returned_vel.shape == face_out.shape
    assert returned_common is common_out

    with pytest.raises(ValueError, match="out has wrong shape"):
        projected_interior_line_flux(q, volume, surface, out=np.empty((1, 1, 1)))

    with pytest.raises(ValueError, match="out has wrong shape"):
        projected_line_velocity(volume, surface, out=np.empty((1, 1, 1)))

    with pytest.raises(ValueError, match="out has wrong shape"):
        common_projected_line_velocity(volume, surface, trace, out=np.empty((1, 1, 1)))
