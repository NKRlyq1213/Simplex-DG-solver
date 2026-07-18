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
    surface_lift_correction_projected_flux,
    surface_lift_correction_split_projected_flux,
    volume_divergence_conservative,
    volume_divergence_split,
)
from simplex_dg.time import manifold_integral
from simplex_dg.trace import build_trace_cache, pair_face_traces


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
    return ref, mesh, conn, geom, trace, volume


def _surface(ref, geom, trace, flux_type, lf_alpha):
    return build_surface_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=(0.3, -0.2, 0.7),
        flux_type=flux_type,
        lf_alpha=lf_alpha,
        project_velocity=True,
        validate=True,
    )


def _state(geom):
    return 1.0 + 0.4 * geom.X[..., 0] - 0.3 * geom.X[..., 1] + 0.2 * geom.X[..., 2]


@pytest.mark.parametrize(
    ("flux_type", "lf_alpha"),
    [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 2.0),
    ],
)
def test_table2_integrated_lift_identity(table2_surface_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, flux_type, lf_alpha)
    q = _state(geom)
    traces = pair_face_traces(q, trace, use_numba=False)

    line_flux_m = projected_interior_line_flux(q, volume, surface)
    common = common_projected_line_velocity(volume, surface, trace, use_numba=False)
    flux_star = numerical_flux(traces.qM, traces.qP, common, surface.flux_id, lf_alpha=surface.lf_alpha)

    penalty_cons = line_flux_m - flux_star
    corr_cons = surface_lift_correction_projected_flux(q, traces, volume, surface, trace, use_numba=False)

    lhs_cons = np.sum(ref.area * ref.weights[None, :] * geom.sqrt_g * corr_cons)
    rhs_cons = 0.0
    for f in range(surface.n_faces):
        rhs_cons += np.sum(ref.edge_rules[f + 1].weights[None, :] * penalty_cons[:, f, :])

    np.testing.assert_allclose(lhs_cons, rhs_cons, atol=5e-12, rtol=5e-12)

    line_velocity_m = projected_line_velocity(volume, surface)
    penalty_split = 0.5 * line_flux_m + 0.5 * line_velocity_m * traces.qM - flux_star
    corr_split = surface_lift_correction_split_projected_flux(q, traces, volume, surface, trace, use_numba=False)

    lhs_split = np.sum(ref.area * ref.weights[None, :] * geom.sqrt_g * corr_split)
    rhs_split = 0.0
    for f in range(surface.n_faces):
        rhs_split += np.sum(ref.edge_rules[f + 1].weights[None, :] * penalty_split[:, f, :])

    np.testing.assert_allclose(lhs_split, rhs_split, atol=5e-12, rtol=5e-12)


@pytest.mark.parametrize(
    ("flux_type", "lf_alpha"),
    [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 2.0),
    ],
)
def test_table2_global_weighted_numerical_flux_sum_cancels(table2_surface_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, flux_type, lf_alpha)
    q = _state(geom)
    traces = pair_face_traces(q, trace, use_numba=False)
    common = common_projected_line_velocity(volume, surface, trace, use_numba=False)
    flux = numerical_flux(traces.qM, traces.qP, common, surface.flux_id, lf_alpha=surface.lf_alpha)

    total = 0.0
    for f in range(surface.n_faces):
        total += np.sum(ref.edge_rules[f + 1].weights[None, :] * flux[:, f, :])

    assert abs(float(total)) < 5e-12


@pytest.mark.parametrize(
    ("flux_type", "lf_alpha", "volume_form"),
    [
        ("central", 1.0, "conservative"),
        ("upwind", 1.0, "conservative"),
        ("lf", 1.0, "conservative"),
        ("lf", 2.0, "conservative"),
        ("central", 1.0, "split"),
        ("upwind", 1.0, "split"),
        ("lf", 1.0, "split"),
        ("lf", 2.0, "split"),
    ],
)
def test_table2_global_mass_conservation_coupling(table2_surface_case, flux_type, lf_alpha, volume_form):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, flux_type, lf_alpha)
    q = _state(geom)
    traces = pair_face_traces(q, trace, use_numba=False)

    if volume_form == "conservative":
        div = volume_divergence_conservative(q, volume, use_numba=False)
        surf = surface_lift_correction_projected_flux(q, traces, volume, surface, trace, use_numba=False)
    else:
        div = volume_divergence_split(q, volume, use_numba=False)
        surf = surface_lift_correction_split_projected_flux(q, traces, volume, surface, trace, use_numba=False)

    semi_discrete = -div + surf
    mass_residual = abs(manifold_integral(semi_discrete, ref, geom))
    scale = max(manifold_integral(np.abs(q), ref, geom), 1.0)

    assert mass_residual / scale < 1e-10


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_table2_constant_state_diagnostic_global_integral_is_small(table2_surface_case, volume_form):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, "upwind", 1.0)

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    traces = pair_face_traces(q, trace, use_numba=False)

    if volume_form == "conservative":
        div = volume_divergence_conservative(q, volume, use_numba=False)
        surf = surface_lift_correction_projected_flux(q, traces, volume, surface, trace, use_numba=False)
    else:
        div = volume_divergence_split(q, volume, use_numba=False)
        surf = surface_lift_correction_split_projected_flux(q, traces, volume, surface, trace, use_numba=False)

    combined = -div + surf
    assert np.all(np.isfinite(combined))
    assert abs(manifold_integral(combined, ref, geom)) < 5e-12
