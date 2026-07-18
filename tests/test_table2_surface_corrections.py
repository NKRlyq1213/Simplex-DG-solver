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
)
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
def test_table2_zero_velocity_surface_corrections_are_zero(table2_surface_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace, volume = table2_surface_case

    zero_volume = build_volume_rhs_cache(
        ref,
        geom,
        velocity=np.zeros_like(geom.X),
        project_velocity=True,
        validate=True,
    )
    zero_surface = build_surface_rhs_cache(
        ref,
        geom,
        trace,
        velocity_face=np.zeros_like(geom.X_face),
        flux_type=flux_type,
        lf_alpha=lf_alpha,
        project_velocity=True,
        validate=True,
    )

    rng = np.random.default_rng(20260718)
    q = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    traces = pair_face_traces(q, trace, use_numba=False)

    cons = surface_lift_correction_projected_flux(q, traces, zero_volume, zero_surface, trace, use_numba=False)
    split = surface_lift_correction_split_projected_flux(q, traces, zero_volume, zero_surface, trace, use_numba=False)

    np.testing.assert_allclose(cons, 0.0, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(split, 0.0, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize(
    ("flux_type", "lf_alpha"),
    [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 2.0),
    ],
)
def test_table2_conservative_surface_correction_matches_direct_definition(table2_surface_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, flux_type, lf_alpha)
    q = _state(geom)
    traces = pair_face_traces(q, trace, use_numba=False)

    line_flux_m = projected_interior_line_flux(q, volume, surface)
    line_velocity_common = common_projected_line_velocity(volume, surface, trace, use_numba=False)
    flux_star = numerical_flux(
        qM=traces.qM,
        qP=traces.qP,
        normal_velocity=line_velocity_common,
        flux_id=surface.flux_id,
        lf_alpha=surface.lf_alpha,
    )
    penalty = line_flux_m - flux_star

    expected = np.zeros((mesh.elements.shape[0], ref.rs.shape[0]))
    for f in range(surface.n_faces):
        expected += penalty[:, f, :] @ surface.lift[f].T
    expected /= surface.sqrt_g

    actual = surface_lift_correction_projected_flux(
        q,
        traces,
        volume,
        surface,
        trace,
        use_numba=False,
    )

    np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=5e-12)


@pytest.mark.parametrize(
    ("flux_type", "lf_alpha"),
    [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 2.0),
    ],
)
def test_table2_split_surface_correction_matches_direct_definition(table2_surface_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, flux_type, lf_alpha)
    q = _state(geom)
    traces = pair_face_traces(q, trace, use_numba=False)

    line_flux_cons = projected_interior_line_flux(q, volume, surface)
    line_velocity_m = projected_line_velocity(volume, surface)
    line_flux_split = 0.5 * line_flux_cons + 0.5 * line_velocity_m * traces.qM
    line_velocity_common = common_projected_line_velocity(volume, surface, trace, use_numba=False)
    flux_star = numerical_flux(
        qM=traces.qM,
        qP=traces.qP,
        normal_velocity=line_velocity_common,
        flux_id=surface.flux_id,
        lf_alpha=surface.lf_alpha,
    )
    penalty = line_flux_split - flux_star

    expected = np.zeros((mesh.elements.shape[0], ref.rs.shape[0]))
    for f in range(surface.n_faces):
        expected += penalty[:, f, :] @ surface.lift[f].T
    expected /= surface.sqrt_g

    actual = surface_lift_correction_split_projected_flux(
        q,
        traces,
        volume,
        surface,
        trace,
        use_numba=False,
    )

    np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=5e-12)


@pytest.mark.parametrize(
    ("flux_type", "lf_alpha"),
    [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 2.0),
    ],
)
def test_table2_constant_state_conservative_and_split_corrections_match(table2_surface_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, flux_type, lf_alpha)

    q = np.full((mesh.elements.shape[0], ref.rs.shape[0]), 2.75)
    traces = pair_face_traces(q, trace, use_numba=False)

    cons = surface_lift_correction_projected_flux(q, traces, volume, surface, trace, use_numba=False)
    split = surface_lift_correction_split_projected_flux(q, traces, volume, surface, trace, use_numba=False)

    np.testing.assert_allclose(split, cons, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize(
    ("flux_type", "lf_alpha"),
    [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 2.0),
    ],
)
def test_table2_surface_corrections_are_linear(table2_surface_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, flux_type, lf_alpha)
    rng = np.random.default_rng(271828)
    q1 = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    q2 = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    a = 1.7
    b = -0.4

    traces1 = pair_face_traces(q1, trace, use_numba=False)
    traces2 = pair_face_traces(q2, trace, use_numba=False)
    traces12 = pair_face_traces(a * q1 + b * q2, trace, use_numba=False)

    cons_lhs = surface_lift_correction_projected_flux(a * q1 + b * q2, traces12, volume, surface, trace, use_numba=False)
    cons_rhs = (
        a * surface_lift_correction_projected_flux(q1, traces1, volume, surface, trace, use_numba=False)
        + b * surface_lift_correction_projected_flux(q2, traces2, volume, surface, trace, use_numba=False)
    )
    split_lhs = surface_lift_correction_split_projected_flux(a * q1 + b * q2, traces12, volume, surface, trace, use_numba=False)
    split_rhs = (
        a * surface_lift_correction_split_projected_flux(q1, traces1, volume, surface, trace, use_numba=False)
        + b * surface_lift_correction_split_projected_flux(q2, traces2, volume, surface, trace, use_numba=False)
    )

    np.testing.assert_allclose(cons_lhs, cons_rhs, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(split_lhs, split_rhs, atol=1e-11, rtol=1e-11)


def test_table2_surface_correction_output_buffer_behavior(table2_surface_case):
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, "upwind", 1.0)
    q = _state(geom)
    traces = pair_face_traces(q, trace, use_numba=False)
    out_cons = np.empty((mesh.elements.shape[0], ref.rs.shape[0]))
    out_split = np.empty_like(out_cons)

    returned_cons = surface_lift_correction_projected_flux(
        q,
        traces,
        volume,
        surface,
        trace,
        out=out_cons,
        use_numba=False,
    )
    returned_split = surface_lift_correction_split_projected_flux(
        q,
        traces,
        volume,
        surface,
        trace,
        out=out_split,
        use_numba=False,
    )

    assert returned_cons is out_cons
    assert returned_split is out_split

    with pytest.raises(ValueError, match="out has wrong shape"):
        surface_lift_correction_projected_flux(q, traces, volume, surface, trace, out=np.empty((1, 1)), use_numba=False)

    with pytest.raises(ValueError, match="out has wrong shape"):
        surface_lift_correction_split_projected_flux(q, traces, volume, surface, trace, out=np.empty((1, 1)), use_numba=False)


def test_table2_surface_numpy_and_numba_paths_agree(table2_surface_case):
    pytest.importorskip("numba")
    ref, mesh, conn, geom, trace, volume = table2_surface_case
    surface = _surface(ref, geom, trace, "lf", 2.0)
    q = _state(geom)
    traces_np = pair_face_traces(q, trace, use_numba=False)
    traces_nb = pair_face_traces(q, trace, use_numba=True)

    cons_np = surface_lift_correction_projected_flux(q, traces_np, volume, surface, trace, use_numba=False)
    cons_nb = surface_lift_correction_projected_flux(q, traces_nb, volume, surface, trace, use_numba=True)
    split_np = surface_lift_correction_split_projected_flux(q, traces_np, volume, surface, trace, use_numba=False)
    split_nb = surface_lift_correction_split_projected_flux(q, traces_nb, volume, surface, trace, use_numba=True)

    np.testing.assert_allclose(cons_nb, cons_np, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(split_nb, split_np, atol=1e-12, rtol=1e-12)
