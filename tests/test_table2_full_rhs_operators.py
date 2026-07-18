from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    build_full_rhs_cache,
    full_rhs,
    surface_lift_correction_projected_flux,
    surface_lift_correction_split_projected_flux,
    volume_divergence_conservative,
    volume_divergence_split,
)
from simplex_dg.trace import build_trace_cache, pair_face_traces


FLUX_CASES = [
    ("central", 1.0),
    ("upwind", 1.0),
    ("lf", 1.0),
    ("lf", 2.0),
]


@pytest.fixture(scope="module")
def table2_full_case():
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
    return ref, mesh, conn, geom, trace


def _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form, **kwargs):
    return build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=(0.3, -0.2, 0.7),
        flux_type=flux_type,
        lf_alpha=lf_alpha,
        volume_form=volume_form,
        validate=True,
        **kwargs,
    )


def _smooth_state(geom):
    return 1.0 + 0.4 * geom.X[..., 0] - 0.3 * geom.X[..., 1] + 0.2 * geom.X[..., 2]


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_full_rhs_matches_direct_composition(table2_full_case, volume_form, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form)
    q = _smooth_state(geom)
    traces = pair_face_traces(q, trace, use_numba=False)

    if volume_form == "conservative":
        div = volume_divergence_conservative(q, full.volume, use_numba=False)
        surf = surface_lift_correction_projected_flux(
            q,
            traces,
            full.volume,
            full.surface,
            full.trace,
            use_numba=False,
        )
    else:
        div = volume_divergence_split(q, full.volume, use_numba=False)
        surf = surface_lift_correction_split_projected_flux(
            q,
            traces,
            full.volume,
            full.surface,
            full.trace,
            use_numba=False,
        )

    actual = full_rhs(q, full, use_numba=False)
    expected = -div + surf

    np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=5e-12)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_zero_velocity_full_rhs_is_zero(table2_full_case, volume_form, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        velocity_volume=np.zeros_like(geom.X),
        velocity_face=np.zeros_like(geom.X_face),
        flux_type=flux_type,
        lf_alpha=lf_alpha,
        volume_form=volume_form,
        validate=True,
    )
    rng = np.random.default_rng(20260718)
    q = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))

    rhs = full_rhs(q, full, use_numba=False)
    np.testing.assert_allclose(rhs, 0.0, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_full_rhs_is_linear(table2_full_case, volume_form, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form)

    rng = np.random.default_rng(314159)
    q1 = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    q2 = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    a = 1.7
    b = -0.4

    lhs = full_rhs(a * q1 + b * q2, full, use_numba=False)
    rhs = a * full_rhs(q1, full, use_numba=False) + b * full_rhs(q2, full, use_numba=False)

    np.testing.assert_allclose(lhs, rhs, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_full_rhs_shape_and_finiteness(table2_full_case, volume_form, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form)

    states = [
        np.zeros((mesh.elements.shape[0], ref.rs.shape[0])),
        np.ones((mesh.elements.shape[0], ref.rs.shape[0])),
        _smooth_state(geom),
        np.random.default_rng(271828).standard_normal((mesh.elements.shape[0], ref.rs.shape[0])),
    ]

    for q in states:
        rhs = full_rhs(q, full, use_numba=False)
        assert rhs.shape == q.shape
        assert np.all(np.isfinite(rhs))


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_zero_state_gives_zero_full_rhs(table2_full_case, volume_form, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form)

    q = np.zeros((mesh.elements.shape[0], ref.rs.shape[0]))
    rhs = full_rhs(q, full, use_numba=False)

    np.testing.assert_allclose(rhs, 0.0, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_table2_full_rhs_numpy_and_numba_agree(table2_full_case, volume_form):
    pytest.importorskip("numba")
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(ref, geom, trace, "lf", 2.0, volume_form)
    q_random = np.random.default_rng(12345).standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    q_const = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))

    for q in (q_random, q_const):
        rhs_np = full_rhs(q, full, use_numba=False)
        rhs_nb = full_rhs(q, full, use_numba=True)
        np.testing.assert_allclose(rhs_nb, rhs_np, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_table2_full_rhs_output_buffer_behavior(table2_full_case, volume_form):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(ref, geom, trace, "upwind", 1.0, volume_form)
    q = _smooth_state(geom)
    q_copy = q.copy()
    out = np.empty_like(q)

    returned = full_rhs(q, full, out=out, use_numba=False)

    assert returned is out
    np.testing.assert_allclose(q, q_copy, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(returned, full_rhs(q, full, use_numba=False), atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="q must have shape"):
        full_rhs(np.zeros((1, 1)), full, use_numba=False)

    with pytest.raises(ValueError, match="out has wrong shape"):
        full_rhs(q, full, out=np.empty((1, 1)), use_numba=False)
