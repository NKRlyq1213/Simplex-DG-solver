from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache, full_rhs
from simplex_dg.time import manifold_integral


FLUX_CASES = [
    ("central", 1.0),
    ("upwind", 1.0),
    ("lf", 0.0),
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


from simplex_dg.trace import build_trace_cache


def _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form):
    return build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=(0.3, -0.2, 0.7),
        flux_type=flux_type,
        lf_alpha=lf_alpha,
        volume_form=volume_form,
        validate=True,
    )


@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_constant_state_conservative_and_split_match(table2_full_case, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))

    full_cons = _build_full(ref, geom, trace, flux_type, lf_alpha, "conservative")
    full_split = _build_full(ref, geom, trace, flux_type, lf_alpha, "split")

    rhs_cons = full_rhs(q, full_cons, use_numba=False)
    rhs_split = full_rhs(q, full_split, use_numba=False)

    np.testing.assert_allclose(rhs_cons, rhs_split, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_table2_constant_state_flux_independence(table2_full_case, volume_form):
    ref, mesh, conn, geom, trace = table2_full_case
    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))

    results = []
    for flux_type, lf_alpha in FLUX_CASES:
        full = _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form)
        results.append(full_rhs(q, full, use_numba=False))

    for rhs in results[1:]:
        np.testing.assert_allclose(rhs, results[0], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_constant_state_global_conservation(table2_full_case, volume_form, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    full = _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form)

    rhs = full_rhs(q, full, use_numba=False)
    global_integral = abs(manifold_integral(rhs, ref, geom))
    sphere_area = manifold_integral(np.ones_like(q), ref, geom)
    scaled = global_integral / max(abs(sphere_area), 1.0)

    assert scaled < 1e-12


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_random_state_global_mass_conservation(table2_full_case, volume_form, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form)

    for seed in (12345, 271828, 314159):
        q = np.random.default_rng(seed).standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
        rhs = full_rhs(q, full, use_numba=False)
        mass_residual = abs(manifold_integral(rhs, ref, geom))
        scale = max(manifold_integral(np.abs(q), ref, geom), 1.0)
        assert mass_residual / scale < 1e-10


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_mass_functional_linearity(table2_full_case, volume_form, flux_type, lf_alpha):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(ref, geom, trace, flux_type, lf_alpha, volume_form)

    rng = np.random.default_rng(271828)
    q1 = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    q2 = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    a = 1.7
    b = -0.4

    lhs = manifold_integral(full_rhs(a * q1 + b * q2, full, use_numba=False), ref, geom)
    rhs = (
        a * manifold_integral(full_rhs(q1, full, use_numba=False), ref, geom)
        + b * manifold_integral(full_rhs(q2, full, use_numba=False), ref, geom)
    )

    np.testing.assert_allclose(lhs, rhs, atol=5e-12, rtol=5e-12)
