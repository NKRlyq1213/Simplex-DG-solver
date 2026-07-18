import numpy as np

from simplex_dg.diagnostics import error_report, l2_error, relative_l2_error
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.problems import (
    exact_gaussian_solid_body,
    gaussian_center_solid_body,
    gaussian_on_sphere,
)
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache, full_rhs
from simplex_dg.time import cfl_dt_from_geometry, integrate_lsrk54, manifold_integral
from simplex_dg.trace import build_trace_cache


def _build_case(ndivs=2, order=3):
    ref = build_reference_cache(order=order, table="table1")
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)
    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=(0.0, 0.0, 1.0),
        flux_type="upwind",
    )

    return mesh, ref, geom, trace, full


def test_gaussian_center_at_R00_has_peak_one_at_center():
    X = np.array([[[1.0, 0.0, 0.0]]])
    q = gaussian_on_sphere(X, radius=1.0, sigma=0.35)

    assert np.allclose(q[0, 0], 1.0, atol=1e-14, rtol=1e-14)


def test_gaussian_center_rotates_around_z_axis():
    c = gaussian_center_solid_body(
        t=np.pi / 2.0,
        radius=1.0,
        center0=(1.0, 0.0, 0.0),
        omega=(0.0, 0.0, 1.0),
    )

    assert np.allclose(c, np.array([0.0, 1.0, 0.0]), atol=1e-14, rtol=1e-14)


def test_exact_gaussian_t0_matches_initial_gaussian():
    mesh, ref, geom, trace, full = _build_case(ndivs=2, order=3)

    q0 = gaussian_on_sphere(geom.X, radius=mesh.radius, sigma=0.35)
    qe = exact_gaussian_solid_body(geom.X, t=0.0, radius=mesh.radius, sigma=0.35)

    assert np.allclose(q0, qe, atol=1e-14, rtol=1e-14)


def test_error_report_zero_for_exact_match():
    mesh, ref, geom, trace, full = _build_case(ndivs=2, order=3)

    q = gaussian_on_sphere(geom.X, radius=mesh.radius, sigma=0.35)

    rep = error_report(q, q, ref, geom)

    assert rep.l2_error == 0.0
    assert rep.relative_l2_error == 0.0
    assert rep.linf_error == 0.0
    assert rep.mass_error == 0.0


def test_gaussian_short_run_error_is_finite():
    mesh, ref, geom, trace, full = _build_case(ndivs=2, order=3)

    sigma = 0.35

    q0 = gaussian_on_sphere(
        geom.X,
        center=(mesh.radius, 0.0, 0.0),
        radius=mesh.radius,
        sigma=sigma,
    )

    dt = cfl_dt_from_geometry(ref, geom, full.volume.max_speed, cfl=10)
    tf = 1.0

    def rhs(t, q):
        return full_rhs(q, full, use_numba=False)

    result = integrate_lsrk54(rhs, q0, 0.0, tf, dt)

    q_exact = exact_gaussian_solid_body(
        geom.X,
        t=tf,
        radius=mesh.radius,
        sigma=sigma,
        center0=(mesh.radius, 0.0, 0.0),
        omega=(0.0, 0.0, 1.0),
    )

    rep = error_report(result.q, q_exact, ref, geom)

    assert np.isfinite(rep.l2_error)
    assert np.isfinite(rep.relative_l2_error)
    assert np.isfinite(rep.linf_error)
    assert np.isfinite(rep.mass_error)

    assert rep.l2_error >= 0.0
    assert rep.relative_l2_error >= 0.0


def test_gaussian_mass_is_positive():
    mesh, ref, geom, trace, full = _build_case(ndivs=2, order=3)

    q = gaussian_on_sphere(geom.X, radius=mesh.radius, sigma=0.35)

    mass = manifold_integral(q, ref, geom)

    assert mass > 0.0
