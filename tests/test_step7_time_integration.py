import numpy as np

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache, full_rhs
from simplex_dg.time import (
    cfl_dt,
    cfl_dt_from_geometry,
    face_lengths_from_geometry,
    integrate_lsrk54,
    lsrk54_step,
    manifold_integral,
    manifold_l2_norm,
    mass_history_entry,
    minimum_face_length,
)
from simplex_dg.trace import build_trace_cache


def _build_full_case(ndivs=2, order=3, flux_type="upwind", zero_velocity=False):
    ref = build_reference_cache(order=order, table="table1")
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    if zero_velocity:
        velocity_volume = np.zeros_like(geom.X)
        velocity_face = np.zeros_like(geom.X_face)
    else:
        velocity_volume = None
        velocity_face = None

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        velocity_volume=velocity_volume,
        velocity_face=velocity_face,
        omega=(0.0, 0.0, 1.0),
        flux_type=flux_type,
    )

    return mesh, ref, geom, trace, full


def test_cfl_dt_positive():
    dt = cfl_dt(cfl=0.25, h=0.1, order=4, max_speed=2.0)

    assert dt > 0.0


def test_face_lengths_positive():
    mesh, ref, geom, trace, full = _build_full_case(ndivs=2, order=3)

    lengths = face_lengths_from_geometry(ref, geom)

    assert lengths.shape == (mesh.elements.shape[0], 3)
    assert np.all(lengths > 0.0)
    assert minimum_face_length(ref, geom) == np.min(lengths)


def test_cfl_dt_from_geometry_positive():
    mesh, ref, geom, trace, full = _build_full_case(ndivs=2, order=3)

    dt = cfl_dt_from_geometry(ref, geom, max_speed=full.volume.max_speed, cfl=0.1)

    assert dt > 0.0


def test_lsrk54_scalar_decay():
    q0 = np.array([[1.0]])

    def rhs(t, q):
        return -q

    dt = 0.01
    result = integrate_lsrk54(rhs, q0, 0.0, 0.1, dt)

    exact = np.exp(-0.1)

    assert result.nsteps == 10
    assert abs(result.t - 0.1) < 1e-14
    assert np.allclose(result.q[0, 0], exact, atol=1e-8, rtol=1e-8)


def test_lsrk54_step_shape():
    q0 = np.ones((4, 5))

    def rhs(t, q):
        return -2.0 * q

    q1 = lsrk54_step(rhs, 0.0, q0, 0.01)

    assert q1.shape == q0.shape
    assert np.all(np.isfinite(q1))


def test_manifold_integral_and_l2_are_finite():
    mesh, ref, geom, trace, full = _build_full_case(ndivs=2, order=3)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1]

    mass = manifold_integral(q, ref, geom)
    l2 = manifold_l2_norm(q, ref, geom)

    assert np.isfinite(mass)
    assert np.isfinite(l2)
    assert l2 > 0.0


def test_zero_velocity_integration_preserves_state():
    mesh, ref, geom, trace, full = _build_full_case(ndivs=2, order=3, zero_velocity=True)

    rng = np.random.default_rng(1234)
    q0 = rng.normal(size=(mesh.elements.shape[0], ref.rs.shape[0]))

    def rhs(t, q):
        return full_rhs(q, full, use_numba=False)

    result = integrate_lsrk54(rhs, q0, 0.0, 0.1, 0.01)

    assert np.allclose(result.q, q0, atol=1e-12, rtol=1e-12)


def test_short_full_rhs_run_is_finite():
    mesh, ref, geom, trace, full = _build_full_case(ndivs=2, order=3)

    q0 = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    dt = cfl_dt_from_geometry(ref, geom, max_speed=full.volume.max_speed, cfl=0.02)
    tf = 3.0 * dt

    def rhs(t, q):
        return full_rhs(q, full, use_numba=False)

    def monitor(t, q):
        return mass_history_entry(t, q, ref, geom)

    result = integrate_lsrk54(
        rhs,
        q0,
        0.0,
        tf,
        dt,
        monitor=monitor,
        monitor_every=1,
    )

    assert result.nsteps == 3
    assert abs(result.t - tf) < 1e-14
    assert np.all(np.isfinite(result.q))
    assert len(result.history) == 4

    for entry in result.history:
        assert np.isfinite(entry["mass"])
        assert np.isfinite(entry["l2"])
        assert np.isfinite(entry["linf"])
