import numpy as np

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache, full_rhs_split
from simplex_dg.time import RK4C, cfl_dt_from_geometry, integrate_lsrk54
from simplex_dg.trace import build_trace_cache


def _build_case(level=1, order=3):
    ref = build_reference_cache(order=order, table="table1")
    mesh = build_octa_sphere_mesh(level=level, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)
    full = build_full_rhs_cache(ref, geom, trace, constant_preserving=True)

    return mesh, ref, geom, trace, full


def test_lsrk54_rk4c_coefficient_is_correct():
    expected = 2526269341429.0 / 6820363962896.0

    assert RK4C[2] == expected


def test_constant_state_full_rhs_zero():
    mesh, ref, geom, trace, full = _build_case(level=1, order=3)

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))

    rhs = full_rhs_split(q, full)

    assert np.max(np.abs(rhs)) < 1e-10


def test_constant_state_integration_preserved():
    mesh, ref, geom, trace, full = _build_case(level=1, order=3)

    q0 = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))

    dt = cfl_dt_from_geometry(ref, geom, full.volume.max_speed, cfl=0.02)
    tf = 3.0 * dt

    def rhs(t, q):
        return full_rhs_split(q, full, use_numba=False)

    result = integrate_lsrk54(rhs, q0, 0.0, tf, dt)

    assert np.allclose(result.q, q0, atol=1e-10, rtol=1e-10)