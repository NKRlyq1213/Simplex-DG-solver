import numpy as np

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    build_full_rhs_cache,
    full_rhs_split,
    surface_lift_correction,
)
from simplex_dg.trace import build_trace_cache, pair_face_traces


def _build_case(level=2, order=4, table="table1", flux_type="upwind", velocity_volume=None, velocity_face=None):
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(level=level, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        velocity_volume=velocity_volume,
        velocity_face=velocity_face,
        flux_type=flux_type,
        constant_preserving=True,
    )

    return mesh, ref, conn, geom, trace, full


def test_surface_cache_shapes():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4)

    K = mesh.elements.shape[0]
    Np = ref.rs.shape[0]
    Nf = ref.edge_rules[1].n_points

    surface = full.surface

    assert surface.lift.shape == (3, Np, Nf)
    assert surface.sqrt_g.shape == (K, Np)
    assert surface.face_jacobian.shape == (K, 3, Nf)
    assert surface.face_velocity.shape == (K, 3, Nf, 3)
    assert surface.normal_velocity.shape == (K, 3, Nf)


def test_face_velocity_is_tangent():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4)

    err = np.max(np.abs(np.sum(full.surface.face_velocity * geom.face_normal, axis=3)))

    assert err < 1e-10


def test_constant_field_surface_correction_zero_upwind():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4, flux_type="upwind")

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    traces = pair_face_traces(q, trace)
    surf = surface_lift_correction(traces, full.surface)

    assert np.max(np.abs(surf)) < 1e-10


def test_constant_field_surface_correction_zero_central():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4, flux_type="central")

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    traces = pair_face_traces(q, trace)
    surf = surface_lift_correction(traces, full.surface)

    assert np.max(np.abs(surf)) < 1e-10


def test_constant_field_surface_correction_zero_lf():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4, flux_type="lf")

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    traces = pair_face_traces(q, trace)
    surf = surface_lift_correction(traces, full.surface)

    assert np.max(np.abs(surf)) < 1e-10


def test_constant_field_full_rhs_zero_for_solid_body_rotation():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4, flux_type="upwind")

    q = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    rhs = full_rhs_split(q, full)

    assert np.max(np.abs(rhs)) < 1e-10


def test_zero_velocity_full_rhs_zero():
    ref = build_reference_cache(order=4, table="table1")
    mesh = build_octa_sphere_mesh(level=1, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    velocity_volume = np.zeros_like(geom.X)
    velocity_face = np.zeros_like(geom.X_face)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        velocity_volume=velocity_volume,
        velocity_face=velocity_face,
        flux_type="upwind",
        constant_preserving=True,
    )

    rng = np.random.default_rng(1234)
    q = rng.normal(size=(mesh.elements.shape[0], ref.rs.shape[0]))

    rhs = full_rhs_split(q, full)

    assert np.allclose(rhs, 0.0, atol=1e-14, rtol=1e-14)


def test_full_rhs_shape():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1]
    rhs = full_rhs_split(q, full)

    assert rhs.shape == q.shape


def test_numpy_and_numba_surface_paths_agree():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]
    traces = pair_face_traces(q, trace)

    surf_np = surface_lift_correction(traces, full.surface, use_numba=False)
    surf_nb = surface_lift_correction(traces, full.surface, use_numba=True)

    assert np.allclose(surf_np, surf_nb, atol=1e-11, rtol=1e-11)


def test_numpy_and_numba_full_rhs_paths_agree():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    rhs_np = full_rhs_split(q, full, use_numba=False)
    rhs_nb = full_rhs_split(q, full, use_numba=True)

    assert np.allclose(rhs_np, rhs_nb, atol=1e-11, rtol=1e-11)


def test_full_rhs_linearity():
    mesh, ref, conn, geom, trace, full = _build_case(level=2, order=4)

    q1 = geom.X[:, :, 0]
    q2 = geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    a = 1.25
    b = -0.75

    lhs = full_rhs_split(a * q1 + b * q2, full)
    rhs = a * full_rhs_split(q1, full) + b * full_rhs_split(q2, full)

    assert np.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)