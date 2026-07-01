import numpy as np

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    build_volume_rhs_cache,
    volume_divergence_conservative,
    volume_divergence_split,
    volume_rhs_split,
)


def _build_case(ndivs=4, order=4, table="table1", velocity=None):
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    geom = build_geometry_cache(mesh, ref)
    rhs_cache = build_volume_rhs_cache(ref, geom, velocity=velocity)

    return mesh, ref, geom, rhs_cache


def test_volume_rhs_cache_shapes():
    mesh, ref, geom, rhs_cache = _build_case(ndivs=4, order=4)

    K = mesh.elements.shape[0]
    Np = ref.rs.shape[0]

    assert rhs_cache.Dr.shape == (Np, Np)
    assert rhs_cache.Ds.shape == (Np, Np)

    assert rhs_cache.sqrt_g.shape == (K, Np)
    assert rhs_cache.velocity.shape == (K, Np, 3)
    assert rhs_cache.speed.shape == (K, Np)

    assert rhs_cache.alpha.shape == (K, Np)
    assert rhs_cache.beta.shape == (K, Np)
    assert rhs_cache.Dr_alpha.shape == (K, Np)
    assert rhs_cache.Ds_beta.shape == (K, Np)
    assert rhs_cache.div_velocity.shape == (K, Np)


def test_velocity_is_tangent():
    mesh, ref, geom, rhs_cache = _build_case(ndivs=4, order=4)

    tangent_error = np.max(np.abs(np.sum(rhs_cache.velocity * geom.normal, axis=2)))

    assert tangent_error < 1e-10


def test_zero_velocity_gives_zero_divergence():
    ref = build_reference_cache(order=4, table="table1")
    mesh = build_octa_sphere_mesh(ndivs=2, radius=1.0)
    geom = build_geometry_cache(mesh, ref)

    zero_velocity = np.zeros_like(geom.X)
    rhs_cache = build_volume_rhs_cache(ref, geom, velocity=zero_velocity)

    rng = np.random.default_rng(1234)
    q = rng.normal(size=(mesh.elements.shape[0], ref.rs.shape[0]))

    div_split = volume_divergence_split(q, rhs_cache)
    div_cons = volume_divergence_conservative(q, rhs_cache)

    assert np.allclose(div_split, 0.0, atol=1e-14, rtol=1e-14)
    assert np.allclose(div_cons, 0.0, atol=1e-14, rtol=1e-14)


def test_volume_rhs_is_negative_divergence():
    mesh, ref, geom, rhs_cache = _build_case(ndivs=4, order=4)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1]

    div = volume_divergence_split(q, rhs_cache)
    rhs = volume_rhs_split(q, rhs_cache)

    assert np.allclose(rhs, -div, atol=1e-12, rtol=1e-12)


def test_split_operator_linearity():
    mesh, ref, geom, rhs_cache = _build_case(ndivs=4, order=4)

    q1 = geom.X[:, :, 0]
    q2 = geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    a = 1.7
    b = -0.4

    lhs = volume_divergence_split(a * q1 + b * q2, rhs_cache)
    rhs = a * volume_divergence_split(q1, rhs_cache) + b * volume_divergence_split(q2, rhs_cache)

    assert np.allclose(lhs, rhs, atol=1e-11, rtol=1e-11)


def test_numpy_and_numba_split_paths_agree():
    mesh, ref, geom, rhs_cache = _build_case(ndivs=4, order=4)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    div_np = volume_divergence_split(q, rhs_cache, use_numba=False)
    div_nb = volume_divergence_split(q, rhs_cache, use_numba=True)

    assert np.allclose(div_np, div_nb, atol=1e-11, rtol=1e-11)


def test_numpy_and_numba_conservative_paths_agree():
    mesh, ref, geom, rhs_cache = _build_case(ndivs=4, order=4)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    div_np = volume_divergence_conservative(q, rhs_cache, use_numba=False)
    div_nb = volume_divergence_conservative(q, rhs_cache, use_numba=True)

    assert np.allclose(div_np, div_nb, atol=1e-11, rtol=1e-11)


def test_solid_body_rotation_has_nonzero_speed():
    mesh, ref, geom, rhs_cache = _build_case(ndivs=4, order=4)

    assert rhs_cache.max_speed > 0.0
    assert np.max(rhs_cache.speed) == rhs_cache.max_speed
