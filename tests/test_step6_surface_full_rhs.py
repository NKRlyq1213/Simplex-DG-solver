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
from simplex_dg.time import manifold_integral
from simplex_dg.trace import build_trace_cache, pair_face_traces


def _build_case(
    ndivs=4,
    order=4,
    table="table1",
    flux_type="upwind",
    velocity_volume=None,
    velocity_face=None,
    volume_form="conservative",
):
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
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
        volume_form=volume_form,
    )

    return mesh, ref, geom, trace, full


def test_surface_cache_shapes():
    mesh, ref, geom, trace, full = _build_case(ndivs=4, order=4)

    k_elements = mesh.elements.shape[0]
    n_points = ref.rs.shape[0]
    n_face_points = ref.edge_rules[1].n_points

    surface = full.surface

    assert surface.lift.shape == (3, n_points, n_face_points)
    assert surface.sqrt_g.shape == (k_elements, n_points)
    assert surface.face_jacobian.shape == (k_elements, 3, n_face_points)
    assert surface.face_velocity.shape == (k_elements, 3, n_face_points, 3)
    assert surface.normal_velocity.shape == (k_elements, 3, n_face_points)


def test_face_velocity_is_tangent():
    mesh, ref, geom, trace, full = _build_case(ndivs=4, order=4)

    err = np.max(np.abs(np.sum(full.surface.face_velocity * geom.face_normal, axis=3)))

    assert err < 1e-10


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_zero_velocity_full_rhs_zero(volume_form: str):
    ref = build_reference_cache(order=4, table="table1")
    mesh = build_octa_sphere_mesh(ndivs=2, radius=1.0)
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
        volume_form=volume_form,
    )

    rng = np.random.default_rng(1234)
    q = rng.normal(size=(mesh.elements.shape[0], ref.rs.shape[0]))

    rhs = full_rhs(q, full)

    assert np.allclose(rhs, 0.0, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_full_rhs_shape_and_finite(volume_form: str):
    mesh, ref, geom, trace, full = _build_case(ndivs=4, order=4, volume_form=volume_form)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1]
    rhs = full_rhs(q, full)

    assert rhs.shape == q.shape
    assert np.all(np.isfinite(rhs))


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_full_rhs_matches_volume_plus_projected_surface(volume_form: str):
    mesh, ref, geom, trace, full = _build_case(ndivs=4, order=4, volume_form=volume_form)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]
    traces = pair_face_traces(q, trace)

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

    rhs = full_rhs(q, full, use_numba=False)

    assert np.allclose(rhs, -div + surf, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_numpy_and_numba_full_rhs_paths_agree(volume_form: str):
    mesh, ref, geom, trace, full = _build_case(ndivs=4, order=4, volume_form=volume_form)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    rhs_np = full_rhs(q, full, use_numba=False)
    rhs_nb = full_rhs(q, full, use_numba=True)

    assert np.allclose(rhs_np, rhs_nb, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_full_rhs_linearity(volume_form: str):
    mesh, ref, geom, trace, full = _build_case(ndivs=4, order=4, volume_form=volume_form)

    q1 = geom.X[:, :, 0]
    q2 = geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    a = 1.25
    b = -0.75

    lhs = full_rhs(a * q1 + b * q2, full)
    rhs = a * full_rhs(q1, full) + b * full_rhs(q2, full)

    assert np.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_full_rhs_global_mass_residual_is_small(volume_form: str):
    mesh, ref, geom, trace, full = _build_case(ndivs=4, order=4, volume_form=volume_form)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]
    rhs = full_rhs(q, full, use_numba=False)

    mass_residual = abs(manifold_integral(rhs, ref, geom))
    scale = max(manifold_integral(np.abs(q), ref, geom), 1.0)

    assert mass_residual / scale < 1e-10
