from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache
from simplex_dg.trace import build_trace_cache


OMEGA = np.array([0.3, -0.2, 0.7])


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


def _build_full(ref, geom, trace, **kwargs):
    return build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        validate=True,
        **kwargs,
    )


def test_table2_full_cache_dimensions_and_finiteness(table2_full_case):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(
        ref,
        geom,
        trace,
        omega=OMEGA,
        flux_type="upwind",
        lf_alpha=1.0,
        volume_form="conservative",
        project_velocity=True,
    )

    assert full.volume.n_elements == 128
    assert full.volume.n_points == 16
    assert full.surface.n_elements == 128
    assert full.surface.n_points == 16
    assert full.surface.n_faces == 3
    assert full.surface.n_face_points == 5
    assert full.trace is trace
    assert full.volume_form == "conservative"
    assert full.surface.face_interp.shape == (3, 5, 16)
    assert full.surface.lift.shape == (3, 16, 5)

    for array in (
        full.volume.sqrt_g,
        full.volume.velocity,
        full.volume.alpha,
        full.surface.sqrt_g,
        full.surface.face_velocity,
        full.surface.normal_velocity,
    ):
        assert np.all(np.isfinite(array))


@pytest.mark.parametrize(
    ("volume_form", "expected"),
    [
        ("conservative", "conservative"),
        ("cons", "conservative"),
        ("divergence", "conservative"),
        ("split", "split"),
        ("split_form", "split"),
        ("skew", "split"),
    ],
)
def test_table2_volume_form_aliases(table2_full_case, volume_form, expected):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(
        ref,
        geom,
        trace,
        omega=OMEGA,
        flux_type="upwind",
        volume_form=volume_form,
    )

    assert full.volume_form == expected


@pytest.mark.parametrize(
    ("flux_type", "expected_id", "expected_name"),
    [
        ("central", 0, "central"),
        ("upwind", 1, "upwind"),
        ("lf", 2, "lf"),
        ("lax_friedrichs", 2, "lax_friedrichs"),
        ("lax-friedrichs", 2, "lax-friedrichs"),
    ],
)
def test_table2_flux_parameter_propagation(table2_full_case, flux_type, expected_id, expected_name):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(
        ref,
        geom,
        trace,
        omega=OMEGA,
        flux_type=flux_type,
        lf_alpha=2.0,
        volume_form="conservative",
    )

    assert full.surface.flux_type == expected_name
    assert full.surface.flux_id == expected_id
    assert full.surface.lf_alpha == pytest.approx(2.0)


def test_table2_volume_and_face_velocity_definitions_and_tangency(table2_full_case):
    ref, mesh, conn, geom, trace = table2_full_case
    full = _build_full(
        ref,
        geom,
        trace,
        omega=OMEGA,
        flux_type="upwind",
        volume_form="conservative",
    )

    expected_volume = np.cross(np.broadcast_to(OMEGA, geom.X.shape), geom.X)
    expected_face = np.cross(np.broadcast_to(OMEGA, geom.X_face.shape), geom.X_face)

    np.testing.assert_allclose(full.volume.velocity, expected_volume, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(full.surface.face_velocity, expected_face, atol=2e-14, rtol=2e-14)

    volume_tangency = float(np.max(np.abs(np.sum(full.volume.velocity * geom.normal, axis=2))))
    face_tangency = float(np.max(np.abs(np.sum(full.surface.face_velocity * geom.face_normal, axis=3))))

    assert volume_tangency < 2e-14
    assert face_tangency < 2e-14


def test_table2_custom_velocity_inputs_are_used(table2_full_case):
    ref, mesh, conn, geom, trace = table2_full_case

    tangent_volume = np.cross(np.broadcast_to(OMEGA, geom.X.shape), geom.X)
    tangent_face = np.cross(np.broadcast_to(OMEGA, geom.X_face.shape), geom.X_face)

    full = _build_full(
        ref,
        geom,
        trace,
        velocity_volume=tangent_volume,
        velocity_face=tangent_face,
        flux_type="central",
        volume_form="split",
    )

    np.testing.assert_allclose(full.volume.velocity, tangent_volume, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(full.surface.face_velocity, tangent_face, atol=2e-14, rtol=2e-14)


def test_table2_zero_custom_velocities_propagate_to_subcaches(table2_full_case):
    ref, mesh, conn, geom, trace = table2_full_case

    full = _build_full(
        ref,
        geom,
        trace,
        velocity_volume=np.zeros_like(geom.X),
        velocity_face=np.zeros_like(geom.X_face),
        flux_type="upwind",
        volume_form="conservative",
    )

    np.testing.assert_allclose(full.volume.velocity, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(full.surface.face_velocity, 0.0, atol=0.0, rtol=0.0)


def test_table2_full_cache_validation_failures(table2_full_case):
    ref, mesh, conn, geom, trace = table2_full_case

    with pytest.raises(ValueError, match="volume_form must be"):
        _build_full(ref, geom, trace, volume_form="bad", flux_type="upwind")

    with pytest.raises(ValueError, match="flux_type must be"):
        _build_full(ref, geom, trace, volume_form="conservative", flux_type="bad")

    with pytest.raises(ValueError, match="lf_alpha must be non-negative"):
        _build_full(ref, geom, trace, volume_form="conservative", flux_type="lf", lf_alpha=-1.0)

    with pytest.raises(ValueError, match="velocity must have shape"):
        _build_full(
            ref,
            geom,
            trace,
            volume_form="conservative",
            flux_type="upwind",
            velocity_volume=np.zeros((1, 1, 3)),
        )

    with pytest.raises(ValueError, match="velocity_face must have shape"):
        _build_full(
            ref,
            geom,
            trace,
            volume_form="conservative",
            flux_type="upwind",
            velocity_face=np.zeros((1, 1, 1, 3)),
        )
