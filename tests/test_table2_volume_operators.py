from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    apply_reference_operator,
    build_volume_rhs_cache,
    volume_divergence_conservative,
    volume_divergence_split,
    volume_rhs_conservative,
    volume_rhs_split,
)


@pytest.fixture(scope="module")
def table2_volume_case():
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
    geom = build_geometry_cache(
        mesh,
        ref,
        validate=True,
    )
    volume = build_volume_rhs_cache(
        ref,
        geom,
        omega=(0.3, -0.2, 0.7),
        project_velocity=True,
        validate=True,
    )
    return mesh, ref, geom, volume


def _deterministic_q(geom):
    return 1.0 + 0.4 * geom.X[..., 0] - 0.3 * geom.X[..., 1] + 0.2 * geom.X[..., 2]


def test_table2_volume_conservative_operator_matches_direct_definition(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    q = _deterministic_q(geom)
    dr_alpha_q = apply_reference_operator(volume.Dr, volume.alpha * q)
    ds_beta_q = apply_reference_operator(volume.Ds, volume.beta * q)
    expected = (dr_alpha_q + ds_beta_q) / volume.sqrt_g

    actual = volume_divergence_conservative(q, volume, use_numba=False)

    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_table2_volume_split_operator_matches_direct_definition(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    q = _deterministic_q(geom)
    qr = apply_reference_operator(volume.Dr, q)
    qs = apply_reference_operator(volume.Ds, q)
    dr_alpha_q = apply_reference_operator(volume.Dr, volume.alpha * q)
    ds_beta_q = apply_reference_operator(volume.Ds, volume.beta * q)

    split_r = 0.5 * (dr_alpha_q + volume.alpha * qr + q * volume.Dr_alpha)
    split_s = 0.5 * (ds_beta_q + volume.beta * qs + q * volume.Ds_beta)
    expected = (split_r + split_s) / volume.sqrt_g

    actual = volume_divergence_split(q, volume, use_numba=False)

    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_table2_volume_rhs_sign_matches_divergence(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    q = _deterministic_q(geom)

    div_cons = volume_divergence_conservative(q, volume, use_numba=False)
    rhs_cons = volume_rhs_conservative(q, volume, use_numba=False)
    div_split = volume_divergence_split(q, volume, use_numba=False)
    rhs_split = volume_rhs_split(q, volume, use_numba=False)

    np.testing.assert_allclose(rhs_cons, -div_cons, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(rhs_split, -div_split, atol=2e-13, rtol=2e-13)


def test_table2_volume_operators_are_linear(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    rng = np.random.default_rng(314159)
    q1 = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    q2 = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    a = 1.7
    b = -0.4

    cons_lhs = volume_divergence_conservative(a * q1 + b * q2, volume, use_numba=False)
    cons_rhs = (
        a * volume_divergence_conservative(q1, volume, use_numba=False)
        + b * volume_divergence_conservative(q2, volume, use_numba=False)
    )
    split_lhs = volume_divergence_split(a * q1 + b * q2, volume, use_numba=False)
    split_rhs = (
        a * volume_divergence_split(q1, volume, use_numba=False)
        + b * volume_divergence_split(q2, volume, use_numba=False)
    )

    np.testing.assert_allclose(cons_lhs, cons_rhs, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(split_lhs, split_rhs, atol=2e-12, rtol=2e-12)


def test_table2_volume_constant_state_matches_div_velocity(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    ones = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    div_cons = volume_divergence_conservative(ones, volume, use_numba=False)
    div_split = volume_divergence_split(ones, volume, use_numba=False)

    np.testing.assert_allclose(div_cons, volume.div_velocity, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(div_split, volume.div_velocity, atol=2e-12, rtol=2e-12)


def test_table2_volume_zero_velocity_gives_zero_operators():
    ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    geom = build_geometry_cache(mesh, ref, validate=True)
    zero_cache = build_volume_rhs_cache(
        ref,
        geom,
        velocity=np.zeros_like(geom.X),
        project_velocity=True,
        validate=True,
    )

    rng = np.random.default_rng(20260718)
    q = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))

    np.testing.assert_allclose(
        volume_divergence_conservative(q, zero_cache, use_numba=False),
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        volume_divergence_split(q, zero_cache, use_numba=False),
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        volume_rhs_conservative(q, zero_cache, use_numba=False),
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        volume_rhs_split(q, zero_cache, use_numba=False),
        0.0,
        atol=0.0,
        rtol=0.0,
    )


def test_table2_volume_numpy_and_numba_paths_agree(table2_volume_case):
    pytest.importorskip("numba")
    mesh, ref, geom, volume = table2_volume_case

    rng = np.random.default_rng(271828)
    q = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))

    cons_np = volume_divergence_conservative(q, volume, use_numba=False)
    cons_nb = volume_divergence_conservative(q, volume, use_numba=True)
    split_np = volume_divergence_split(q, volume, use_numba=False)
    split_nb = volume_divergence_split(q, volume, use_numba=True)

    np.testing.assert_allclose(cons_nb, cons_np, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(split_nb, split_np, atol=2e-12, rtol=2e-12)


def test_table2_volume_output_buffer_behavior(table2_volume_case):
    mesh, ref, geom, volume = table2_volume_case

    q = _deterministic_q(geom)
    out_cons = np.empty_like(q)
    out_split = np.empty_like(q)
    out_rhs_cons = np.empty_like(q)
    out_rhs_split = np.empty_like(q)

    returned_cons = volume_divergence_conservative(q, volume, out=out_cons, use_numba=False)
    returned_split = volume_divergence_split(q, volume, out=out_split, use_numba=False)
    returned_rhs_cons = volume_rhs_conservative(q, volume, out=out_rhs_cons, use_numba=False)
    returned_rhs_split = volume_rhs_split(q, volume, out=out_rhs_split, use_numba=False)

    assert returned_cons is out_cons
    assert returned_split is out_split
    assert returned_rhs_cons is out_rhs_cons
    assert returned_rhs_split is out_rhs_split

    with pytest.raises(ValueError, match="out has wrong shape"):
        volume_divergence_conservative(q, volume, out=np.empty((1, 1)), use_numba=False)

    with pytest.raises(ValueError, match="out has wrong shape"):
        volume_divergence_split(q, volume, out=np.empty((1, 1)), use_numba=False)
