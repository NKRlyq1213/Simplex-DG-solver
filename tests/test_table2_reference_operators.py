from __future__ import annotations

import numpy as np

from simplex_dg.reference import build_reference_cache, vandermonde2d


def reference_polynomial(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 1.0 + r + 2.0 * s + r**2 + r * s + 3.0 * s**2 + r**4


def reference_polynomial_dr(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 1.0 + 2.0 * r + s + 4.0 * r**3


def reference_polynomial_ds(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 2.0 + r + 6.0 * s


def test_table2_order4_reference_operator_shapes():
    ref = build_reference_cache(
        order=4,
        table="table2",
        n_face=5,
        validate=True,
    )

    assert ref.V.shape == (16, 15)
    assert ref.Vr.shape == (16, 15)
    assert ref.Vs.shape == (16, 15)
    assert ref.M.shape == (15, 15)
    assert ref.Minv.shape == (15, 15)
    assert ref.projection.shape == (15, 16)
    assert ref.Dr.shape == (16, 16)
    assert ref.Ds.shape == (16, 16)

    for face_id in (1, 2, 3):
        assert ref.face_interp[face_id].shape == (5, 16)


def test_table2_order4_vandermonde_has_full_column_rank_and_finite_conditioning():
    ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)

    rank = np.linalg.matrix_rank(ref.V)
    cond_v = np.linalg.cond(ref.V)
    cond_m = np.linalg.cond(ref.M)

    assert rank == ref.V.shape[1]
    assert np.isfinite(cond_v)
    assert np.isfinite(cond_m)


def test_table2_order4_projection_is_exact_on_polynomial_space():
    ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)

    np.testing.assert_allclose(
        ref.projection @ ref.V,
        np.eye(ref.V.shape[1]),
        atol=2e-11,
        rtol=2e-11,
    )


def test_table2_order4_differentiation_matrices_match_vandermonde_derivatives():
    ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)

    np.testing.assert_allclose(ref.Dr @ ref.V, ref.Vr, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(ref.Ds @ ref.V, ref.Vs, atol=2e-11, rtol=2e-11)

    ones = np.ones(ref.rs.shape[0])
    np.testing.assert_allclose(ref.Dr @ ones, 0.0, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(ref.Ds @ ones, 0.0, atol=2e-12, rtol=2e-12)


def test_table2_order4_differentiation_is_exact_for_sample_polynomial():
    ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)

    r = ref.rs[:, 0]
    s = ref.rs[:, 1]
    q = reference_polynomial(r, s)

    np.testing.assert_allclose(
        ref.Dr @ q,
        reference_polynomial_dr(r, s),
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        ref.Ds @ q,
        reference_polynomial_ds(r, s),
        atol=2e-11,
        rtol=2e-11,
    )


def test_table2_order4_face_interpolation_matches_face_vandermonde_and_polynomial_values():
    ref = build_reference_cache(order=4, table="table2", n_face=5, validate=True)

    q_volume = reference_polynomial(ref.rs[:, 0], ref.rs[:, 1])

    for face_id in (1, 2, 3):
        edge = ref.edge_rules[face_id]
        v_face = vandermonde2d(ref.order, edge.rs[:, 0], edge.rs[:, 1])
        e_face = ref.face_interp[face_id]

        np.testing.assert_allclose(e_face @ ref.V, v_face, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(
            e_face @ np.ones(ref.rs.shape[0]),
            np.ones(edge.rs.shape[0]),
            atol=2e-11,
            rtol=2e-11,
        )
        np.testing.assert_allclose(
            e_face @ q_volume,
            reference_polynomial(edge.rs[:, 0], edge.rs[:, 1]),
            atol=2e-11,
            rtol=2e-11,
        )
