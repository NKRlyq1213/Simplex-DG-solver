from __future__ import annotations

from math import comb, factorial

import numpy as np

from simplex_dg.reference.quadrature import REFERENCE_AREA, edge_gl_rule, load_triangle_rule


def exact_reference_monomial_integral(a: int, b: int) -> float:
    total = 0.0

    for i in range(a + 1):
        for j in range(b + 1):
            total += (
                comb(a, i)
                * comb(b, j)
                * (2.0 ** (i + j))
                * ((-1.0) ** ((a - i) + (b - j)))
                * factorial(i)
                * factorial(j)
                / factorial(i + j + 2)
            )

    return 4.0 * total


def test_table2_order4_rule_metadata_and_weights():
    rule = load_triangle_rule(table="table2", order=4)

    assert rule.table == "table2"
    assert rule.order == 4
    assert rule.rs.shape == (16, 2)
    assert rule.weights.shape == (16,)
    assert rule.edge_weights is None
    assert np.all(rule.weights > 0.0)
    np.testing.assert_allclose(rule.weights.sum(), 1.0, atol=1e-14, rtol=1e-14)


def test_table2_order4_barycentric_nodes_are_strictly_interior():
    rule = load_triangle_rule(table="table2", order=4)

    tol = 1e-14

    assert rule.bary_raw.shape == (16, 3)
    assert np.all(rule.bary_raw > tol)
    np.testing.assert_allclose(rule.bary_raw.sum(axis=1), 1.0, atol=1e-14, rtol=1e-14)


def test_table2_order4_volume_quadrature_is_exact_through_degree8():
    rule = load_triangle_rule(table="table2", order=4)
    r = rule.rs[:, 0]
    s = rule.rs[:, 1]

    max_error = 0.0

    for a in range(9):
        for b in range(9 - a):
            numerical = REFERENCE_AREA * np.sum(rule.weights * (r**a) * (s**b))
            exact = exact_reference_monomial_integral(a, b)
            max_error = max(max_error, abs(numerical - exact))
            np.testing.assert_allclose(numerical, exact, atol=5e-13, rtol=5e-13)

    assert max_error < 5e-13


def test_order4_face_gl_rule_metadata_geometry_and_exactness():
    for face_id in (1, 2, 3):
        edge = edge_gl_rule(face_id, n_points=5)

        assert edge.t01.shape == (5,)
        assert edge.weights.shape == (5,)
        assert edge.rs.shape == (5, 2)
        assert np.all(edge.weights > 0.0)
        assert np.all(edge.t01 > 0.0)
        assert np.all(edge.t01 < 1.0)
        assert np.all(np.diff(edge.t01) > 0.0)

        np.testing.assert_allclose(edge.weights.sum(), 1.0, atol=5e-14, rtol=5e-14)

        if face_id == 1:
            np.testing.assert_allclose(edge.rs[:, 0] + edge.rs[:, 1], 0.0, atol=1e-14, rtol=1e-14)
        elif face_id == 2:
            np.testing.assert_allclose(edge.rs[:, 0], -1.0, atol=1e-14, rtol=1e-14)
        else:
            np.testing.assert_allclose(edge.rs[:, 1], -1.0, atol=1e-14, rtol=1e-14)

        for k in range(10):
            numerical = np.sum(edge.weights * (edge.t01**k))
            exact = 1.0 / (k + 1)
            np.testing.assert_allclose(numerical, exact, atol=5e-14, rtol=5e-14)
