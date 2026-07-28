from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.reference import (
    build_table1_direct_boundary_data,
    edge_gl_rule,
    load_triangle_rule,
    reference_edge_nodes,
    vandermonde2d,
)
from simplex_dg.reference.quadrature import TABLE1_RAW


TABLE1_ORDERS = sorted(TABLE1_RAW)

_FACE_DRDT = {
    1: -2.0,
    2: 0.0,
    3: 2.0,
}

_FACE_DSDT = {
    1: 2.0,
    2: -2.0,
    3: 0.0,
}


@pytest.mark.parametrize("order", TABLE1_ORDERS)
def test_table1_direct_boundary_face_node_counts(order: int):
    rule = load_triangle_rule(table="table1", order=order)
    data = build_table1_direct_boundary_data(rule=rule)

    for face_id in (1, 2, 3):
        assert data.face_indices[face_id].shape == (order + 1,)
        assert data.face_extract[face_id].shape == (order + 1, rule.rs.shape[0])
        assert data.face_weights[face_id].shape == (order + 1,)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
def test_table1_direct_boundary_coordinates_and_weights_match_face_rules(order: int):
    rule = load_triangle_rule(table="table1", order=order)
    data = build_table1_direct_boundary_data(rule=rule)
    r = rule.rs[:, 0]
    s = rule.rs[:, 1]

    for face_id in (1, 2, 3):
        edge = edge_gl_rule(face_id, order + 1)
        extract = data.face_extract[face_id]

        np.testing.assert_allclose(extract @ r, edge.rs[:, 0], atol=5e-13, rtol=5e-13)
        np.testing.assert_allclose(extract @ s, edge.rs[:, 1], atol=5e-13, rtol=5e-13)
        np.testing.assert_allclose(
            extract @ r,
            reference_edge_nodes(face_id, edge.t01)[:, 0],
            atol=5e-13,
            rtol=5e-13,
        )
        np.testing.assert_allclose(
            extract @ s,
            reference_edge_nodes(face_id, edge.t01)[:, 1],
            atol=5e-13,
            rtol=5e-13,
        )
        np.testing.assert_array_equal(
            data.face_weights[face_id],
            rule.edge_weights[data.face_indices[face_id]],
        )
        np.testing.assert_allclose(
            data.face_weights[face_id],
            edge.weights,
            atol=5e-13,
            rtol=5e-13,
        )
        assert np.all(np.isfinite(data.face_weights[face_id]))
        assert np.all(data.face_weights[face_id] > 0.0)


@pytest.mark.parametrize("order", TABLE1_ORDERS)
def test_table1_direct_boundary_extraction_identity_and_one_hot_rows(order: int):
    rule = load_triangle_rule(table="table1", order=order)
    data = build_table1_direct_boundary_data(rule=rule)
    rng = np.random.default_rng(20260728 + order)
    q = rng.normal(size=rule.rs.shape[0])

    for face_id in (1, 2, 3):
        indices = data.face_indices[face_id]
        extract = data.face_extract[face_id]

        np.testing.assert_allclose(extract @ q, q[indices], atol=0.0, rtol=0.0)
        assert np.unique(indices).size == indices.size
        np.testing.assert_allclose(extract.sum(axis=1), 1.0, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            np.count_nonzero(extract, axis=1),
            np.ones(indices.size),
            atol=0.0,
            rtol=0.0,
        )
        assert np.all(np.isin(extract, (0.0, 1.0)))


@pytest.mark.parametrize("order", TABLE1_ORDERS)
def test_table1_direct_boundary_polynomial_trace_and_direct_product_compatibility(order: int):
    rule = load_triangle_rule(table="table1", order=order)
    data = build_table1_direct_boundary_data(rule=rule)
    rng = np.random.default_rng(314159 + order)
    a = rng.normal(size=rule.rs.shape[0])
    q = rng.normal(size=rule.rs.shape[0])
    v_raw = vandermonde2d(order, rule.rs[:, 0], rule.rs[:, 1])

    for face_id in (1, 2, 3):
        edge = edge_gl_rule(face_id, order + 1)
        v_face = vandermonde2d(order, edge.rs[:, 0], edge.rs[:, 1])
        extract = data.face_extract[face_id]

        np.testing.assert_allclose(extract @ v_raw, v_face, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(
            extract @ (a * q),
            (extract @ a) * (extract @ q),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize("order", TABLE1_ORDERS)
def test_table1_direct_boundary_matrices_are_symmetric_and_match_manual_face_sum(order: int):
    rule = load_triangle_rule(table="table1", order=order)
    data = build_table1_direct_boundary_data(rule=rule)

    br_manual = np.zeros_like(data.Br)
    bs_manual = np.zeros_like(data.Bs)

    for face_id in (1, 2, 3):
        extract = data.face_extract[face_id]
        weights = data.face_weights[face_id]
        face_term = extract.T @ (weights[:, None] * extract)
        br_manual += _FACE_DSDT[face_id] * face_term
        bs_manual += (-_FACE_DRDT[face_id]) * face_term

    np.testing.assert_allclose(data.Br, br_manual, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.Bs, bs_manual, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.Br, data.Br.T, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.Bs, data.Bs.T, atol=0.0, rtol=0.0)


def test_table1_direct_boundary_rejects_table2():
    rule = load_triangle_rule(table="table2", order=4)

    with pytest.raises(ValueError, match="table1"):
        build_table1_direct_boundary_data(rule=rule)
