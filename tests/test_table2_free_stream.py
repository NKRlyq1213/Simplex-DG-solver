from __future__ import annotations

import numpy as np

from examples.check_free_stream import attach_rates, compute_free_stream_row


def test_table2_free_stream_refinement_diagnostic():
    alpha0 = -np.pi / 4.0
    u0 = 2.0 * np.pi / 10.0
    ndivs_values = (1, 2, 4, 8)

    rows = [
        compute_free_stream_row(
            order=4,
            table="table2",
            ndivs=ndivs,
            radius=1.0,
            alpha0=alpha0,
            u0=u0,
            flux_type="central",
            lf_alpha=1.0,
            volume_form="conservative",
            project_velocity=True,
        )
        for ndivs in ndivs_values
    ]

    rows = attach_rates(rows)

    linf_values = [row.constant_linf for row in rows]
    l2_values = [row.constant_physical_l2 for row in rows]
    global_values = [row.constant_global_integral for row in rows]

    assert all(np.isfinite(value) for value in linf_values)
    assert all(np.isfinite(value) for value in l2_values)
    assert all(np.isfinite(value) for value in global_values)
    assert all(value >= 0.0 for value in linf_values)
    assert all(value >= 0.0 for value in l2_values)
    assert all(value >= 0.0 for value in global_values)
    assert all(value < 1e-12 for value in global_values)

    if max(linf_values) > 1e-12:
        assert linf_values[-1] < linf_values[0]

    if max(l2_values) > 1e-12:
        assert l2_values[-1] < l2_values[0]

    assert all(row.ndivs == ndivs for row, ndivs in zip(rows, ndivs_values))
    assert all(row.n_points_per_element == 16 for row in rows)
    assert all(row.hmin > 0.0 for row in rows)
