from __future__ import annotations

import numpy as np

from examples.check_metric_divergence import attach_rates, compute_metric_divergence_row


def test_table2_metric_divergence_refines_monotonically():
    alpha0 = -np.pi / 4.0
    u0 = 2.0 * np.pi / 10.0
    ndivs_values = (1, 2, 4, 8)

    rows = [
        compute_metric_divergence_row(
            order=4,
            table="table2",
            ndivs=ndivs,
            radius=1.0,
            alpha0=alpha0,
            u0=u0,
            project_velocity=True,
        )
        for ndivs in ndivs_values
    ]

    rows = attach_rates(rows)

    cons_linf_values = [row.conservative_linf for row in rows]
    cons_l2_values = [row.conservative_l2_ref for row in rows]
    cons_mean_values = [row.conservative_weighted_mean for row in rows]
    phys_linf_values = [row.physical_linf for row in rows]
    phys_l2_values = [row.physical_l2 for row in rows]
    phys_mean_values = [row.physical_weighted_mean for row in rows]

    for values in (
        cons_linf_values,
        cons_l2_values,
        cons_mean_values,
        phys_linf_values,
        phys_l2_values,
        phys_mean_values,
    ):
        assert all(np.isfinite(value) for value in values)
        assert all(value >= 0.0 for value in values)

    assert cons_linf_values[-1] < cons_linf_values[0]
    assert cons_l2_values[-1] < cons_l2_values[0]
    assert phys_linf_values[-1] < phys_linf_values[0]
    assert phys_l2_values[-1] < phys_l2_values[0]

    for values in (
        cons_linf_values,
        cons_l2_values,
        phys_linf_values,
        phys_l2_values,
    ):
        assert all(curr < prev for prev, curr in zip(values[:-1], values[1:]))

    assert all(row.ndivs == ndivs for row, ndivs in zip(rows, ndivs_values))
    assert all(row.n_points_per_element == 16 for row in rows)
    assert all(row.n_elements > 0 for row in rows)
    assert all(row.hmin > 0.0 for row in rows)
