from __future__ import annotations

import numpy as np
import pytest

from examples import step9_gaussian_convergence as step9
from simplex_dg.diagnostics import rows_to_dicts_with_rates


def _run_small_case(table: str):
    results = [
        step9.run_one_ndiv(
            ndivs=ndivs,
            order=4,
            table=table,
            cfl=0.5,
            tf=0.5,
            sigma=0.35,
            radius=1.0,
            amplitude=1.0,
            alpha0=-np.pi / 4.0,
            u0=2.0 * np.pi / 10.0,
            lf_alpha=1.0,
            flux_type="central",
            volume_form="conservative",
            use_numba=False,
            history_every=10,
        )
        for ndivs in (1, 2, 4)
    ]
    return [result.row for result in results]


def test_table2_small_gaussian_convergence_regression():
    rows = _run_small_case("table2")

    assert all(np.isfinite(row.l2_error) for row in rows)
    assert all(np.isfinite(row.relative_l2_error) for row in rows)
    assert all(np.isfinite(row.linf_error) for row in rows)
    assert all(curr.hmin < prev.hmin for prev, curr in zip(rows[:-1], rows[1:]))
    assert all(curr.total_dofs > prev.total_dofs for prev, curr in zip(rows[:-1], rows[1:]))
    assert rows[-1].l2_error < rows[0].l2_error
    assert rows[-1].linf_error < rows[0].linf_error
    assert abs(rows[-1].relative_mass_drift) < 1e-10

    dicts = rows_to_dicts_with_rates(rows)
    expected_rate = np.log(rows[0].l2_error / rows[1].l2_error) / np.log(rows[0].hmin / rows[1].hmin)
    assert dicts[1]["l2_rate"] == pytest.approx(expected_rate)


def test_table1_small_gaussian_convergence_regression():
    rows = _run_small_case("table1")

    assert rows[0].n_points_per_element != 16
    assert all(np.isfinite(row.l2_error) for row in rows)
    assert all(np.isfinite(row.linf_error) for row in rows)
    assert rows[-1].l2_error < rows[0].l2_error
    assert rows[-1].linf_error < rows[0].linf_error
