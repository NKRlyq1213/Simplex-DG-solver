from pathlib import Path
import tempfile

import numpy as np

from simplex_dg.diagnostics import (
    ConvergenceRow,
    estimate_convergence_rates,
    format_convergence_table,
    rows_to_dicts_with_rates,
    write_convergence_csv,
)


def test_estimate_convergence_rates():
    values = [4.0, 1.0, 0.25]
    hmins = [1.0, 0.5, 0.25]
    rates = estimate_convergence_rates(values, hmins)

    assert rates[0] is None
    assert np.allclose(rates[1], 2.0)
    assert np.allclose(rates[2], 2.0)


def test_estimate_convergence_rates_uses_actual_hmin_ratio():
    values = [0.4, 0.1]
    hmins = [1.0, 1.0 / 3.0]
    rates = estimate_convergence_rates(values, hmins)

    assert rates[0] is None
    assert np.allclose(rates[1], np.log(4.0) / np.log(3.0))


def test_convergence_rows_to_dicts_with_rates():
    rows = [
        ConvergenceRow(
            ndivs=1,
            order=3,
            n_elements=8,
            n_points_per_element=18,
            total_dofs=144,
            dt=0.1,
            tf=1.0,
            nsteps=10,
            hmin=1.0,
            l2_error=0.4,
            relative_l2_error=0.2,
            linf_error=0.8,
            mass_drift=0.0,
            l2_norm_drift=0.0,
        ),
        ConvergenceRow(
            ndivs=3,
            order=3,
            n_elements=72,
            n_points_per_element=18,
            total_dofs=1296,
            dt=0.05,
            tf=1.0,
            nsteps=20,
            hmin=1.0 / 3.0,
            l2_error=0.1,
            relative_l2_error=0.05,
            linf_error=0.2,
            mass_drift=0.0,
            l2_norm_drift=0.0,
        ),
    ]

    dicts = rows_to_dicts_with_rates(rows)

    assert dicts[0]["l2_rate"] == ""
    assert np.allclose(dicts[1]["l2_rate"], np.log(4.0) / np.log(3.0))


def test_write_convergence_csv():
    row = ConvergenceRow(
        ndivs=1,
        order=3,
        n_elements=8,
        n_points_per_element=18,
        total_dofs=144,
        dt=0.1,
        tf=1.0,
        nsteps=10,
        hmin=1.0,
        l2_error=0.4,
        relative_l2_error=0.2,
        linf_error=0.8,
        mass_drift=0.0,
        l2_norm_drift=0.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "conv.csv"
        write_convergence_csv(path, [row])

        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "l2_error" in text
        assert "relative_l2_error" in text


def test_format_convergence_table_nonempty():
    row = ConvergenceRow(
        ndivs=1,
        order=3,
        n_elements=8,
        n_points_per_element=18,
        total_dofs=144,
        dt=0.1,
        tf=1.0,
        nsteps=10,
        hmin=1.0,
        l2_error=0.4,
        relative_l2_error=0.2,
        linf_error=0.8,
        mass_drift=0.0,
        l2_norm_drift=0.0,
    )

    table = format_convergence_table([row])

    assert "ndivs" in table
    assert "L2 err" in table
