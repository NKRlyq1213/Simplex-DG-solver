from __future__ import annotations

import numpy as np
import pytest

from examples import step9_gaussian_convergence as step9


FLUX_CASES = [
    ("central", 1.0),
    ("upwind", 1.0),
    ("lf", 1.0),
    ("lf", 2.0),
]


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_gaussian_short_run_completes(volume_form, flux_type, lf_alpha):
    result = step9.run_one_ndiv(
        ndivs=1,
        order=4,
        table="table2",
        cfl=0.5,
        tf=0.05,
        sigma=0.35,
        radius=1.0,
        amplitude=1.0,
        alpha0=-np.pi / 4.0,
        u0=2.0 * np.pi / 10.0,
        lf_alpha=lf_alpha,
        flux_type=flux_type,
        volume_form=volume_form,
        use_numba=False,
        history_every=1,
    )

    row = result.row

    assert row.n_points_per_element == 16
    assert row.n_elements == 8
    assert row.nsteps > 0
    assert row.dt > 0.0
    assert np.all(np.isfinite(result.q_final))
    assert np.all(np.isfinite(result.q_exact))
    assert np.isfinite(row.l2_error)
    assert np.isfinite(row.relative_l2_error)
    assert np.isfinite(row.linf_error)
    assert np.isfinite(row.relative_mass_drift)
    assert np.isfinite(row.relative_energy_drift)
    assert np.isfinite(row.q_min)
    assert np.isfinite(row.q_max)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
@pytest.mark.parametrize(("flux_type", "lf_alpha"), FLUX_CASES)
def test_table2_gaussian_short_run_numpy_and_numba_match(volume_form, flux_type, lf_alpha):
    pytest.importorskip("numba")

    kwargs = dict(
        ndivs=1,
        order=4,
        table="table2",
        cfl=0.5,
        tf=0.05,
        sigma=0.35,
        radius=1.0,
        amplitude=1.0,
        alpha0=-np.pi / 4.0,
        u0=2.0 * np.pi / 10.0,
        lf_alpha=lf_alpha,
        flux_type=flux_type,
        volume_form=volume_form,
        history_every=1,
    )

    result_np = step9.run_one_ndiv(use_numba=False, **kwargs)
    result_nb = step9.run_one_ndiv(use_numba=True, **kwargs)

    np.testing.assert_allclose(result_nb.q_final, result_np.q_final, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(result_nb.q_exact, result_np.q_exact, atol=0.0, rtol=0.0)
    assert result_nb.row.nsteps == result_np.row.nsteps
    assert result_nb.row.dt == pytest.approx(result_np.row.dt)
    assert result_nb.row.l2_error == pytest.approx(result_np.row.l2_error, abs=2e-12, rel=2e-12)
    assert result_nb.row.relative_l2_error == pytest.approx(result_np.row.relative_l2_error, abs=2e-12, rel=2e-12)
    assert result_nb.row.linf_error == pytest.approx(result_np.row.linf_error, abs=2e-12, rel=2e-12)
