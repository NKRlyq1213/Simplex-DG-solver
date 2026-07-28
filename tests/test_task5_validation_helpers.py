from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from examples import step9_gaussian_convergence as step9
from scripts import run_task5_full_sbp_validation as task5


def _case(**overrides) -> task5.Task5Case:
    kwargs = {
        "ndivs": 1,
        "table": "table1",
        "order": 4,
        "sbp_variant": "projected",
        "cfl": 0.5,
        "tf": 0.05,
        "flux_type": "central",
        "volume_form": "conservative",
        "use_numba": False,
        "history_every": 1,
    }
    kwargs.update(overrides)
    return task5.Task5Case(**kwargs)


def test_run_step9_case_capture_states_is_read_only():
    case = _case()
    direct = step9.run_one_ndiv(**case.run_kwargs())
    run = task5.run_step9_case(case, capture_states=True)

    assert run.result.row.nsteps == direct.row.nsteps
    assert len(run.states) == len(run.result.history)
    assert run.actual_tf == pytest.approx(case.tf)
    assert run.min_dt == pytest.approx(run.result.row.dt)
    assert run.max_dt == pytest.approx(run.result.row.dt)

    np.testing.assert_allclose(run.result.q_final, direct.q_final, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(run.result.q_exact, direct.q_exact, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(run.states[-1], run.result.q_final, atol=0.0, rtol=0.0)


def test_compare_time_histories_full_raw_and_orth_multistep():
    raw_case = _case(sbp_variant="full-raw", tf=0.5, cfl=1.0)
    orth_case = _case(sbp_variant="full-orth", tf=0.5, cfl=1.0)

    raw = task5.run_step9_case(raw_case, capture_states=True)
    orth = task5.run_step9_case(orth_case, capture_states=True)
    comparison = task5.compare_time_histories(raw, orth)

    assert raw.result.row.nsteps > 1
    assert orth.result.row.nsteps > 1
    assert comparison["same_nsteps"] is True
    assert comparison["time_grid_max_abs_diff"] == pytest.approx(0.0, abs=0.0, rel=0.0)
    assert comparison["max_time_state_inf_diff"] < 1.0e-11
    assert comparison["final_state_inf_diff"] < 1.0e-11
    assert comparison["max_time_mass_abs_diff"] < 1.0e-12
    assert comparison["max_time_energy_abs_diff"] < 1.0e-12


def test_compute_product_rule_residual_is_finite_and_raw_matches_orth():
    projected = task5.compute_product_rule_residual(_case(ndivs=2, sbp_variant="projected"))
    raw = task5.compute_product_rule_residual(_case(ndivs=2, sbp_variant="full-raw"))
    orth = task5.compute_product_rule_residual(_case(ndivs=2, sbp_variant="full-orth"))

    assert projected["norm_scaling"] == "H=|T|W"
    assert np.isfinite(projected["tau_r_l2"])
    assert np.isfinite(projected["tau_s_l2"])
    assert np.isfinite(raw["tau_r_l2"])
    assert np.isfinite(raw["tau_s_l2"])
    assert np.isfinite(orth["tau_r_l2"])
    assert np.isfinite(orth["tau_s_l2"])

    assert abs(raw["tau_r_l2"] - orth["tau_r_l2"]) < 1.0e-12
    assert abs(raw["tau_s_l2"] - orth["tau_s_l2"]) < 1.0e-12
    assert abs(raw["tau_r_max_abs"] - orth["tau_r_max_abs"]) < 1.0e-12
    assert abs(raw["tau_s_max_abs"] - orth["tau_s_max_abs"]) < 1.0e-12


def test_run_selected_phases_writes_summary_schema(monkeypatch: pytest.MonkeyPatch):
    def fake_phase(layout, **kwargs):
        return {
            "rows": [
                {
                    "scheme_id": "table1_projected_conservative_central",
                    "ndivs": 4,
                    "relative_l2_error": 1.0e-2,
                }
            ],
            "failures": [],
        }

    monkeypatch.setattr(task5, "phase_equivalence", fake_phase)
    monkeypatch.setattr(task5, "warm_numba_if_requested", lambda use_numba: None)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        payload = task5.run_selected_phases(
            task5.parse_args(
                [
                    "--phase",
                    "equivalence",
                    "--output-dir",
                    str(Path(tmpdir)),
                ]
            )
        )

        summary_csv = Path(tmpdir) / "summary.csv"
        summary_json = Path(tmpdir) / "summary.json"
        report_md = Path(tmpdir) / "report.md"

        assert summary_csv.exists()
        assert summary_json.exists()
        assert report_md.exists()

        data = json.loads(summary_json.read_text(encoding="utf-8"))
        assert "environment" in data
        assert "results" in data
        assert "equivalence" in data["results"]
        assert payload["results"]["report_path"] == str(report_md)
