from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from examples import step9_gaussian_convergence as step9


def _short_run_kwargs(**overrides):
    kwargs = {
        "ndivs": 1,
        "order": 4,
        "table": "table1",
        "cfl": 0.5,
        "tf": 0.05,
        "sigma": 0.35,
        "radius": 1.0,
        "amplitude": 1.0,
        "alpha0": -np.pi / 4.0,
        "u0": 2.0 * np.pi / 10.0,
        "lf_alpha": 1.0,
        "flux_type": "central",
        "volume_form": "conservative",
        "use_numba": False,
        "history_every": 1,
    }
    kwargs.update(overrides)
    return kwargs


def _assert_rows_match(a, b, *, atol: float, rtol: float):
    assert a.n_elements == b.n_elements
    assert a.total_dofs == b.total_dofs
    assert a.nsteps == b.nsteps
    assert a.n_points_per_element == b.n_points_per_element
    assert a.ndivs == b.ndivs
    assert a.order == b.order
    assert a.dt == pytest.approx(b.dt, abs=atol, rel=rtol)
    assert a.l2_error == pytest.approx(b.l2_error, abs=atol, rel=rtol)
    assert a.relative_l2_error == pytest.approx(b.relative_l2_error, abs=atol, rel=rtol)
    assert a.linf_error == pytest.approx(b.linf_error, abs=atol, rel=rtol)
    assert a.initial_mass == pytest.approx(b.initial_mass, abs=atol, rel=rtol)
    assert a.final_mass == pytest.approx(b.final_mass, abs=atol, rel=rtol)
    assert a.initial_energy == pytest.approx(b.initial_energy, abs=atol, rel=rtol)
    assert a.final_energy == pytest.approx(b.final_energy, abs=atol, rel=rtol)


def _fake_run_result() -> step9.RunOneNdivResult:
    row = step9.ConvergenceRow(
        ndivs=1,
        order=4,
        n_elements=8,
        n_points_per_element=16,
        total_dofs=128,
        dt=0.05,
        tf=0.05,
        nsteps=1,
        hmin=0.25,
        l2_error=1.0e-2,
        relative_l2_error=2.0e-2,
        linf_error=3.0e-2,
        mass_drift=0.0,
        l2_norm_drift=0.0,
        initial_mass=1.0,
        final_mass=1.0,
        absolute_mass_drift=0.0,
        relative_mass_drift=0.0,
        initial_energy=0.5,
        final_energy=0.5,
        absolute_energy_drift=0.0,
        relative_energy_drift=0.0,
        q_min=0.0,
        q_max=1.0,
        undershoot=0.0,
        overshoot=0.0,
        elapsed_seconds=0.01,
    )
    history = [
        {
            "ndivs": 1.0,
            "t": 0.0,
            "l2_error": row.l2_error,
            "relative_l2_error": row.relative_l2_error,
            "linf_error": row.linf_error,
            "mass": row.final_mass,
            "l2_norm": np.sqrt(2.0 * row.final_energy),
            "energy": row.final_energy,
            "relative_mass_error": row.relative_mass_drift,
            "relative_energy_error": row.relative_energy_drift,
            "signed_relative_mass_error": row.relative_mass_drift,
            "signed_relative_energy_error": row.relative_energy_drift,
            "q_min": row.q_min,
            "q_max": row.q_max,
            "undershoot": row.undershoot,
            "overshoot": row.overshoot,
        }
    ]
    q = np.ones((8, 16), dtype=float)
    return step9.RunOneNdivResult(row=row, history=history, q0=q, q_final=q, q_exact=q)


def test_parse_args_defaults_to_projected_sbp():
    args = step9.parse_args([])

    assert args.sbp == "projected"


@pytest.mark.parametrize("sbp_variant", ["projected", "full-raw", "full-orth"])
def test_parse_args_accepts_explicit_sbp_choices(sbp_variant: str):
    args = step9.parse_args(["--sbp", sbp_variant])

    assert args.sbp == sbp_variant


def test_parse_args_rejects_invalid_sbp_choice():
    with pytest.raises(SystemExit):
        step9.parse_args(["--sbp", "invalid"])


@pytest.mark.parametrize("sbp_variant", ["full-raw", "full-orth"])
def test_parse_args_rejects_table2_full_variants_with_clear_message(
    sbp_variant: str,
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(SystemExit):
        step9.parse_args(["--table", "table2", "--sbp", sbp_variant])

    captured = capsys.readouterr()
    assert "require --table table1" in captured.err
    assert "boundary volume nodes" in captured.err


def test_scheme_output_and_plot_paths_include_sbp_variant():
    projected_output = step9.output_path_for_scheme(
        "outputs/convergence/gaussian.csv",
        table="table1",
        sbp_variant="projected",
        volume_form="conservative",
        flux_type="central",
    )
    full_output = step9.output_path_for_scheme(
        "outputs/convergence/gaussian.csv",
        table="table1",
        sbp_variant="full-orth",
        volume_form="conservative",
        flux_type="central",
    )
    projected_plots = step9.plot_dir_for_scheme(
        "outputs/convergence/plots",
        table="table1",
        sbp_variant="projected",
        volume_form="conservative",
        flux_type="central",
    )
    full_plots = step9.plot_dir_for_scheme(
        "outputs/convergence/plots",
        table="table1",
        sbp_variant="full-orth",
        volume_form="conservative",
        flux_type="central",
    )

    assert projected_output != full_output
    assert "projected" in projected_output.name
    assert "full-orth" in full_output.name
    assert projected_plots != full_plots
    assert projected_plots.name == "table1_projected_conservative_central"
    assert full_plots.name == "table1_full-orth_conservative_central"


def test_main_propagates_sbp_variant_and_writes_variant_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    captured_kwargs: dict[str, object] = {}

    def fake_run_one_ndiv(**kwargs):
        captured_kwargs.update(kwargs)
        return _fake_run_result()

    monkeypatch.setattr(step9, "run_one_ndiv", fake_run_one_ndiv)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        output_base = Path(tmpdir) / "gaussian.csv"
        plot_base = Path(tmpdir) / "plots"

        step9.main(
            [
                "--ndivs",
                "1",
                "--table",
                "table1",
                "--sbp",
                "full-orth",
                "--form",
                "conservative",
                "--flux",
                "central",
                "--output",
                str(output_base),
                "--plot-dir",
                str(plot_base),
                "--no-plots",
            ]
        )

        output_path = Path(tmpdir) / "gaussian_table1_full-orth_conservative_central.csv"
        metadata_path = Path(tmpdir) / "gaussian_table1_full-orth_conservative_central_metadata.json"

        assert captured_kwargs["sbp_variant"] == "full-orth"
        assert output_path.exists()
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["sbp_variant"] == "full-orth"
        assert metadata["scheme_id"] == "table1_full-orth_conservative_central"
        assert metadata["output_csv"] == str(output_path)
        assert metadata["plot_dir"] is None

    captured = capsys.readouterr()
    assert "sbp           : full-orth" in captured.out
    assert "gaussian_table1_full-orth_conservative_central.csv" in captured.out


def test_run_one_ndiv_propagates_sbp_variant_to_reference_builder(monkeypatch: pytest.MonkeyPatch):
    captured_variant: dict[str, str] = {}
    real_build_reference_cache = step9.build_reference_cache

    def wrapped_build_reference_cache(*args, **kwargs):
        captured_variant["sbp_variant"] = kwargs["sbp_variant"]
        return real_build_reference_cache(*args, **kwargs)

    monkeypatch.setattr(step9, "build_reference_cache", wrapped_build_reference_cache)

    result = step9.run_one_ndiv(**_short_run_kwargs(order=1, sbp_variant="full-orth"))

    assert captured_variant["sbp_variant"] == "full-orth"
    assert result.row.nsteps > 0


def test_run_one_ndiv_default_and_explicit_projected_match():
    result_default = step9.run_one_ndiv(**_short_run_kwargs())
    result_projected = step9.run_one_ndiv(**_short_run_kwargs(sbp_variant="projected"))

    _assert_rows_match(result_default.row, result_projected.row, atol=5e-12, rtol=5e-12)
    np.testing.assert_allclose(result_default.q_final, result_projected.q_final, atol=5e-12, rtol=5e-12)
    np.testing.assert_allclose(result_default.q_exact, result_projected.q_exact, atol=0.0, rtol=0.0)


def test_run_one_ndiv_full_raw_and_orth_smoke_match():
    result_raw = step9.run_one_ndiv(**_short_run_kwargs(sbp_variant="full-raw"))
    result_orth = step9.run_one_ndiv(**_short_run_kwargs(sbp_variant="full-orth"))

    _assert_rows_match(result_raw.row, result_orth.row, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(result_raw.q_final, result_orth.q_final, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(result_raw.q_exact, result_orth.q_exact, atol=0.0, rtol=0.0)
