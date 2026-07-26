from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def test_leibniz_defect_cli_smoke_run_writes_outputs():
    script = Path("examples/check_leibniz_defect_convergence.py")

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_csv = tmpdir_path / "leibniz.csv"
        metadata_json = tmpdir_path / "leibniz_metadata.json"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--ndivs",
                "1",
                "2",
                "--order",
                "2",
                "--no-plots",
                "--output",
                str(output_csv),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert output_csv.exists()
        assert metadata_json.exists()
        assert "Table A: reference-form defects" in result.stdout
        assert "Table B: physical defects and energy residual" in result.stdout
        assert "max projection closure error" in result.stdout

        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2

        required_columns = {
            "state",
            "q_projection_closure_linf",
            "tau_r_linf",
            "tau_r_l2_ref",
            "tau_s_linf",
            "tau_s_l2_ref",
            "tau_sum_linf",
            "tau_sum_l2_ref",
            "physical_tau_r_linf",
            "physical_tau_r_l2",
            "physical_tau_s_linf",
            "physical_tau_s_l2",
            "physical_tau_sum_linf",
            "physical_tau_sum_l2",
            "energy",
            "abs_energy_residual",
            "relative_energy_residual",
            "tau_sum_l2_ref_rate",
            "physical_tau_sum_l2_rate",
        }
        assert required_columns.issubset(rows[0].keys())

        nonnegative_fields = [
            "q_projection_closure_linf",
            "tau_r_linf",
            "tau_r_l2_ref",
            "tau_s_linf",
            "tau_s_l2_ref",
            "tau_sum_linf",
            "tau_sum_l2_ref",
            "physical_tau_r_linf",
            "physical_tau_r_l2",
            "physical_tau_s_linf",
            "physical_tau_s_l2",
            "physical_tau_sum_linf",
            "physical_tau_sum_l2",
            "energy",
            "abs_energy_residual",
            "relative_energy_residual",
        ]

        for row in rows:
            assert row["state"] == "projected-gaussian"
            for field_name in nonnegative_fields:
                value = float(row[field_name])
                assert np.isfinite(value), field_name
                assert value >= 0.0, field_name

        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
        assert metadata["order"] == 2
        assert metadata["table"] == "table1"
        assert metadata["ndivs"] == [1, 2]
        assert metadata["state"] == "projected-gaussian"
        assert metadata["project_velocity"] is True
