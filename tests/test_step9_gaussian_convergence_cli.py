from __future__ import annotations

import json
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes

from examples import step9_gaussian_convergence as step9


@pytest.mark.parametrize(
    ("span_seconds", "expected_unit", "expected_scale", "expected_label"),
    [
        (600.0, "second", 1.0, "time (seconds)"),
        (601.0, "minute", 60.0, "time (minutes)"),
        (7200.0, "minute", 60.0, "time (minutes)"),
        (7201.0, "hour", 3600.0, "time (hours)"),
        (86400.0, "hour", 3600.0, "time (hours)"),
        (86401.0, "day", 86400.0, "time (days)"),
    ],
)
def test_resolve_plot_time_axis_auto_boundaries(
    span_seconds: float,
    expected_unit: str,
    expected_scale: float,
    expected_label: str,
):
    axis = step9.resolve_plot_time_axis(span_seconds=span_seconds)

    assert axis.unit == expected_unit
    assert axis.scale == expected_scale
    assert axis.xlabel == expected_label


@pytest.mark.parametrize(
    ("plot_time_unit", "expected_scale", "expected_label"),
    [
        ("second", 1.0, "time (seconds)"),
        ("minute", 60.0, "time (minutes)"),
        ("hour", 3600.0, "time (hours)"),
        ("day", 86400.0, "time (days)"),
    ],
)
def test_resolve_plot_time_axis_explicit_override(
    plot_time_unit: str,
    expected_scale: float,
    expected_label: str,
):
    axis = step9.resolve_plot_time_axis(span_seconds=999999.0, plot_time_unit=plot_time_unit)

    assert axis.unit == plot_time_unit
    assert axis.scale == expected_scale
    assert axis.xlabel == expected_label


def test_resolve_history_time_axis_empty_histories_defaults_to_seconds():
    axis = step9.resolve_history_time_axis([], plot_time_unit="auto")

    assert axis.unit == "second"
    assert axis.scale == 1.0
    assert axis.xlabel == "time (seconds)"


def test_parse_args_defaults_plot_time_unit_auto():
    args = step9.parse_args([])

    assert args.plot_time_unit == "auto"
    assert args.ndivs == [1, 2, 4, 8]
    assert args.table == "table1"


@pytest.mark.parametrize("plot_time_unit", ["second", "minute", "hour", "day"])
def test_parse_args_accepts_plot_time_unit_choices(plot_time_unit: str):
    args = step9.parse_args(["--plot-time-unit", plot_time_unit])

    assert args.plot_time_unit == plot_time_unit


@pytest.mark.parametrize(
    "ndivs_args",
    [
        ["--ndivs", "1", "2", "3", "6", "12"],
        ["--ndivs", "2", "5"],
    ],
)
def test_parse_args_accepts_custom_ndivs_sequences(ndivs_args: list[str]):
    args = step9.parse_args(ndivs_args)

    assert args.ndivs == [int(value) for value in ndivs_args[1:]]


def test_parse_args_accepts_table2_and_expression_arguments():
    args = step9.parse_args(
        [
            "--table",
            "table2",
            "--alpha0",
            "-pi/4",
            "--u0",
            "2*pi/10",
        ]
    )

    assert args.table == "table2"
    assert args.alpha0 == pytest.approx(-np.pi / 4.0)
    assert args.u0 == pytest.approx(2.0 * np.pi / 10.0)


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--ndivs", "1", "1", "2"],
        ["--ndivs", "1", "3", "2"],
        ["--ndivs", "0", "2"],
        ["--ndivs", "-1", "2"],
        ["--levels", "0", "1", "2"],
    ],
)
def test_parse_args_rejects_invalid_or_removed_ndivs_inputs(bad_args: list[str]):
    with pytest.raises(SystemExit):
        step9.parse_args(bad_args)


def test_plot_time_history_quantity_scales_combined_histories_and_sets_xlabel(
    monkeypatch: pytest.MonkeyPatch,
):
    histories = {
        0: [
            {"t": 0.0, "relative_l2_error": 1.0e-2},
            {"t": 601.0, "relative_l2_error": 1.0e-3},
        ],
        1: [
            {"t": 10.0, "relative_l2_error": 2.0e-2},
            {"t": 400.0, "relative_l2_error": 2.0e-3},
        ],
    }
    captured_x: list[np.ndarray] = []
    captured_labels: list[str] = []
    original_semilogy = Axes.semilogy
    original_set_xlabel = Axes.set_xlabel

    def record_semilogy(self, x, y, *args, **kwargs):
        captured_x.append(np.asarray(x, dtype=float).copy())
        return original_semilogy(self, x, y, *args, **kwargs)

    def record_set_xlabel(self, xlabel, *args, **kwargs):
        captured_labels.append(str(xlabel))
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    monkeypatch.setattr(Axes, "semilogy", record_semilogy)
    monkeypatch.setattr(Axes, "set_xlabel", record_set_xlabel)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        output_path = Path(tmpdir) / "combined.png"

        step9.plot_time_history_quantity(
            histories=histories,
            output_path=output_path,
            quantity="relative_l2_error",
            ylabel="relative L2 error",
            title="Gaussian advection relative L2 error history",
            semilogy=True,
            plot_time_unit="auto",
        )

        assert output_path.exists()

    assert captured_labels[-1] == "time (minutes)"
    assert np.allclose(captured_x[0], [0.0, 601.0 / 60.0])
    assert np.allclose(captured_x[1], [10.0 / 60.0, 400.0 / 60.0])


def test_plot_time_history_quantity_each_level_uses_per_level_time_units(
    monkeypatch: pytest.MonkeyPatch,
):
    histories = {
        0: [
            {"t": 0.0, "relative_energy_error": 1.0e-2},
            {"t": 601.0, "relative_energy_error": 1.0e-3},
        ],
        1: [
            {"t": 0.0, "relative_energy_error": 2.0e-2},
            {"t": 7201.0, "relative_energy_error": 2.0e-3},
        ],
    }

    captured_x: list[np.ndarray] = []
    captured_labels: list[str] = []
    original_semilogy = Axes.semilogy
    original_set_xlabel = Axes.set_xlabel

    def record_semilogy(self, x, y, *args, **kwargs):
        captured_x.append(np.asarray(x, dtype=float).copy())
        return original_semilogy(self, x, y, *args, **kwargs)

    def record_set_xlabel(self, xlabel, *args, **kwargs):
        captured_labels.append(str(xlabel))
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    monkeypatch.setattr(Axes, "semilogy", record_semilogy)
    monkeypatch.setattr(Axes, "set_xlabel", record_set_xlabel)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        output_paths = step9.plot_time_history_quantity_each_level(
            histories=histories,
            output_dir=Path(tmpdir),
            filename_prefix="gaussian_rel_energy_error_history",
            quantity="relative_energy_error",
            ylabel="relative energy error",
            title_prefix="Relative energy error history",
            semilogy=True,
            plot_time_unit="auto",
        )

        assert len(output_paths) == 2
        assert all(path.exists() for path in output_paths)

    assert captured_labels == ["time (minutes)", "time (hours)"]
    assert np.allclose(captured_x[0], [0.0, 601.0 / 60.0])
    assert np.allclose(captured_x[1], [0.0, 7201.0 / 3600.0])


def test_plot_observed_order_uses_ndivs_on_x_axis(
    monkeypatch: pytest.MonkeyPatch,
):
    rows = [
        step9.ConvergenceRow(
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
        step9.ConvergenceRow(
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

    captured_x: list[np.ndarray] = []
    captured_labels: list[str] = []
    original_plot = Axes.plot
    original_set_xlabel = Axes.set_xlabel

    def record_plot(self, x, y, *args, **kwargs):
        captured_x.append(np.asarray(x, dtype=float).copy())
        return original_plot(self, x, y, *args, **kwargs)

    def record_set_xlabel(self, xlabel, *args, **kwargs):
        captured_labels.append(str(xlabel))
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)
    monkeypatch.setattr(Axes, "set_xlabel", record_set_xlabel)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        output_path = Path(tmpdir) / "observed_order.png"
        step9.plot_observed_order(rows=rows, output_path=output_path)
        assert output_path.exists()

    assert captured_labels[-1] == "ndivs"
    assert np.allclose(captured_x[0], [3.0])


def test_metadata_sidecar_path_and_json_write():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        output_path = Path(tmpdir) / "convergence.csv"
        metadata_path = step9.metadata_path_from_output(output_path)

        assert metadata_path.name == "convergence_metadata.json"

        step9.write_metadata_json(
            metadata_path,
            {
                "table": "table2",
                "ndivs": [1, 2, 4],
                "git_commit": "abc123",
            },
        )

        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert data["table"] == "table2"
        assert data["ndivs"] == [1, 2, 4]
        assert data["git_commit"] == "abc123"
