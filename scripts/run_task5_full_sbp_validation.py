from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from examples import step9_gaussian_convergence as step9
from simplex_dg.diagnostics import estimate_convergence_rates, rows_to_dicts_with_rates
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.problems import gaussian_on_sphere
from simplex_dg.reference import build_reference_cache
from simplex_dg.reference.quadrature import REFERENCE_AREA
from simplex_dg.rhs import build_full_rhs_cache
from simplex_dg.rhs.volume import apply_reference_operator, build_volume_rhs_cache
from simplex_dg.time import minimum_face_length


DEFAULT_OUTPUT_DIR = ROOT / "results" / "task5_full_sbp_validation"
DEFAULT_TABLE = "table1"
DEFAULT_ORDER = 4
DEFAULT_CFL = 1.0
DEFAULT_SIGMA_ANGLE = 0.35
DEFAULT_RADIUS = 1.0
DEFAULT_AMPLITUDE = 1.0
DEFAULT_ALPHA0 = -float(np.pi) / 4.0
DEFAULT_U0 = 1.0
DEFAULT_LF_ALPHA = 1.0
DEFAULT_HISTORY_EVERY = 1
DEFAULT_NUMPY_DTYPE = "float64"
FULL_VARIANTS = ("full-raw", "full-orth")
STUDY_VARIANTS = ("projected", "full-orth")
FLUXES = ("central", "upwind", "lf")
FORMS = ("conservative", "split")
PHASES = ("equivalence", "mass", "convergence", "stability", "product-rule", "all")


@dataclass(frozen=True)
class Task5Case:
    ndivs: int
    table: str = DEFAULT_TABLE
    order: int = DEFAULT_ORDER
    sbp_variant: str = "projected"
    cfl: float = DEFAULT_CFL
    tf: float = 1.0
    sigma_angle: float = DEFAULT_SIGMA_ANGLE
    radius: float = DEFAULT_RADIUS
    amplitude: float = DEFAULT_AMPLITUDE
    alpha0: float = DEFAULT_ALPHA0
    u0: float = DEFAULT_U0
    lf_alpha: float = DEFAULT_LF_ALPHA
    flux_type: str = "central"
    volume_form: str = "conservative"
    use_numba: bool = False
    history_every: int = DEFAULT_HISTORY_EVERY

    def sigma_physical(self) -> float:
        return step9.resolve_sigma_physical(
            radius=self.radius,
            sigma_angle=self.sigma_angle,
            sigma_physical=None,
        )

    def scheme_id(self) -> str:
        return step9.scheme_identifier(
            table=self.table,
            sbp_variant=self.sbp_variant,
            volume_form=self.volume_form,
            flux_type=self.flux_type,
        )

    def scheme_label(self) -> str:
        return step9.scheme_label(
            table=self.table,
            sbp_variant=self.sbp_variant,
            volume_form=self.volume_form,
            flux_type=self.flux_type,
        )

    def run_kwargs(self) -> dict[str, Any]:
        return {
            "ndivs": self.ndivs,
            "order": self.order,
            "table": self.table,
            "sbp_variant": self.sbp_variant,
            "cfl": self.cfl,
            "tf": self.tf,
            "sigma": self.sigma_physical(),
            "radius": self.radius,
            "amplitude": self.amplitude,
            "alpha0": self.alpha0,
            "u0": self.u0,
            "lf_alpha": self.lf_alpha,
            "flux_type": self.flux_type,
            "volume_form": self.volume_form,
            "use_numba": self.use_numba,
            "history_every": self.history_every,
        }


@dataclass
class Step9RunData:
    case: Task5Case
    result: step9.RunOneNdivResult
    times: np.ndarray
    l2_errors: np.ndarray
    relative_l2_errors: np.ndarray
    linf_errors: np.ndarray
    masses: np.ndarray
    energies: np.ndarray
    relative_mass_errors: np.ndarray
    relative_energy_errors: np.ndarray
    q_mins: np.ndarray
    q_maxs: np.ndarray
    states: list[np.ndarray] | None

    @property
    def scheme_id(self) -> str:
        return self.case.scheme_id()

    @property
    def actual_tf(self) -> float:
        return float(self.times[-1])

    @property
    def min_dt(self) -> float:
        if self.times.size <= 1:
            return float(self.result.row.dt)
        return float(np.min(np.diff(self.times)))

    @property
    def max_dt(self) -> float:
        if self.times.size <= 1:
            return float(self.result.row.dt)
        return float(np.max(np.diff(self.times)))

    @property
    def max_relative_mass_history_drift(self) -> float:
        return float(np.max(self.relative_mass_errors))

    @property
    def max_relative_energy_history_drift(self) -> float:
        return float(np.max(self.relative_energy_errors))

    @property
    def initial_energy(self) -> float:
        return float(self.energies[0])

    @property
    def final_energy(self) -> float:
        return float(self.energies[-1])

    @property
    def max_energy_ratio(self) -> float:
        denom = max(abs(self.initial_energy), np.finfo(float).tiny)
        return float(np.max(self.energies / denom))

    @property
    def min_energy_ratio(self) -> float:
        denom = max(abs(self.initial_energy), np.finfo(float).tiny)
        return float(np.min(self.energies / denom))

    @property
    def initial_mass(self) -> float:
        return float(self.masses[0])

    @property
    def final_mass(self) -> float:
        return float(self.masses[-1])

    @property
    def finite(self) -> bool:
        arrays = (
            self.times,
            self.l2_errors,
            self.relative_l2_errors,
            self.linf_errors,
            self.masses,
            self.energies,
            self.relative_mass_errors,
            self.relative_energy_errors,
            self.q_mins,
            self.q_maxs,
            self.result.q_final,
            self.result.q_exact,
        )
        return all(np.all(np.isfinite(arr)) for arr in arrays)


@dataclass(frozen=True)
class ProductRuleResidualRow:
    sbp_variant: str
    ndivs: int
    elements: int
    hmin: float
    tau_r_l2: float
    tau_s_l2: float
    rate_r: float | None
    rate_s: float | None
    norm_scaling: str = "H=|T|W"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 5 Step9 SBP validation orchestrator.")
    parser.add_argument("--phase", choices=PHASES, default="equivalence")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--use-numba", action="store_true")
    parser.add_argument("--include-ndivs64", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def ensure_output_layout(output_dir: Path) -> dict[str, Path]:
    layout = {
        "root": output_dir,
        "equivalence": output_dir / "raw_orth_equivalence",
        "mass": output_dir / "mass",
        "convergence": output_dir / "convergence",
        "stability": output_dir / "stability",
        "product-rule": output_dir / "product_rule",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        rows = [{"status": "empty"}]

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_existing_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_step9_case(case: Task5Case, *, capture_states: bool = False) -> Step9RunData:
    states: list[np.ndarray] | None = [] if capture_states else None

    def monitor_hook(t: float, q: np.ndarray, entry: dict[str, float]) -> None:
        if states is not None:
            states.append(np.asarray(q, dtype=float).copy())

    result = step9.run_one_ndiv(
        **case.run_kwargs(),
        monitor_hook=monitor_hook if capture_states else None,
    )
    history = result.history

    times = np.array([entry["t"] for entry in history], dtype=float)
    l2_errors = np.array([entry["l2_error"] for entry in history], dtype=float)
    relative_l2_errors = np.array([entry["relative_l2_error"] for entry in history], dtype=float)
    linf_errors = np.array([entry["linf_error"] for entry in history], dtype=float)
    masses = np.array([entry["mass"] for entry in history], dtype=float)
    energies = np.array([entry["energy"] for entry in history], dtype=float)
    relative_mass_errors = np.array([entry["relative_mass_error"] for entry in history], dtype=float)
    relative_energy_errors = np.array([entry["relative_energy_error"] for entry in history], dtype=float)
    q_mins = np.array([entry["q_min"] for entry in history], dtype=float)
    q_maxs = np.array([entry["q_max"] for entry in history], dtype=float)

    return Step9RunData(
        case=case,
        result=result,
        times=times,
        l2_errors=l2_errors,
        relative_l2_errors=relative_l2_errors,
        linf_errors=linf_errors,
        masses=masses,
        energies=energies,
        relative_mass_errors=relative_mass_errors,
        relative_energy_errors=relative_energy_errors,
        q_mins=q_mins,
        q_maxs=q_maxs,
        states=states,
    )


def build_run_summary_row(run: Step9RunData, *, phase: str) -> dict[str, Any]:
    row = run.result.row
    return {
        "phase": phase,
        "scheme_id": run.scheme_id,
        "sbp_variant": run.case.sbp_variant,
        "form": run.case.volume_form,
        "flux": run.case.flux_type,
        "table": run.case.table,
        "order": run.case.order,
        "ndivs": run.case.ndivs,
        "elements": row.n_elements,
        "hmin": row.hmin,
        "tf": run.case.tf,
        "actual_tf": run.actual_tf,
        "cfl": run.case.cfl,
        "use_numba": run.case.use_numba,
        "nsteps": row.nsteps,
        "dt": row.dt,
        "min_dt": run.min_dt,
        "max_dt": run.max_dt,
        "l2_error": row.l2_error,
        "relative_l2_error": row.relative_l2_error,
        "linf_error": row.linf_error,
        "initial_mass": row.initial_mass,
        "final_mass": row.final_mass,
        "absolute_mass_drift": row.absolute_mass_drift,
        "relative_mass_drift": row.relative_mass_drift,
        "max_relative_mass_history_drift": run.max_relative_mass_history_drift,
        "initial_energy": row.initial_energy,
        "final_energy": row.final_energy,
        "absolute_energy_drift": row.absolute_energy_drift,
        "relative_energy_drift": row.relative_energy_drift,
        "max_relative_energy_history_drift": run.max_relative_energy_history_drift,
        "max_energy_ratio": run.max_energy_ratio,
        "min_energy_ratio": run.min_energy_ratio,
        "q_min": row.q_min,
        "q_max": row.q_max,
        "elapsed_seconds": row.elapsed_seconds,
        "finite": run.finite,
    }


def compare_time_histories(raw: Step9RunData, orth: Step9RunData) -> dict[str, Any]:
    if raw.states is None or orth.states is None:
        raise ValueError("State histories are required for raw/orth comparison.")

    if len(raw.states) != len(orth.states):
        raise ValueError("raw/orth history lengths do not match.")

    q_diffs = [
        float(np.max(np.abs(q_raw - q_orth)))
        for q_raw, q_orth in zip(raw.states, orth.states)
    ]

    return {
        "sbp_variant_raw": raw.case.sbp_variant,
        "sbp_variant_orth": orth.case.sbp_variant,
        "scheme_id_raw": raw.scheme_id,
        "scheme_id_orth": orth.scheme_id,
        "ndivs": raw.case.ndivs,
        "form": raw.case.volume_form,
        "flux": raw.case.flux_type,
        "tf": raw.case.tf,
        "cfl": raw.case.cfl,
        "nsteps_raw": raw.result.row.nsteps,
        "nsteps_orth": orth.result.row.nsteps,
        "same_nsteps": raw.result.row.nsteps == orth.result.row.nsteps,
        "time_grid_max_abs_diff": float(np.max(np.abs(raw.times - orth.times))),
        "max_time_state_inf_diff": float(max(q_diffs)),
        "final_state_inf_diff": float(np.max(np.abs(raw.result.q_final - orth.result.q_final))),
        "max_time_mass_abs_diff": float(np.max(np.abs(raw.masses - orth.masses))),
        "max_time_energy_abs_diff": float(np.max(np.abs(raw.energies - orth.energies))),
        "final_relative_l2_error_abs_diff": float(
            abs(raw.result.row.relative_l2_error - orth.result.row.relative_l2_error)
        ),
        "final_linf_error_abs_diff": float(abs(raw.result.row.linf_error - orth.result.row.linf_error)),
        "final_relative_mass_drift_abs_diff": float(
            abs(raw.result.row.relative_mass_drift - orth.result.row.relative_mass_drift)
        ),
        "final_relative_energy_drift_abs_diff": float(
            abs(raw.result.row.relative_energy_drift - orth.result.row.relative_energy_drift)
        ),
        "min_dt_raw": raw.min_dt,
        "max_dt_raw": raw.max_dt,
        "min_dt_orth": orth.min_dt,
        "max_dt_orth": orth.max_dt,
    }


def weighted_reference_l2_global(values: np.ndarray, h_diag: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    weights = np.asarray(h_diag, dtype=float)

    if arr.ndim != 2:
        raise ValueError("values must have shape (K, Np).")

    if arr.shape[1] != weights.shape[0]:
        raise ValueError("values and h_diag shapes do not match.")

    return float(np.sqrt(np.sum((arr * arr) * weights[None, :])))


def compute_product_rule_residual(case: Task5Case) -> dict[str, Any]:
    ref = build_reference_cache(order=case.order, table=case.table, sbp_variant=case.sbp_variant, validate=True)
    mesh = build_octa_sphere_mesh(ndivs=case.ndivs, radius=case.radius)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    _ = conn
    geom = build_geometry_cache(mesh, ref, validate=True)
    volume = build_volume_rhs_cache(
        ref=ref,
        geom=geom,
        omega=tuple(float(case.u0) * a for a in step9.rotation_axis_from_alpha0(case.alpha0)),
        project_velocity=True,
        validate=True,
    )

    q = gaussian_on_sphere(
        X=geom.X,
        center=(case.radius, 0.0, 0.0),
        radius=case.radius,
        sigma=case.sigma_physical(),
        amplitude=case.amplitude,
    )

    dr_q = apply_reference_operator(ref.Dr, q)
    ds_q = apply_reference_operator(ref.Ds, q)
    dr_alpha_q = apply_reference_operator(ref.Dr, volume.alpha * q)
    ds_beta_q = apply_reference_operator(ref.Ds, volume.beta * q)

    tau_r = dr_alpha_q - volume.alpha * dr_q - q * volume.Dr_alpha
    tau_s = ds_beta_q - volume.beta * ds_q - q * volume.Ds_beta

    h_diag = REFERENCE_AREA * ref.weights

    return {
        "sbp_variant": case.sbp_variant,
        "ndivs": case.ndivs,
        "elements": int(mesh.elements.shape[0]),
        "hmin": float(minimum_face_length(ref, geom)),
        "tau_r_l2": weighted_reference_l2_global(tau_r, h_diag),
        "tau_s_l2": weighted_reference_l2_global(tau_s, h_diag),
        "norm_scaling": "H=|T|W",
        "tau_r_max_abs": float(np.max(np.abs(tau_r))),
        "tau_s_max_abs": float(np.max(np.abs(tau_s))),
    }


def plot_energy_ratio_histories(
    runs: list[Step9RunData],
    *,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    for run in runs:
        denom = max(abs(run.initial_energy), np.finfo(float).tiny)
        ratio = run.energies / denom
        label = f"{run.case.sbp_variant} / {run.case.flux_type}"
        ax.plot(run.times, ratio, linewidth=1.4, label=label)

    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("E(t) / E(0)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def environment_summary(*, use_numba: bool) -> dict[str, Any]:
    try:
        import numba as numba_mod
        numba_version = numba_mod.__version__
    except Exception:
        numba_version = None

    try:
        git_status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except Exception:
        git_status = []

    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "numba_used": bool(use_numba),
        "numba_version": numba_version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "dtype": DEFAULT_NUMPY_DTYPE,
        "git_commit": step9.current_git_commit(),
        "git_status_short": git_status,
    }


def warm_numba_if_requested(use_numba: bool) -> None:
    if not use_numba:
        return

    case = Task5Case(
        ndivs=1,
        table="table1",
        order=4,
        sbp_variant="projected",
        cfl=0.5,
        tf=0.05,
        flux_type="central",
        volume_form="conservative",
        use_numba=True,
    )
    run_step9_case(case, capture_states=False)


def phase_equivalence(layout: dict[str, Path], *, use_numba: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    tf = 0.5
    cfl = 1.0

    for ndivs in (1, 2):
        for volume_form in FORMS:
            for flux in FLUXES:
                base_kwargs = dict(
                    ndivs=ndivs,
                    sbp_variant="full-raw",
                    cfl=cfl,
                    tf=tf,
                    flux_type=flux,
                    volume_form=volume_form,
                    use_numba=use_numba,
                    history_every=1,
                )
                raw_case = Task5Case(**base_kwargs)
                orth_case = Task5Case(**{**base_kwargs, "sbp_variant": "full-orth"})

                try:
                    raw_run = run_step9_case(raw_case, capture_states=True)
                    orth_run = run_step9_case(orth_case, capture_states=True)
                    if raw_run.result.row.nsteps <= 1 or orth_run.result.row.nsteps <= 1:
                        raise RuntimeError("Equivalence phase did not produce a multistep time history.")
                    row = compare_time_histories(raw_run, orth_run)
                    rows.append(row)
                except Exception as exc:
                    failures.append(
                        {
                            "phase": "equivalence",
                            "ndivs": ndivs,
                            "form": volume_form,
                            "flux": flux,
                            "error": str(exc),
                        }
                    )

    csv_path = layout["equivalence"] / "equivalence.csv"
    write_rows_csv(csv_path, rows)
    write_json(layout["equivalence"] / "equivalence.json", {"rows": rows, "failures": failures, "tf": tf, "cfl": cfl})
    return {"rows": rows, "failures": failures, "csv": str(csv_path), "tf": tf, "cfl": cfl}


def phase_mass(layout: dict[str, Path], *, use_numba: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for sbp_variant in STUDY_VARIANTS:
        for volume_form in FORMS:
            for flux in FLUXES:
                for ndivs in (2, 4, 8):
                    case = Task5Case(
                        ndivs=ndivs,
                        sbp_variant=sbp_variant,
                        flux_type=flux,
                        volume_form=volume_form,
                        use_numba=use_numba,
                        history_every=1,
                        tf=1.0,
                        cfl=1.0,
                    )
                    try:
                        run = run_step9_case(case, capture_states=False)
                        row = build_run_summary_row(run, phase="mass")
                        rows.append(row)
                    except Exception as exc:
                        failures.append({"phase": "mass", "scheme_id": case.scheme_id(), "error": str(exc)})

    csv_path = layout["mass"] / "mass.csv"
    write_rows_csv(csv_path, rows)
    write_json(layout["mass"] / "mass.json", {"rows": rows, "failures": failures})
    return {"rows": rows, "failures": failures, "csv": str(csv_path)}


def phase_convergence(layout: dict[str, Path], *, use_numba: bool, include_ndivs64: bool) -> dict[str, Any]:
    rows_by_scheme: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    scheme_csv_paths: list[str] = []
    ndivs_values = [4, 8, 16, 32]
    skipped: list[dict[str, Any]] = []

    if include_ndivs64:
        ndivs_values.append(64)
    else:
        skipped.append({"phase": "convergence", "scheme": "all", "reason": "ndivs=64 not requested"})

    for sbp_variant in STUDY_VARIANTS:
        for volume_form in FORMS:
            for flux in FLUXES:
                scheme_runs: list[Step9RunData] = []
                scheme_id = step9.scheme_identifier(
                    table="table1",
                    sbp_variant=sbp_variant,
                    volume_form=volume_form,
                    flux_type=flux,
                )
                for ndivs in ndivs_values:
                    case = Task5Case(
                        ndivs=ndivs,
                        sbp_variant=sbp_variant,
                        flux_type=flux,
                        volume_form=volume_form,
                        use_numba=use_numba,
                        history_every=1,
                        tf=1.0,
                        cfl=1.0,
                    )
                    try:
                        scheme_runs.append(run_step9_case(case, capture_states=False))
                    except Exception as exc:
                        failures.append({"phase": "convergence", "scheme_id": scheme_id, "ndivs": ndivs, "error": str(exc)})
                        break

                if len(scheme_runs) < 4:
                    continue

                row_dicts = rows_to_dicts_with_rates([run.result.row for run in scheme_runs])
                scheme_rows: list[dict[str, Any]] = []
                for row_dict, run in zip(row_dicts, scheme_runs):
                    merged = dict(row_dict)
                    merged.update(
                        {
                            "scheme_id": scheme_id,
                            "table": run.case.table,
                            "sbp_variant": run.case.sbp_variant,
                            "flux": run.case.flux_type,
                            "form": run.case.volume_form,
                            "actual_tf": run.actual_tf,
                            "min_dt": run.min_dt,
                            "max_dt": run.max_dt,
                            "max_relative_mass_history_drift": run.max_relative_mass_history_drift,
                            "max_relative_energy_history_drift": run.max_relative_energy_history_drift,
                            "max_energy_ratio": run.max_energy_ratio,
                            "min_energy_ratio": run.min_energy_ratio,
                            "finite": run.finite,
                        }
                    )
                    scheme_rows.append(merged)
                rows_by_scheme[scheme_id] = scheme_rows
                csv_path = layout["convergence"] / f"{scheme_id}_convergence.csv"
                scheme_csv_paths.append(str(csv_path))
                write_rows_csv(csv_path, scheme_rows)

    write_json(
        layout["convergence"] / "convergence.json",
        {
            "rows_by_scheme": rows_by_scheme,
            "failures": failures,
            "skipped": skipped,
            "scheme_csv_paths": scheme_csv_paths,
        },
    )
    return {
        "rows_by_scheme": rows_by_scheme,
        "failures": failures,
        "skipped": skipped,
        "scheme_csv_paths": scheme_csv_paths,
    }


def phase_stability(layout: dict[str, Path], *, use_numba: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped = [{"phase": "stability", "ndivs": 8, "reason": "not run in this pass due runtime budget"}]
    plot_paths: list[str] = []

    runs_by_form_ndiv: dict[tuple[str, int], list[Step9RunData]] = {}

    for ndivs in (4,):
        for volume_form in FORMS:
            runs_by_form_ndiv[(volume_form, ndivs)] = []
            for sbp_variant in STUDY_VARIANTS:
                for flux in FLUXES:
                    case = Task5Case(
                        ndivs=ndivs,
                        sbp_variant=sbp_variant,
                        flux_type=flux,
                        volume_form=volume_form,
                        use_numba=use_numba,
                        history_every=1,
                        tf=20.0,
                        cfl=1.0,
                    )
                    try:
                        run = run_step9_case(case, capture_states=False)
                        rows.append(build_run_summary_row(run, phase="stability"))
                        runs_by_form_ndiv[(volume_form, ndivs)].append(run)
                    except Exception as exc:
                        failures.append({"phase": "stability", "scheme_id": case.scheme_id(), "error": str(exc)})

    for (volume_form, ndivs), runs in runs_by_form_ndiv.items():
        if not runs:
            continue
        output_path = layout["stability"] / f"energy_ratio_{volume_form}_ndiv{ndivs}.png"
        plot_energy_ratio_histories(
            runs,
            output_path=output_path,
            title=f"Energy ratio history / {volume_form} / ndiv {ndivs}",
        )
        plot_paths.append(str(output_path))

    csv_path = layout["stability"] / "stability.csv"
    write_rows_csv(csv_path, rows)
    write_json(layout["stability"] / "stability.json", {"rows": rows, "failures": failures, "skipped": skipped, "plot_paths": plot_paths})
    return {"rows": rows, "failures": failures, "skipped": skipped, "plot_paths": plot_paths, "csv": str(csv_path)}


def _rates_from_residual_rows(rows: list[dict[str, Any]]) -> list[ProductRuleResidualRow]:
    if not rows:
        return []

    h = [float(row["hmin"]) for row in rows]
    tau_r = [float(row["tau_r_l2"]) for row in rows]
    tau_s = [float(row["tau_s_l2"]) for row in rows]
    rate_r = estimate_convergence_rates(tau_r, h)
    rate_s = estimate_convergence_rates(tau_s, h)

    out: list[ProductRuleResidualRow] = []
    for row, rr, rs in zip(rows, rate_r, rate_s):
        out.append(
            ProductRuleResidualRow(
                sbp_variant=str(row["sbp_variant"]),
                ndivs=int(row["ndivs"]),
                elements=int(row["elements"]),
                hmin=float(row["hmin"]),
                tau_r_l2=float(row["tau_r_l2"]),
                tau_s_l2=float(row["tau_s_l2"]),
                rate_r=None if rr is None else float(rr),
                rate_s=None if rs is None else float(rs),
            )
        )
    return out


def phase_product_rule(layout: dict[str, Path], *, include_ndivs64: bool) -> dict[str, Any]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    ndivs_values = [4, 8, 16, 32]

    if include_ndivs64:
        ndivs_values.append(64)
    else:
        skipped.append({"phase": "product-rule", "variant": "all", "reason": "ndivs=64 not requested"})

    for sbp_variant in ("projected", "full-orth"):
        raw_rows: list[dict[str, Any]] = []
        for ndivs in ndivs_values:
            case = Task5Case(ndivs=ndivs, sbp_variant=sbp_variant, tf=1.0, cfl=1.0, use_numba=False)
            try:
                raw_rows.append(compute_product_rule_residual(case))
            except Exception as exc:
                failures.append({"phase": "product-rule", "variant": sbp_variant, "ndivs": ndivs, "error": str(exc)})
                break

        rows_by_variant[sbp_variant] = [asdict(row) for row in _rates_from_residual_rows(raw_rows)]
        if rows_by_variant[sbp_variant]:
            write_rows_csv(
                layout["product-rule"] / f"{sbp_variant}_product_rule.csv",
                rows_by_variant[sbp_variant],
            )

    try:
        raw_check = compute_product_rule_residual(Task5Case(ndivs=4, sbp_variant="full-raw", use_numba=False))
        orth_check = compute_product_rule_residual(Task5Case(ndivs=4, sbp_variant="full-orth", use_numba=False))
        raw_orth_check = {
            "ndivs": 4,
            "tau_r_l2_abs_diff": abs(raw_check["tau_r_l2"] - orth_check["tau_r_l2"]),
            "tau_s_l2_abs_diff": abs(raw_check["tau_s_l2"] - orth_check["tau_s_l2"]),
            "tau_r_max_abs_diff": abs(raw_check["tau_r_max_abs"] - orth_check["tau_r_max_abs"]),
            "tau_s_max_abs_diff": abs(raw_check["tau_s_max_abs"] - orth_check["tau_s_max_abs"]),
        }
    except Exception as exc:
        raw_orth_check = {"error": str(exc)}
        failures.append({"phase": "product-rule", "variant": "full-raw-vs-full-orth", "error": str(exc)})

    write_json(
        layout["product-rule"] / "product_rule.json",
        {
            "rows_by_variant": rows_by_variant,
            "raw_orth_check": raw_orth_check,
            "failures": failures,
            "skipped": skipped,
        },
    )
    return {
        "rows_by_variant": rows_by_variant,
        "raw_orth_check": raw_orth_check,
        "failures": failures,
        "skipped": skipped,
    }


def generate_report(output_dir: Path, results: dict[str, Any], environment: dict[str, Any]) -> Path:
    report_path = output_dir / "report.md"
    lines = [
        "# Task 5 full-SBP validation",
        "",
        "## Environment",
        f"- Python: `{environment['python_version'].splitlines()[0]}`",
        f"- NumPy: `{environment['numpy_version']}`",
        f"- Numba used: `{environment['numba_used']}`",
        f"- Platform: `{environment['platform']}`",
        f"- Git commit: `{environment['git_commit']}`",
        "",
        "## Phase files",
    ]

    for phase, payload in results.items():
        lines.append(f"- `{phase}`: `{json.dumps(payload, default=str)[:200]}`")

    lines.extend(
        [
            "",
            "## Constant-state scope",
            "The nonzero constant-state RHS was not investigated or modified in this task, as requested.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_selected_phases(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    layout = ensure_output_layout(output_dir)
    previous_summary = load_existing_summary(output_dir / "summary.json")
    merged_results: dict[str, Any] = dict(previous_summary.get("results", {}))
    warm_numba_if_requested(bool(args.use_numba))
    results: dict[str, Any] = {}

    selected = list(PHASES[:-1]) if args.phase == "all" else [args.phase]

    for phase in selected:
        start = time.perf_counter()
        print(f"[task5] phase={phase} output_dir={output_dir} use_numba={bool(args.use_numba)}")
        if phase == "equivalence":
            payload = phase_equivalence(layout, use_numba=bool(args.use_numba))
        elif phase == "mass":
            payload = phase_mass(layout, use_numba=bool(args.use_numba))
        elif phase == "convergence":
            payload = phase_convergence(layout, use_numba=bool(args.use_numba), include_ndivs64=bool(args.include_ndivs64))
        elif phase == "stability":
            payload = phase_stability(layout, use_numba=bool(args.use_numba))
        elif phase == "product-rule":
            payload = phase_product_rule(layout, include_ndivs64=bool(args.include_ndivs64))
        else:
            raise ValueError(f"Unsupported phase: {phase}")

        payload["elapsed_seconds"] = time.perf_counter() - start
        results[phase] = payload
        merged_results[phase] = payload

    environment = environment_summary(use_numba=bool(args.use_numba))
    summary_rows: list[dict[str, Any]] = []

    for phase, payload in merged_results.items():
        if phase == "convergence":
            for scheme_rows in payload.get("rows_by_scheme", {}).values():
                summary_rows.extend({"phase": phase, **row} for row in scheme_rows)
        elif phase == "product-rule":
            for variant_rows in payload.get("rows_by_variant", {}).values():
                summary_rows.extend({"phase": phase, **row} for row in variant_rows)
        else:
            summary_rows.extend({"phase": phase, **row} for row in payload.get("rows", []))

    write_rows_csv(output_dir / "summary.csv", summary_rows)
    write_json(output_dir / "summary.json", {"environment": environment, "results": merged_results})
    report_path = generate_report(output_dir, merged_results, environment)
    merged_results["report_path"] = str(report_path)
    return {"environment": environment, "results": merged_results, "output_dir": str(output_dir)}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_selected_phases(args)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
