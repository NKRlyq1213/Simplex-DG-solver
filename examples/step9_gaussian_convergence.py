from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import operator
from pathlib import Path
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplex_dg.diagnostics import (
    ConvergenceRow,
    error_report,
    format_convergence_table,
    rows_to_dicts_with_rates,
    write_convergence_csv,
)
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.problems import exact_gaussian_solid_body, gaussian_on_sphere
from simplex_dg.reference import SBPVariant, build_reference_cache, is_full_sbp_variant, normalize_sbp_variant
from simplex_dg.rhs import build_full_rhs_cache, full_rhs
from simplex_dg.time import (
    cfl_dt_from_geometry,
    integrate_lsrk54,
    manifold_integral,
    manifold_l2_norm,
    minimum_face_length,
)
from simplex_dg.trace import build_trace_cache



_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_PLOT_TIME_UNIT_SECONDS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
}

_PLOT_TIME_UNIT_LABELS = {
    "second": "time (seconds)",
    "minute": "time (minutes)",
    "hour": "time (hours)",
    "day": "time (days)",
}


@dataclass(frozen=True)
class PlotTimeAxis:
    unit: str
    scale: float
    xlabel: str


@dataclass(frozen=True)
class RunOneNdivResult:
    row: ConvergenceRow
    history: list[dict[str, float]]
    q0: np.ndarray
    q_final: np.ndarray
    q_exact: np.ndarray


_SBP_VARIANT_CHOICES = ("projected", "full-raw", "full-orth")


def parse_float_expr(value: str) -> float:
    """Parse CLI float expressions such as 1.0, -0.5, pi/4, -pi/4."""
    if isinstance(value, (float, int)):
        return float(value)

    s = str(value).strip()

    try:
        return float(s)
    except ValueError:
        pass

    node = ast.parse(s, mode="eval").body

    def eval_node(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)

        if isinstance(n, ast.Name) and n.id == "pi":
            return float(np.pi)

        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(n.op)](eval_node(n.operand))

        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(n.op)](eval_node(n.left), eval_node(n.right))

        raise ValueError(f"Unsupported numeric expression: {value!r}")

    return float(eval_node(node))


def rotation_axis_from_alpha0(alpha0: float) -> tuple[float, float, float]:
    """Rotation axis convention.

    alpha0 = 0 gives axis (0, 0, 1), so the initial center (R, 0, 0)
    rotates along the equator.

    alpha0 tilts the axis inside the y-z plane:
        axis = (0, sin(alpha0), cos(alpha0)).
    """
    return (
        -float(np.sin(alpha0)),
        0.0,
        float(np.cos(alpha0)),
    )


def resolve_sigma_physical(
    *,
    radius: float,
    sigma_angle: float,
    sigma_physical: float | None = None,
) -> float:
    radius = float(radius)
    sigma_angle = float(sigma_angle)

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if sigma_physical is not None:
        sigma = float(sigma_physical)
    else:
        sigma = radius * sigma_angle

    if sigma <= 0.0:
        raise ValueError("sigma_physical must be positive.")

    return sigma


def validate_ndivs(ndivs: list[int]) -> list[int]:
    if not ndivs:
        raise ValueError("ndivs must not be empty.")

    out = [int(ndiv) for ndiv in ndivs]

    if any(ndiv < 1 for ndiv in out):
        raise ValueError("ndivs must contain only positive integers.")

    if len(set(out)) != len(out):
        raise ValueError("ndivs must be unique.")

    if any(curr <= prev for prev, curr in zip(out, out[1:])):
        raise ValueError("ndivs must be strictly increasing.")

    return out


def _normalize_expr_option_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    expr_options = {
        "--cfl",
        "--tf",
        "--sigma",
        "--sigma-physical",
        "--amplitude",
        "--height",
        "--radius",
        "--R",
        "--alpha0",
        "--u0",
        "--lf-alpha",
        "--lf-lambda",
    }
    i = 0

    while i < len(argv):
        token = argv[i]

        if token in expr_options and i + 1 < len(argv):
            normalized.append(f"{token}={argv[i + 1]}")
            i += 2
            continue

        normalized.append(token)
        i += 1

    return normalized


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    commit = result.stdout.strip()
    return commit or "unknown"


def metadata_path_from_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_metadata.json")


def validate_step9_configuration(*, table: str, sbp_variant: str) -> SBPVariant:
    sbp_variant_norm = normalize_sbp_variant(sbp_variant)

    if is_full_sbp_variant(sbp_variant_norm) and table != "table1":
        raise ValueError(
            "full-raw and full-orth require --table table1 because they use direct "
            "extraction of Table 1 boundary volume nodes."
        )

    return sbp_variant_norm


def scheme_identifier(
    *,
    table: str,
    sbp_variant: str,
    volume_form: str,
    flux_type: str,
) -> str:
    sbp_variant_norm = normalize_sbp_variant(sbp_variant)
    return f"{table}_{sbp_variant_norm}_{volume_form}_{flux_type}"


def scheme_label(
    *,
    table: str,
    sbp_variant: str,
    volume_form: str,
    flux_type: str,
) -> str:
    sbp_variant_norm = normalize_sbp_variant(sbp_variant)
    return f"{table} / {sbp_variant_norm} / {volume_form} / {flux_type}"


def output_path_for_scheme(
    output_path: str | Path,
    *,
    table: str,
    sbp_variant: str,
    volume_form: str,
    flux_type: str,
) -> Path:
    output = Path(output_path)
    scheme_id = scheme_identifier(
        table=table,
        sbp_variant=sbp_variant,
        volume_form=volume_form,
        flux_type=flux_type,
    )
    stem = output.stem

    if stem == scheme_id or stem.endswith(f"_{scheme_id}"):
        return output

    return output.with_name(f"{stem}_{scheme_id}{output.suffix}")


def plot_dir_for_scheme(
    plot_dir: str | Path,
    *,
    table: str,
    sbp_variant: str,
    volume_form: str,
    flux_type: str,
) -> Path:
    output_dir = Path(plot_dir)
    scheme_id = scheme_identifier(
        table=table,
        sbp_variant=sbp_variant,
        volume_form=volume_form,
        flux_type=flux_type,
    )

    if output_dir.name == scheme_id:
        return output_dir

    return output_dir / scheme_id


def titled_scheme_plot(base_title: str, *, scheme_text: str) -> str:
    return f"{base_title} [{scheme_text}]"


def build_run_metadata(
    *,
    args: argparse.Namespace,
    sigma_physical: float,
    output_csv: Path,
    plot_dir: Path | None,
) -> dict[str, object]:
    metadata = {
        "table": args.table,
        "sbp_variant": args.sbp,
        "scheme_id": scheme_identifier(
            table=args.table,
            sbp_variant=args.sbp,
            volume_form=args.form,
            flux_type=args.flux,
        ),
        "order": args.order,
        "ndivs": args.ndivs,
        "cfl_requested": float(args.cfl),
        "tf": float(args.tf),
        "sigma_angle": float(args.sigma),
        "sigma_physical": float(sigma_physical),
        "amplitude": float(args.amplitude),
        "radius": float(args.radius),
        "alpha0": float(args.alpha0),
        "u0": float(args.u0),
        "omega": list(rotation_axis_from_alpha0(args.alpha0)),
        "flux": args.flux,
        "lf_alpha": float(args.lf_alpha),
        "form": args.form,
        "use_numba": bool(not args.no_numba),
        "history_every": int(args.history_every),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "output_csv": str(output_csv),
        "plot_dir": None if plot_dir is None else str(plot_dir),
    }
    metadata["omega"] = [float(args.u0) * float(component) for component in metadata["omega"]]
    return metadata


def write_metadata_json(path: str | Path, metadata: dict[str, object]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    return output_path


def run_one_ndiv(
    *,
    ndivs: int,
    order: int,
    table: str,
    sbp_variant: SBPVariant | str = "projected",
    cfl: float,
    tf: float,
    sigma: float,
    radius: float,
    amplitude: float,
    alpha0: float,
    u0: float,
    lf_alpha: float,
    flux_type: str,
    volume_form: str,
    use_numba: bool,
    history_every: int,
    monitor_hook: Callable[[float, np.ndarray, dict[str, float]], None] | None = None,
) -> RunOneNdivResult:
    start = time.perf_counter()
    sbp_variant_norm = validate_step9_configuration(table=table, sbp_variant=sbp_variant)
    axis = rotation_axis_from_alpha0(alpha0)
    omega = tuple(float(u0) * a for a in axis)
    center0 = (radius, 0.0, 0.0)

    ref = build_reference_cache(order=order, table=table, sbp_variant=sbp_variant_norm)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=radius)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=omega,
        flux_type=flux_type,
        lf_alpha=lf_alpha,
        volume_form=volume_form,
    )

    q0 = gaussian_on_sphere(
        X=geom.X,
        center=center0,
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
    )

    # Initial invariants / ?????
    mass0_ref = manifold_integral(q0, ref, geom)
    l2_norm0_ref = manifold_l2_norm(q0, ref, geom)
    energy0_ref = 0.5 * l2_norm0_ref * l2_norm0_ref

    # Denominators for relative errors / ??????
    mass0_denom = max(abs(mass0_ref), np.finfo(float).tiny)
    energy0_denom = max(abs(energy0_ref), np.finfo(float).tiny)

    dt_raw = cfl_dt_from_geometry(
        ref=ref,
        geom=geom,
        max_speed=full.volume.max_speed,
        cfl=cfl,
    )

    nsteps = max(1, int(np.ceil(tf / dt_raw)))
    dt = float(tf / nsteps)

    def rhs(t, q):
        return full_rhs(q, full, use_numba=use_numba)

    def monitor(t, q):
        q_exact_t = exact_gaussian_solid_body(
            X=geom.X,
            t=t,
            radius=radius,
            sigma=sigma,
            amplitude=amplitude,
            center0=center0,
            omega=omega,
        )

        rep = error_report(q, q_exact_t, ref, geom)

        mass_t = manifold_integral(q, ref, geom)
        l2_norm_t = manifold_l2_norm(q, ref, geom)
        energy_t = 0.5 * l2_norm_t * l2_norm_t

        # Absolute relative errors / ??????
        relative_mass_error = abs(mass_t - mass0_ref) / mass0_denom
        relative_energy_error = abs(energy_t - energy0_ref) / energy0_denom

        # Signed relative errors / ??????
        signed_relative_mass_error = (mass_t - mass0_ref) / mass0_denom
        signed_relative_energy_error = (energy_t - energy0_ref) / energy0_denom
        q_min = float(np.min(q))
        q_max = float(np.max(q))

        entry = {
            "ndivs": float(ndivs),
            "t": float(t),
            "l2_error": rep.l2_error,
            "relative_l2_error": rep.relative_l2_error,
            "linf_error": rep.linf_error,
            "mass": mass_t,
            "l2_norm": l2_norm_t,
            "energy": energy_t,
            "relative_mass_error": relative_mass_error,
            "relative_energy_error": relative_energy_error,
            "signed_relative_mass_error": signed_relative_mass_error,
            "signed_relative_energy_error": signed_relative_energy_error,
            "q_min": q_min,
            "q_max": q_max,
            "undershoot": min(0.0, q_min),
            "overshoot": max(0.0, q_max - float(amplitude)),
        }

        if monitor_hook is not None:
            monitor_hook(float(t), np.asarray(q, dtype=float), entry)

        return entry

    result = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=0.0,
        tf=tf,
        dt=dt,
        monitor=monitor,
        monitor_every=max(1, history_every),
    )

    q_exact = exact_gaussian_solid_body(
        X=geom.X,
        t=tf,
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
        center0=center0,
        omega=omega,
    )

    rep = error_report(result.q, q_exact, ref, geom)

    mass0 = manifold_integral(q0, ref, geom)
    massf = manifold_integral(result.q, ref, geom)

    l20 = manifold_l2_norm(q0, ref, geom)
    l2f = manifold_l2_norm(result.q, ref, geom)
    energy0 = 0.5 * l20 * l20
    energyf = 0.5 * l2f * l2f
    mass0_denom = max(abs(mass0), np.finfo(float).tiny)
    energy0_denom = max(abs(energy0), np.finfo(float).tiny)
    q_min = float(np.min(result.q))
    q_max = float(np.max(result.q))
    elapsed_seconds = time.perf_counter() - start

    relative_mass_drift = (massf - mass0) / mass0_denom
    relative_energy_drift = (energyf - energy0) / energy0_denom

    row = ConvergenceRow(
        ndivs=ndivs,
        order=order,
        n_elements=mesh.elements.shape[0],
        n_points_per_element=ref.rs.shape[0],
        total_dofs=mesh.elements.shape[0] * ref.rs.shape[0],
        dt=dt,
        tf=tf,
        nsteps=result.nsteps,
        hmin=minimum_face_length(ref, geom),
        l2_error=rep.l2_error,
        relative_l2_error=rep.relative_l2_error,
        linf_error=rep.linf_error,
        mass_drift=relative_mass_drift,
        l2_norm_drift=l2f - l20,
        initial_mass=mass0,
        final_mass=massf,
        absolute_mass_drift=abs(massf - mass0),
        relative_mass_drift=relative_mass_drift,
        initial_energy=energy0,
        final_energy=energyf,
        absolute_energy_drift=abs(energyf - energy0),
        relative_energy_drift=relative_energy_drift,
        q_min=q_min,
        q_max=q_max,
        undershoot=min(0.0, q_min),
        overshoot=max(0.0, q_max - float(amplitude)),
        elapsed_seconds=elapsed_seconds,
    )

    return RunOneNdivResult(
        row=row,
        history=result.history,
        q0=q0,
        q_final=result.q,
        q_exact=q_exact,
    )


def history_time_span(histories: list[list[dict[str, float]]]) -> float:
    tmin: float | None = None
    tmax: float | None = None

    for hist in histories:
        if not hist:
            continue

        times = [float(entry["t"]) for entry in hist]

        if not times:
            continue

        hist_min = min(times)
        hist_max = max(times)

        tmin = hist_min if tmin is None else min(tmin, hist_min)
        tmax = hist_max if tmax is None else max(tmax, hist_max)

    if tmin is None or tmax is None:
        return 0.0

    return max(0.0, tmax - tmin)


def resolve_plot_time_axis(
    *,
    span_seconds: float,
    plot_time_unit: str = "auto",
) -> PlotTimeAxis:
    if plot_time_unit == "auto":
        if span_seconds > 86400.0:
            unit = "day"
        elif span_seconds > 7200.0:
            unit = "hour"
        elif span_seconds > 600.0:
            unit = "minute"
        else:
            unit = "second"
    else:
        unit = plot_time_unit

    return PlotTimeAxis(
        unit=unit,
        scale=_PLOT_TIME_UNIT_SECONDS[unit],
        xlabel=_PLOT_TIME_UNIT_LABELS[unit],
    )


def resolve_history_time_axis(
    histories: list[list[dict[str, float]]],
    *,
    plot_time_unit: str = "auto",
) -> PlotTimeAxis:
    span_seconds = history_time_span(histories)
    return resolve_plot_time_axis(span_seconds=span_seconds, plot_time_unit=plot_time_unit)



def plot_time_history_quantity(
    histories: dict[int, list[dict[str, float]]],
    output_path: Path,
    *,
    quantity: str,
    ylabel: str,
    title: str,
    semilogy: bool = True,
    plot_time_unit: str = "auto",
) -> None:
    """Plot a monitored scalar quantity over time.

    ?????
        ??? relative mass error?relative energy error?
        signed relative energy error ???????
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = resolve_history_time_axis(list(histories.values()), plot_time_unit=plot_time_unit)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for ndivs, hist in sorted(histories.items()):
        t = np.array([entry["t"] for entry in hist], dtype=float) / time_axis.scale
        y = np.array([entry[quantity] for entry in hist], dtype=float)

        if semilogy:
            # Avoid log(0). Exact zero is shown near machine tiny.
            # ?? log(0)???? 0 ??????? machine tiny ???
            y_plot = np.maximum(y, np.finfo(float).tiny)
            ax.semilogy(t, y_plot, linewidth=1.5, label=f"ndiv {ndivs}")
        else:
            ax.plot(t, y, linewidth=1.5, label=f"ndiv {ndivs}")

    ax.set_xlabel(time_axis.xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if semilogy:
        # Fixed y-axis range for relative error plots.
        # ??????? y ????10^{-18} ? 10^{1}
        ax.set_ylim(1.0e-18, 1.0e1)
    else:
        ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_history_quantity_each_level(
    histories: dict[int, list[dict[str, float]]],
    output_dir: Path,
    *,
    filename_prefix: str,
    quantity: str,
    ylabel: str,
    title_prefix: str,
    semilogy: bool = True,
    plot_time_unit: str = "auto",
) -> list[Path]:
    """Plot one time-history figure per subdivision count.

    ?????
        ??? energy error history ?? ndiv ??????
        ??????????????
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    for ndivs, hist in sorted(histories.items()):
        output_path = output_dir / f"{filename_prefix}_ndiv{ndivs}.png"
        time_axis = resolve_history_time_axis([hist], plot_time_unit=plot_time_unit)

        t = np.array([entry["t"] for entry in hist], dtype=float) / time_axis.scale
        y = np.array([entry[quantity] for entry in hist], dtype=float)

        fig, ax = plt.subplots(figsize=(8.0, 5.0))

        if semilogy:
            y_plot = np.maximum(y, np.finfo(float).tiny)
            ax.semilogy(t, y_plot, linewidth=1.5, label=f"ndiv {ndivs}")
            ax.set_ylim(1.0e-18, 1.0e1)
        else:
            ax.plot(t, y, linewidth=1.5, label=f"ndiv {ndivs}")
            ax.axhline(0.0, linestyle="--", linewidth=1.0)

        ax.set_xlabel(time_axis.xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}, ndiv {ndivs}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        output_paths.append(output_path)

    return output_paths

def plot_error_convergence(
    rows: list[ConvergenceRow],
    output_path: Path,
    *,
    title: str = "Gaussian advection convergence",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h = np.array([row.hmin for row in rows], dtype=float)
    l2 = np.array([row.l2_error for row in rows], dtype=float)
    rel = np.array([row.relative_l2_error for row in rows], dtype=float)
    linf = np.array([row.linf_error for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.loglog(h, l2, marker="o", linewidth=1.5, label="L2 error")
    #ax.loglog(h, rel, marker="s", linewidth=1.5, label="relative L2 error")
    ax.loglog(h, linf, marker="^", linewidth=1.5, label="Linf error")

    if len(h) >= 2:
        p = rows[0].order
        guide = l2[-1] * (h / h[-1]) ** p
        ax.loglog(h, guide, linestyle="--", linewidth=1.2, label=f"O(h^{p}) guide")

    ax.set_xlabel("hmin")
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_observed_order(
    rows: list[ConvergenceRow],
    output_path: Path,
    *,
    title: str = "Observed convergence order",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_dicts = rows_to_dicts_with_rates(rows)
    ndivs = np.array([int(row["ndivs"]) for row in row_dicts], dtype=int)
    l2_rates = [None if row["l2_rate"] == "" else float(row["l2_rate"]) for row in row_dicts]
    linf_rates = [None if row["linf_rate"] == "" else float(row["linf_rate"]) for row in row_dicts]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if len(ndivs) >= 2:
        x = ndivs[1:]
        ax.plot(x, [r for r in l2_rates[1:]], marker="o", linewidth=1.5, label="L2 observed order")
        ax.plot(x, [r for r in linf_rates[1:]], marker="^", linewidth=1.5, label="Linf observed order")

        ax.axhline(rows[0].order, linestyle="--", linewidth=1.2, label=f"target order {rows[0].order}")

    ax.set_xlabel("ndivs")
    ax.set_ylabel("observed order")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_solution_bounds_history(
    histories: dict[int, list[dict[str, float]]],
    output_path: Path,
    *,
    amplitude: float,
    title: str = "Gaussian solution bounds history",
    plot_time_unit: str = "auto",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = resolve_history_time_axis(list(histories.values()), plot_time_unit=plot_time_unit)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for ndivs, hist in sorted(histories.items()):
        t = np.array([entry["t"] for entry in hist], dtype=float) / time_axis.scale
        q_min = np.array([entry["q_min"] for entry in hist], dtype=float)
        q_max = np.array([entry["q_max"] for entry in hist], dtype=float)
        ax.plot(t, q_min, linewidth=1.2, label=f"q_min ndiv {ndivs}")
        ax.plot(t, q_max, linewidth=1.2, linestyle="--", label=f"q_max ndiv {ndivs}")

    ax.axhline(0.0, linestyle=":", linewidth=1.0, color="black")
    ax.axhline(float(amplitude), linestyle=":", linewidth=1.0, color="gray")
    ax.set_xlabel(time_axis.xlabel)
    ax.set_ylabel("solution bounds")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gaussian solid-body advection convergence runner.")

    run_group = parser.add_argument_group("mesh/run control")
    run_group.add_argument("--ndivs", nargs="+", type=int, default=[1, 2, 4, 8])
    run_group.add_argument("--order", type=int, default=4)
    run_group.add_argument("--table", type=str, default="table1", choices=["table1", "table2"])
    run_group.add_argument("--cfl", type=float, default=1.0)
    run_group.add_argument("--tf", type=float, default=1.0)
    run_group.add_argument("--history-every", type=int, default=1)

    gaussian_group = parser.add_argument_group("Gaussian/physics parameters")
    gaussian_group.add_argument(
        "--sigma",
        type=parse_float_expr,
        default=0.35,
        help="Gaussian angular width in radians. Physical width is R*sigma unless --sigma-physical is provided.",
    )
    gaussian_group.add_argument(
        "--sigma-physical",
        type=parse_float_expr,
        default=None,
        help="Override Gaussian physical arc-length width. If omitted, sigma_physical = R*sigma.",
    )
    gaussian_group.add_argument(
        "--amplitude",
        "--height",
        dest="amplitude",
        type=parse_float_expr,
        default=1.0,
        help="Gaussian peak height/amplitude.",
    )
    gaussian_group.add_argument("--radius", "--R", dest="radius", type=parse_float_expr, default=1.0)
    gaussian_group.add_argument(
        "--alpha0",
        type=parse_float_expr,
        default=-np.pi / 4.0,
        help="Rotation-axis tilt angle. alpha0=0 gives z-axis rotation, so center (R,0,0) moves on the equator.",
    )
    gaussian_group.add_argument(
        "--u0",
        type=parse_float_expr,
        default=1.0,
        help="Angular speed multiplier. omega = u0 * axis(alpha0).",
    )
    gaussian_group.add_argument(
        "--lf-alpha",
        "--lf-lambda",
        dest="lf_alpha",
        type=parse_float_expr,
        default=1.0,
        help="Lax-Friedrichs penalty multiplier: F*=0.5*a(q-+q+) - 0.5*lf_alpha*|a|*(q+-q-).",
    )

    numerics_group = parser.add_argument_group("flux/form/backend")
    numerics_group.add_argument("--flux", type=str, default="upwind", choices=["upwind", "central", "lf"])
    numerics_group.add_argument(
        "--form",
        type=str,
        default="conservative",
        choices=["conservative", "split"],
        help="Volume/surface pairing: conservative or split.",
    )
    numerics_group.add_argument(
        "--sbp",
        type=str,
        default="projected",
        choices=_SBP_VARIANT_CHOICES,
        help=(
            "SBP variant: projected = existing projected-SBP differentiation, projected trace, "
            "and polynomial lift; full-raw = Table 1 raw-basis full-SBP with direct extraction "
            "and H^{-1}E^T W_b lift; full-orth = algebraically equivalent orthogonalized full-SBP construction."
        ),
    )
    numerics_group.add_argument("--no-numba", action="store_true")

    output_group = parser.add_argument_group("outputs/plots")
    output_group.add_argument("--output", type=str, default="outputs/convergence/gaussian_sphere_convergence.csv")
    output_group.add_argument("--plot-dir", type=str, default="outputs/convergence/plots")
    output_group.add_argument("--no-plots", action="store_true")
    output_group.add_argument(
        "--plot-time-unit",
        type=str,
        default="auto",
        choices=["auto", "second", "minute", "hour", "day"],
        help="Time unit for time-history plots. Default auto-selects from the plotted time span.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(_normalize_expr_option_args(list(argv)))

    try:
        args.ndivs = validate_ndivs(args.ndivs)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        args.sbp = validate_step9_configuration(table=args.table, sbp_variant=args.sbp)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    rows: list[ConvergenceRow] = []
    histories: dict[int, list[dict[str, float]]] = {}

    sigma_physical = resolve_sigma_physical(
        radius=args.radius,
        sigma_angle=args.sigma,
        sigma_physical=args.sigma_physical,
    )

    print("Gaussian convergence run")
    print("------------------------")
    print(f"ndivs         : {args.ndivs}")
    print(f"order         : {args.order}")
    print(f"table         : {args.table}")
    print(f"sbp           : {args.sbp}")
    print(f"cfl           : {args.cfl}")
    print(f"tf            : {args.tf}")
    print(f"sigma_angle   : {args.sigma}")
    print(f"sigma_physical: {sigma_physical}")
    print(f"amplitude     : {args.amplitude}")
    print(f"radius        : {args.radius}")
    print(f"alpha0        : {args.alpha0}")
    print(f"u0            : {args.u0}")
    print(f"lf_alpha      : {args.lf_alpha}")
    print(f"flux          : {args.flux}")
    print(f"form          : {args.form}")
    print(f"history_every : {args.history_every}")
    print(f"plot_time_unit: {args.plot_time_unit}")
    print()

    t0 = time.perf_counter()
    scheme_text = scheme_label(
        table=args.table,
        sbp_variant=args.sbp,
        volume_form=args.form,
        flux_type=args.flux,
    )

    for ndivs in args.ndivs:
        start = time.perf_counter()

        result = run_one_ndiv(
            ndivs=ndivs,
            order=args.order,
            table=args.table,
            sbp_variant=args.sbp,
            cfl=args.cfl,
            tf=args.tf,
            sigma=sigma_physical,
            radius=args.radius,
            amplitude=args.amplitude,
            alpha0=args.alpha0,
            u0=args.u0,
            lf_alpha=args.lf_alpha,
            flux_type=args.flux,
            volume_form=args.form,
            use_numba=not args.no_numba,
            history_every=args.history_every,
        )

        row = result.row
        rows.append(row)
        histories[ndivs] = result.history

        elapsed = time.perf_counter() - start

        print(
            f"ndivs={ndivs}, K={row.n_elements}, DOFs={row.total_dofs}, "
            f"L2={row.l2_error:.6e}, rel={row.relative_l2_error:.6e}, "
            f"Linf={row.linf_error:.6e}, mass ref drift={row.relative_mass_drift:+.6e}, "
            f"time={elapsed:.2f}s"
        )

    print()
    print(format_convergence_table(rows))

    output = output_path_for_scheme(
        args.output,
        table=args.table,
        sbp_variant=args.sbp,
        volume_form=args.form,
        flux_type=args.flux,
    )
    write_convergence_csv(output, rows)
    resolved_plot_dir = None if args.no_plots else plot_dir_for_scheme(
        args.plot_dir,
        table=args.table,
        sbp_variant=args.sbp,
        volume_form=args.form,
        flux_type=args.flux,
    )
    metadata = build_run_metadata(
        args=args,
        sigma_physical=sigma_physical,
        output_csv=output,
        plot_dir=resolved_plot_dir,
    )
    metadata_output = metadata_path_from_output(output)
    write_metadata_json(metadata_output, metadata)

    print()
    print(f"CSV written to: {output}")
    print(f"Metadata written to: {metadata_output}")

    if not args.no_plots:
        plot_dir = resolved_plot_dir
        assert plot_dir is not None
        plot_dir.mkdir(parents=True, exist_ok=True)

        time_plot = plot_dir / "relative_l2_history.png"
        mass_plot = plot_dir / "relative_mass_history.png"
        energy_plot = plot_dir / "relative_energy_history.png"
        signed_energy_plot = plot_dir / "signed_energy_history.png"
        bounds_plot = plot_dir / "solution_bounds_history.png"
        conv_plot = plot_dir / "error_convergence.png"
        order_plot = plot_dir / "observed_order.png"

        plot_time_history_quantity(
            histories=histories,
            output_path=time_plot,
            quantity="relative_l2_error",
            ylabel="relative L2 error",
            title=titled_scheme_plot("Gaussian advection relative L2 error history", scheme_text=scheme_text),
            semilogy=True,
            plot_time_unit=args.plot_time_unit,
        )

        plot_time_history_quantity(
            histories=histories,
            output_path=mass_plot,
            quantity="relative_mass_error",
            ylabel="relative mass error",
            title=titled_scheme_plot("Relative mass error history", scheme_text=scheme_text),
            semilogy=True,
            plot_time_unit=args.plot_time_unit,
        )

        plot_time_history_quantity(
            histories=histories,
            output_path=energy_plot,
            quantity="relative_energy_error",
            ylabel="relative energy error",
            title=titled_scheme_plot("Relative energy error history", scheme_text=scheme_text),
            semilogy=True,
            plot_time_unit=args.plot_time_unit,
        )

        plot_time_history_quantity(
            histories=histories,
            output_path=signed_energy_plot,
            quantity="signed_relative_energy_error",
            ylabel="signed relative energy error",
            title=titled_scheme_plot("Signed relative energy error history", scheme_text=scheme_text),
            semilogy=False,
            plot_time_unit=args.plot_time_unit,
        )

        plot_solution_bounds_history(
            histories=histories,
            output_path=bounds_plot,
            amplitude=float(args.amplitude),
            title=titled_scheme_plot("Gaussian solution bounds history", scheme_text=scheme_text),
            plot_time_unit=args.plot_time_unit,
        )

        plot_error_convergence(
            rows=rows,
            output_path=conv_plot,
            title=titled_scheme_plot("Gaussian advection convergence", scheme_text=scheme_text),
        )

        plot_observed_order(
            rows=rows,
            output_path=order_plot,
            title=titled_scheme_plot("Observed convergence order", scheme_text=scheme_text),
        )

        print()
        print("Plots written to:")
        print(f"  {time_plot}")
        print(f"  {mass_plot}")
        print(f"  {energy_plot}")
        print(f"  {signed_energy_plot}")
        print(f"  {bounds_plot}")
        print(f"  {conv_plot}")
        print(f"  {order_plot}")

    elapsed_total = time.perf_counter() - t0

    print()
    print(f"total elapsed: {elapsed_total:.2f}s")


if __name__ == "__main__":
    main()
