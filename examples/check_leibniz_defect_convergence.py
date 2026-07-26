from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import operator
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from simplex_dg.diagnostics.leibniz import (
    LeibnizDefectRow,
    attach_rates,
    compute_leibniz_defect_row,
    format_physical_table,
    format_reference_table,
    observed_rate,
    write_csv,
)
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.problems import gaussian_on_sphere
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_volume_rhs_cache


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


@dataclass(frozen=True)
class RunOneNdivResult:
    row: LeibnizDefectRow
    q_raw: np.ndarray
    q_h: np.ndarray


def parse_float_expr(value: str | float | int) -> float:
    """Parse CLI expressions such as 1.0, pi/4, and -pi/4."""
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
    """Match the existing rotation-axis convention used by step9."""
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

    sigma = radius * sigma_angle if sigma_physical is None else float(sigma_physical)

    if sigma <= 0.0:
        raise ValueError("sigma_physical must be positive.")

    return sigma


def validate_ndivs(ndivs: list[int]) -> list[int]:
    if not ndivs:
        raise ValueError("ndivs must not be empty.")

    out = [int(ndiv) for ndiv in ndivs]

    if any(ndiv <= 0 for ndiv in out):
        raise ValueError("ndivs must contain only positive integers.")

    if len(set(out)) != len(out):
        raise ValueError("ndivs must be unique.")

    if any(curr <= prev for prev, curr in zip(out, out[1:])):
        raise ValueError("ndivs must be strictly increasing.")

    return out


def _normalize_expr_option_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    expr_options = {
        "--sigma",
        "--sigma-physical",
        "--amplitude",
        "--radius",
        "--alpha0",
        "--u0",
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


def write_metadata_json(path: str | Path, metadata: dict[str, object]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return output_path


def run_one_ndiv(
    *,
    ndivs: int,
    order: int,
    table: str,
    radius: float,
    sigma_physical: float,
    amplitude: float,
    alpha0: float,
    u0: float,
    state: str,
    project_velocity: bool,
) -> RunOneNdivResult:
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=radius)
    geom = build_geometry_cache(mesh, ref, validate=True)

    axis = rotation_axis_from_alpha0(alpha0)
    omega = tuple(float(u0) * component for component in axis)

    volume = build_volume_rhs_cache(
        ref=ref,
        geom=geom,
        omega=omega,
        project_velocity=project_velocity,
        validate=True,
    )

    q_raw = gaussian_on_sphere(
        X=geom.X,
        center=(radius, 0.0, 0.0),
        radius=radius,
        sigma=sigma_physical,
        amplitude=amplitude,
    )

    result = compute_leibniz_defect_row(
        order=order,
        table=table,
        state=state,
        ref=ref,
        geom=geom,
        volume=volume,
        q_raw=q_raw,
        ndivs=ndivs,
    )

    return RunOneNdivResult(
        row=result.row,
        q_raw=result.q_raw,
        q_h=result.q_h,
    )


def _series(rows: list[LeibnizDefectRow], field_name: str) -> np.ndarray:
    return np.array([float(getattr(row, field_name)) for row in rows], dtype=float)


def _rate_series(
    rows: list[LeibnizDefectRow],
    field_name: str,
    *,
    rate_basis: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows_with_rates = attach_rates(rows, rate_basis=rate_basis)
    x_vals: list[float] = []
    y_vals: list[float] = []

    for row in rows_with_rates[1:]:
        rate = getattr(row, field_name)
        if rate is None or not np.isfinite(rate):
            continue
        x_vals.append(float(row.hmin if rate_basis == "hmin" else row.ndivs))
        y_vals.append(float(rate))

    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


def plot_loglog_convergence(
    rows: list[LeibnizDefectRow],
    output_path: Path,
    *,
    series: list[tuple[str, str]],
    ylabel: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h = np.array([row.hmin for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for label, field_name in series:
        ax.loglog(h, _series(rows, field_name), marker="o", linewidth=1.5, label=label)

    ax.set_xlabel("hmin")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.invert_xaxis()
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_observed_orders(
    rows: list[LeibnizDefectRow],
    output_path: Path,
    *,
    rate_basis: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series = [
        ("tau_r L2(ref)", "tau_r_l2_ref_rate"),
        ("tau_s L2(ref)", "tau_s_l2_ref_rate"),
        ("tau_sum L2(ref)", "tau_sum_l2_ref_rate"),
        ("tau_sum/J L2", "physical_tau_sum_l2_rate"),
        ("tau_sum/J Linf", "physical_tau_sum_linf_rate"),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for label, field_name in series:
        x_vals, y_vals = _rate_series(rows, field_name, rate_basis=rate_basis)
        if x_vals.size == 0:
            continue
        ax.plot(x_vals, y_vals, marker="o", linewidth=1.5, label=label)

    ax.set_xlabel("hmin" if rate_basis == "hmin" else "ndivs")
    ax.set_ylabel("observed order")
    ax.set_title(f"Observed Leibniz-defect orders ({rate_basis})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    if rate_basis == "hmin":
        ax.invert_xaxis()
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_relative_energy_residual(rows: list[LeibnizDefectRow], output_path: Path) -> None:
    plot_loglog_convergence(
        rows,
        output_path,
        series=[("|epsilon_tau| / E", "relative_energy_residual")],
        ylabel="relative energy residual",
        title="Relative energy residual convergence",
    )


def _is_strictly_decreasing(values: list[float], *, rtol: float = 1e-12) -> bool:
    return all(curr < prev * (1.0 + rtol) for prev, curr in zip(values[:-1], values[1:]))


def _last_rates(
    rows: list[LeibnizDefectRow],
    field_name: str,
    count: int = 3,
    *,
    rate_basis: str,
) -> list[float]:
    rates = [
        float(getattr(row, field_name))
        for row in attach_rates(rows, rate_basis=rate_basis)
        if getattr(row, field_name) is not None
    ]
    return rates[-count:]


def _format_rates(values: list[float]) -> str:
    if not values:
        return "-"
    return ", ".join(f"{value:.2f}" for value in values)


def build_numerical_summary(
    rows: list[LeibnizDefectRow],
    *,
    rate_basis: str,
) -> list[str]:
    tau_sum_ref = [row.tau_sum_l2_ref for row in rows]
    tau_sum_phys = [row.physical_tau_sum_l2 for row in rows]
    tau_sum_phys_linf = [row.physical_tau_sum_linf for row in rows]
    rel_energy = [row.relative_energy_residual for row in rows]
    closure = [row.q_projection_closure_linf for row in rows]

    summary: list[str] = []

    if _is_strictly_decreasing(tau_sum_ref):
        summary.append("Reference-form tau_sum L2 decreases on every tested refinement.")
    else:
        summary.append("Reference-form tau_sum L2 is not strictly decreasing on every tested refinement.")

    if _is_strictly_decreasing(tau_sum_phys):
        summary.append("Physical tau_sum/J L2 decreases on every tested refinement.")
    else:
        summary.append("Physical tau_sum/J L2 is not strictly decreasing on every tested refinement.")

    summary.append(
        f"Finest observed orders using {rate_basis} "
        f"(tau_sum L2ref / tau_sum/J L2 / tau_sum/J Linf): "
        f"{_format_rates(_last_rates(rows, 'tau_sum_l2_ref_rate', 2, rate_basis=rate_basis))} / "
        f"{_format_rates(_last_rates(rows, 'physical_tau_sum_l2_rate', 2, rate_basis=rate_basis))} / "
        f"{_format_rates(_last_rates(rows, 'physical_tau_sum_linf_rate', 2, rate_basis=rate_basis))}."
    )

    summary.append(
        f"Projection closure stays near discrete-roundoff scale; max ||q_h - P_N q_h||_inf = {max(closure):.3e}."
    )

    if rel_energy and min(rel_energy) > 0.0 and max(rel_energy) / min(rel_energy) > 1.0e3:
        summary.append(
            "The integrated scalar residual is orders of magnitude smaller on the finer meshes, "
            "but it should still be interpreted separately from pointwise defect norms because cancellation is possible."
        )
    else:
        summary.append(
            "The integrated scalar residual should not be used as a substitute for the defect norms because cancellation is possible."
        )

    if len(rows) >= 3:
        finest_rates = _last_rates(rows, "tau_sum_l2_ref_rate", 2, rate_basis=rate_basis)
        if finest_rates and min(finest_rates) < 0.5:
            summary.append("The finest meshes do not yet show a clean asymptotic rate for tau_sum L2ref.")
        elif finest_rates:
            summary.append("The finest meshes show a more stable tau_sum L2ref rate than the coarsest meshes.")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check convergence of the discrete Leibniz/product-rule defect on the sphere mesh."
    )
    parser.add_argument("--ndivs", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--table", type=str, default="table1", choices=["table1", "table2"])
    parser.add_argument("--radius", type=parse_float_expr, default=1.0)
    parser.add_argument("--sigma", type=parse_float_expr, default=0.35)
    parser.add_argument("--sigma-physical", type=parse_float_expr, default=None)
    parser.add_argument("--amplitude", type=parse_float_expr, default=1.0)
    parser.add_argument("--alpha0", type=parse_float_expr, default=-np.pi / 4.0)
    parser.add_argument("--u0", type=parse_float_expr, default=1.0)
    parser.add_argument(
        "--state",
        type=str,
        default="projected-gaussian",
        choices=["projected-gaussian", "raw-gaussian"],
    )
    parser.add_argument("--output", type=str, default="outputs/diagnostics/leibniz_defect_convergence.csv")
    parser.add_argument("--rate-basis", type=str, default="hmin", choices=["hmin", "ndiv"])
    parser.add_argument("--plot-dir", type=str, default="outputs/diagnostics/leibniz_defect_plots")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-project-velocity", action="store_true")
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

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sigma_physical = resolve_sigma_physical(
        radius=args.radius,
        sigma_angle=args.sigma,
        sigma_physical=args.sigma_physical,
    )
    axis = rotation_axis_from_alpha0(args.alpha0)
    omega = [float(args.u0) * float(component) for component in axis]
    project_velocity = not args.no_project_velocity

    print("Leibniz-defect convergence run")
    print("------------------------------")
    print(f"ndivs            : {args.ndivs}")
    print(f"order            : {args.order}")
    print(f"table            : {args.table}")
    print(f"radius           : {args.radius}")
    print(f"sigma_angle      : {args.sigma}")
    print(f"sigma_physical   : {sigma_physical}")
    print(f"amplitude        : {args.amplitude}")
    print(f"alpha0           : {args.alpha0}")
    print(f"u0               : {args.u0}")
    print(f"omega            : {omega}")
    print(f"state            : {args.state}")
    print(f"project_velocity : {project_velocity}")
    print(f"rate_basis       : {args.rate_basis}")
    print()

    rows: list[LeibnizDefectRow] = []

    for ndivs in args.ndivs:
        result = run_one_ndiv(
            ndivs=ndivs,
            order=args.order,
            table=args.table,
            radius=args.radius,
            sigma_physical=sigma_physical,
            amplitude=args.amplitude,
            alpha0=args.alpha0,
            u0=args.u0,
            state=args.state,
            project_velocity=project_velocity,
        )
        rows.append(result.row)
        print(
            f"ndivs={ndivs}, K={result.row.n_elements}, DOFs={result.row.total_dofs}, "
            f"tau_sum_L2ref={result.row.tau_sum_l2_ref:.6e}, "
            f"tau_sum/J_L2={result.row.physical_tau_sum_l2:.6e}, "
            f"|epsilon_tau|/E={result.row.relative_energy_residual:.6e}"
        )

    print()
    print(format_reference_table(rows, rate_basis=args.rate_basis))
    print()
    print(format_physical_table(rows, rate_basis=args.rate_basis))
    print()
    print(f"max projection closure error: {max(row.q_projection_closure_linf for row in rows):.6e}")
    print()
    print("Numerical summary")
    print("-----------------")
    for line in build_numerical_summary(rows, rate_basis=args.rate_basis):
        print(f"- {line}")

    output_path = write_csv(args.output, rows, rate_basis=args.rate_basis)
    metadata_output = metadata_path_from_output(Path(args.output))
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "order": int(args.order),
        "table": str(args.table),
        "ndivs": list(args.ndivs),
        "radius": float(args.radius),
        "sigma_angle": float(args.sigma),
        "sigma_physical": float(sigma_physical),
        "amplitude": float(args.amplitude),
        "alpha0": float(args.alpha0),
        "u0": float(args.u0),
        "omega": omega,
        "state": str(args.state),
        "project_velocity": bool(project_velocity),
        "rate_basis": str(args.rate_basis),
        "definitions": {
            "alpha": "alpha = J * u^r",
            "beta": "beta = J * u^s",
            "tau_r": "D_r(alpha * q_h) - alpha * D_r q_h - q_h * D_r alpha",
            "tau_s": "D_s(beta * q_h) - beta * D_s q_h - q_h * D_s beta",
            "tau_sum": "tau_r + tau_s",
            "physical_defect": "tau / J",
            "energy_residual": "0.5 * sum_K |T_hat| * sum_i w_i q_h (tau_r + tau_s)",
        },
    }
    write_metadata_json(metadata_output, metadata)

    print()
    print(f"CSV written to: {output_path}")
    print(f"Metadata written to: {metadata_output}")

    if not args.no_plots:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_loglog_convergence(
            rows,
            plot_dir / "reference_l2_convergence.png",
            series=[
                (r"$||\tau_r||_{L^2(\hat\Omega_h)}$", "tau_r_l2_ref"),
                (r"$||\tau_s||_{L^2(\hat\Omega_h)}$", "tau_s_l2_ref"),
                (r"$||\tau_{sum}||_{L^2(\hat\Omega_h)}$", "tau_sum_l2_ref"),
            ],
            ylabel="reference L2 defect",
            title="Reference-form Leibniz-defect convergence",
        )

        plot_loglog_convergence(
            rows,
            plot_dir / "physical_l2_convergence.png",
            series=[
                (r"$||\tau_r/J||_{L^2(\mathbb{S}^2)}$", "physical_tau_r_l2"),
                (r"$||\tau_s/J||_{L^2(\mathbb{S}^2)}$", "physical_tau_s_l2"),
                (r"$||\tau_{sum}/J||_{L^2(\mathbb{S}^2)}$", "physical_tau_sum_l2"),
            ],
            ylabel="physical L2 defect",
            title="Physical Leibniz-defect convergence",
        )

        plot_loglog_convergence(
            rows,
            plot_dir / "physical_linf_convergence.png",
            series=[
                (r"$||\tau_r/J||_\infty$", "physical_tau_r_linf"),
                (r"$||\tau_s/J||_\infty$", "physical_tau_s_linf"),
                (r"$||\tau_{sum}/J||_\infty$", "physical_tau_sum_linf"),
            ],
            ylabel="physical Linf defect",
            title="Physical Linf Leibniz-defect convergence",
        )

        plot_observed_orders(rows, plot_dir / "observed_orders.png", rate_basis=args.rate_basis)
        plot_relative_energy_residual(rows, plot_dir / "relative_energy_residual.png")

        print("Plots written to:")
        print(f"  {plot_dir / 'reference_l2_convergence.png'}")
        print(f"  {plot_dir / 'physical_l2_convergence.png'}")
        print(f"  {plot_dir / 'physical_linf_convergence.png'}")
        print(f"  {plot_dir / 'observed_orders.png'}")
        print(f"  {plot_dir / 'relative_energy_residual.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
