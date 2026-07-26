from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import operator
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from simplex_dg.diagnostics import (
    ProjectedProductConvergenceRow,
    ProjectedProductResidualReport,
    attach_projected_product_rates,
    projected_product_residual_report,
    write_projected_product_csv,
)
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.problems import gaussian_on_sphere
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache
from simplex_dg.time import minimum_face_length
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


@dataclass(frozen=True)
class RunOneNdivResult:
    row: ProjectedProductConvergenceRow
    report: ProjectedProductResidualReport
    q: np.ndarray


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
    """Match the rotation-axis convention used by step9 and other diagnostics."""
    return (
        -float(np.sin(alpha0)),
        0.0,
        float(np.cos(alpha0)),
    )


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
        "--radius",
        "--sigma",
        "--amplitude",
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


def run_one_ndiv(
    *,
    ndivs: int,
    order: int,
    table: str,
    radius: float,
    sigma: float,
    amplitude: float,
    alpha0: float,
    u0: float,
) -> RunOneNdivResult:
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=radius)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)

    axis = rotation_axis_from_alpha0(alpha0)
    omega = tuple(float(u0) * component for component in axis)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=omega,
        flux_type="central",
        lf_alpha=0.0,
        volume_form="conservative",
        project_velocity=True,
        validate=True,
    )

    q = gaussian_on_sphere(
        X=geom.X,
        center=(radius, 0.0, 0.0),
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
    )

    report = projected_product_residual_report(
        q=q,
        volume=full.volume,
        surface=full.surface,
        trace=full.trace,
    )

    row = ProjectedProductConvergenceRow(
        ndivs=ndivs,
        order=order,
        n_elements=mesh.elements.shape[0],
        total_dofs=mesh.elements.shape[0] * ref.rs.shape[0],
        hmin=minimum_face_length(ref, geom),
        absolute_weighted_l2=report.absolute_weighted_l2,
        relative_weighted_l2=report.relative_weighted_l2,
        absolute_linf=report.absolute_linf,
        relative_linf=report.relative_linf,
        r_absolute_weighted_l2=report.r_absolute_weighted_l2,
        s_absolute_weighted_l2=report.s_absolute_weighted_l2,
        reference_flux_weighted_l2=report.reference_flux_weighted_l2,
    )

    return RunOneNdivResult(
        row=row,
        report=report,
        q=q,
    )


def format_convergence_table(
    rows: list[ProjectedProductConvergenceRow],
    *,
    rate_basis: str = "hmin",
) -> str:
    lines = [
        "ndivs  elements  hmin         rel_L2       rate    rel_Linf     rate",
        "--------------------------------------------------------------------",
    ]

    for row in attach_projected_product_rates(rows, rate_basis=rate_basis):
        l2_rate = "--" if row.relative_weighted_l2_rate is None else f"{row.relative_weighted_l2_rate:.3f}"
        linf_rate = "--" if row.relative_linf_rate is None else f"{row.relative_linf_rate:.3f}"
        lines.append(
            f"{row.ndivs:<6d} "
            f"{row.n_elements:<9d} "
            f"{row.hmin:<12.4e} "
            f"{row.relative_weighted_l2:<12.4e} "
            f"{l2_rate:>6s} "
            f"{row.relative_linf:<12.4e} "
            f"{linf_rate:>6s}"
        )

    return "\n".join(lines)


def plot_convergence(rows: list[ProjectedProductConvergenceRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h = np.array([row.hmin for row in rows], dtype=float)
    rel_l2 = np.array([row.relative_weighted_l2 for row in rows], dtype=float)
    rel_linf = np.array([row.relative_linf for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.loglog(h, rel_l2, marker="o", linewidth=1.5, label="relative weighted L2 residual")
    ax.loglog(h, rel_linf, marker="^", linewidth=1.5, label="relative Linf residual")
    ax.set_xlabel("hmin")
    ax.set_ylabel("relative residual")
    ax.set_title("Projected boundary-product compatibility convergence")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.invert_xaxis()
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _strictly_decreasing(values: list[float], *, rtol: float = 1.0e-12) -> bool:
    return all(curr < prev * (1.0 + rtol) for prev, curr in zip(values[:-1], values[1:]))


def _format_rates(values: list[float]) -> str:
    if not values:
        return "-"
    return ", ".join(f"{value:.2f}" for value in values)


def build_summary(
    rows: list[ProjectedProductConvergenceRow],
    *,
    rate_basis: str,
) -> list[str]:
    rows_with_rates = attach_projected_product_rates(rows, rate_basis=rate_basis)
    rel_l2 = [row.relative_weighted_l2 for row in rows]
    rel_linf = [row.relative_linf for row in rows]
    l2_rates = [float(row.relative_weighted_l2_rate) for row in rows_with_rates if row.relative_weighted_l2_rate is not None]
    linf_rates = [float(row.relative_linf_rate) for row in rows_with_rates if row.relative_linf_rate is not None]

    summary: list[str] = []

    if _strictly_decreasing(rel_l2):
        summary.append("Relative weighted L2 residual decreases on every tested refinement.")
    else:
        summary.append("Relative weighted L2 residual is not strictly decreasing on every tested refinement.")

    if _strictly_decreasing(rel_linf):
        summary.append("Relative Linf residual decreases on every tested refinement.")
    else:
        summary.append("Relative Linf residual is not strictly decreasing on every tested refinement.")

    summary.append(
        "Finest observed rates "
        f"using {rate_basis} "
        f"(relative L2 / relative Linf): {_format_rates(l2_rates[-3:])} / {_format_rates(linf_rates[-3:])}."
    )

    finest = rows[-1]
    component_scale = max(finest.r_absolute_weighted_l2, finest.s_absolute_weighted_l2, np.finfo(float).tiny)
    if finest.absolute_weighted_l2 < 0.5 * component_scale:
        summary.append("The combined residual is smaller than the dominant r/s component on the finest mesh, so cancellation is present.")
    else:
        summary.append("The combined residual is of the same order as the dominant r/s component on the finest mesh, so cancellation is limited.")

    summary.append(
        "Relative residuals are the main refinement metric here; absolute residuals alone mix compatibility error with the scale of the projected line flux."
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refinement diagnostic for projected boundary-product compatibility."
    )
    parser.add_argument("--ndivs", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--table", type=str, default="table1", choices=["table1", "table2"])
    parser.add_argument("--radius", type=parse_float_expr, default=1.0)
    parser.add_argument("--sigma", type=parse_float_expr, default=0.35)
    parser.add_argument("--amplitude", type=parse_float_expr, default=1.0)
    parser.add_argument("--alpha0", type=parse_float_expr, default=-np.pi / 4.0)
    parser.add_argument("--u0", type=parse_float_expr, default=1.0)
    parser.add_argument("--output", type=str, default="outputs/convergence/projected_product_compatibility.csv")
    parser.add_argument("--rate-basis", type=str, default="hmin", choices=["hmin", "ndiv"])
    parser.add_argument(
        "--plot",
        type=str,
        default="outputs/convergence/plots/projected_product_compatibility.png",
        help="Path for the log-log convergence plot.",
    )
    parser.add_argument("--no-plot", action="store_true")
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

    if args.sigma <= 0.0:
        parser.error("sigma must be positive.")

    if args.radius <= 0.0:
        parser.error("radius must be positive.")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Projected product compatibility run")
    print("-----------------------------------")
    print(f"ndivs     : {args.ndivs}")
    print(f"order     : {args.order}")
    print(f"table     : {args.table}")
    print(f"radius    : {args.radius}")
    print(f"sigma     : {args.sigma}")
    print(f"amplitude : {args.amplitude}")
    print(f"alpha0    : {args.alpha0}")
    print(f"u0        : {args.u0}")
    print("flux      : central")
    print("lf_alpha  : 0.0")
    print("form      : conservative")
    print(f"rate_basis: {args.rate_basis}")
    print()

    rows: list[ProjectedProductConvergenceRow] = []

    for ndivs in args.ndivs:
        result = run_one_ndiv(
            ndivs=ndivs,
            order=args.order,
            table=args.table,
            radius=args.radius,
            sigma=args.sigma,
            amplitude=args.amplitude,
            alpha0=args.alpha0,
            u0=args.u0,
        )
        rows.append(result.row)
        print(
            f"ndivs={ndivs}, K={result.row.n_elements}, DOFs={result.row.total_dofs}, "
            f"rel_L2={result.row.relative_weighted_l2:.6e}, rel_Linf={result.row.relative_linf:.6e}, "
            f"abs_r={result.row.r_absolute_weighted_l2:.6e}, abs_s={result.row.s_absolute_weighted_l2:.6e}"
        )

    print()
    print(format_convergence_table(rows, rate_basis=args.rate_basis))
    print()
    print("Summary")
    print("-------")
    for line in build_summary(rows, rate_basis=args.rate_basis):
        print(f"- {line}")

    output_path = write_projected_product_csv(args.output, rows, rate_basis=args.rate_basis)
    print()
    print(f"CSV written to: {output_path}")

    if not args.no_plot:
        plot_path = Path(args.plot)
        plot_convergence(rows, plot_path)
        print(f"Plot written to: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
