# examples/check_metric_divergence.py
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import operator
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs.volume import build_volume_rhs_cache
from simplex_dg.time import minimum_face_length


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


def parse_float_expr(value: str | float | int) -> float:
    """Parse float expressions such as 1.0, -0.5, pi/4, -pi/4.

    This is used so that CLI inputs can match the notation used in step9.
    """
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
    """Use the same rotation-axis convention as step9."""
    return (
        -float(np.sin(alpha0)),
        0.0,
        float(np.cos(alpha0)),
    )


@dataclass(frozen=True)
class MetricDivergenceRow:
    order: int
    table: str
    ndivs: int
    n_elements: int
    n_points_per_element: int
    hmin: float

    conservative_linf: float
    conservative_l2_ref: float
    conservative_weighted_mean: float

    physical_linf: float
    physical_l2: float
    physical_weighted_mean: float

    conservative_linf_rate: float | None = None
    conservative_l2_ref_rate: float | None = None
    physical_linf_rate: float | None = None
    physical_l2_rate: float | None = None


def weighted_l2_reference_quantity(
    values: np.ndarray,
    weights: np.ndarray,
    area: float,
) -> float:
    """Compute sqrt(sum_K |T| sum_i w_i values_{K,i}^2).

    This is a reference-element quadrature norm. It is useful for
    D_r alpha + D_s beta because this quantity appears before division by J.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if values.ndim != 2:
        raise ValueError("values must have shape (K, Np).")

    if values.shape[1] != weights.size:
        raise ValueError("weights size must match values.shape[1].")

    return float(np.sqrt(np.sum(area * weights[None, :] * values * values)))


def weighted_mean_reference_quantity(
    values: np.ndarray,
    weights: np.ndarray,
    area: float,
) -> float:
    """Compute |sum_K |T| sum_i w_i values_{K,i}|.

    This checks the integrated conservative defect. It can be small even if the
    pointwise defect is not small, so it should not replace the L-infinity norm.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    return float(abs(np.sum(area * weights[None, :] * values)))


def weighted_l2_physical_quantity(
    values: np.ndarray,
    sqrt_g: np.ndarray,
    weights: np.ndarray,
    area: float,
) -> float:
    """Compute sqrt(sum_K |T| sum_i w_i sqrt_g_{K,i} values_{K,i}^2).

    This is the physical surface L2 norm. It is useful for checking
    (D_r alpha + D_s beta) / J.
    """
    values = np.asarray(values, dtype=float)
    sqrt_g = np.asarray(sqrt_g, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if not (values.shape == sqrt_g.shape):
        raise ValueError("values and sqrt_g must have the same shape.")

    if values.shape[1] != weights.size:
        raise ValueError("weights size must match values.shape[1].")

    return float(np.sqrt(np.sum(area * weights[None, :] * sqrt_g * values * values)))


def weighted_mean_physical_quantity(
    values: np.ndarray,
    sqrt_g: np.ndarray,
    weights: np.ndarray,
    area: float,
) -> float:
    """Compute |sum_K |T| sum_i w_i sqrt_g_{K,i} values_{K,i}|."""
    values = np.asarray(values, dtype=float)
    sqrt_g = np.asarray(sqrt_g, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    return float(abs(np.sum(area * weights[None, :] * sqrt_g * values)))


def convergence_rate(prev_error: float, curr_error: float, prev_h: float, curr_h: float) -> float | None:
    """Compute rate using actual h ratios.

    rate = log(prev_error / curr_error) / log(prev_h / curr_h).
    """
    if prev_error <= 0.0 or curr_error <= 0.0:
        return None

    if prev_h <= 0.0 or curr_h <= 0.0 or prev_h == curr_h:
        return None

    return float(np.log(prev_error / curr_error) / np.log(prev_h / curr_h))


def compute_metric_divergence_row(
    *,
    order: int,
    table: str,
    ndivs: int,
    radius: float,
    alpha0: float,
    u0: float,
    project_velocity: bool,
) -> MetricDivergenceRow:
    """Compute D_r alpha + D_s beta diagnostics for one mesh refinement."""
    ref = build_reference_cache(order=order, table=table)

    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=radius)
    geom = build_geometry_cache(mesh, ref)

    axis = rotation_axis_from_alpha0(alpha0)
    omega = tuple(float(u0) * a for a in axis)

    volume = build_volume_rhs_cache(
        ref=ref,
        geom=geom,
        omega=omega,
        project_velocity=project_velocity,
        validate=True,
    )

    # This is the quantity appearing in the split-form energy balance:
    #
    #     D_r alpha + D_s beta.
    #
    # In the code, Dr_alpha and Ds_beta are computed by build_volume_rhs_cache.
    conservative_defect = volume.Dr_alpha + volume.Ds_beta

    # This is the physical divergence-like quantity:
    #
    #     (D_r alpha + D_s beta) / J.
    #
    # In the code this is stored as volume.div_velocity.
    physical_defect = volume.div_velocity

    conservative_linf = float(np.max(np.abs(conservative_defect)))
    physical_linf = float(np.max(np.abs(physical_defect)))

    conservative_l2_ref = weighted_l2_reference_quantity(
        conservative_defect,
        ref.weights,
        ref.area,
    )

    physical_l2 = weighted_l2_physical_quantity(
        physical_defect,
        volume.sqrt_g,
        ref.weights,
        ref.area,
    )

    conservative_weighted_mean = weighted_mean_reference_quantity(
        conservative_defect,
        ref.weights,
        ref.area,
    )

    physical_weighted_mean = weighted_mean_physical_quantity(
        physical_defect,
        volume.sqrt_g,
        ref.weights,
        ref.area,
    )

    return MetricDivergenceRow(
        order=order,
        table=table,
        ndivs=ndivs,
        n_elements=mesh.elements.shape[0],
        n_points_per_element=ref.rs.shape[0],
        hmin=minimum_face_length(ref, geom),
        conservative_linf=conservative_linf,
        conservative_l2_ref=conservative_l2_ref,
        conservative_weighted_mean=conservative_weighted_mean,
        physical_linf=physical_linf,
        physical_l2=physical_l2,
        physical_weighted_mean=physical_weighted_mean,
    )


def attach_rates(rows: list[MetricDivergenceRow]) -> list[MetricDivergenceRow]:
    """Attach adjacent-mesh convergence rates."""
    if not rows:
        return []

    rows_sorted = sorted(rows, key=lambda row: row.hmin, reverse=True)
    out: list[MetricDivergenceRow] = []

    prev: MetricDivergenceRow | None = None

    for row in rows_sorted:
        if prev is None:
            out.append(row)
        else:
            out.append(
                MetricDivergenceRow(
                    order=row.order,
                    table=row.table,
                    ndivs=row.ndivs,
                    n_elements=row.n_elements,
                    n_points_per_element=row.n_points_per_element,
                    hmin=row.hmin,
                    conservative_linf=row.conservative_linf,
                    conservative_l2_ref=row.conservative_l2_ref,
                    conservative_weighted_mean=row.conservative_weighted_mean,
                    physical_linf=row.physical_linf,
                    physical_l2=row.physical_l2,
                    physical_weighted_mean=row.physical_weighted_mean,
                    conservative_linf_rate=convergence_rate(
                        prev.conservative_linf,
                        row.conservative_linf,
                        prev.hmin,
                        row.hmin,
                    ),
                    conservative_l2_ref_rate=convergence_rate(
                        prev.conservative_l2_ref,
                        row.conservative_l2_ref,
                        prev.hmin,
                        row.hmin,
                    ),
                    physical_linf_rate=convergence_rate(
                        prev.physical_linf,
                        row.physical_linf,
                        prev.hmin,
                        row.hmin,
                    ),
                    physical_l2_rate=convergence_rate(
                        prev.physical_l2,
                        row.physical_l2,
                        prev.hmin,
                        row.hmin,
                    ),
                )
            )

        prev = row

    return out


def _fmt_rate(rate: float | None) -> str:
    if rate is None:
        return "-"
    return f"{rate:.2f}"


def print_table(rows: list[MetricDivergenceRow]) -> None:
    print()
    print("Metric-divergence diagnostic")
    print("Quantity 1: conservative defect = D_r alpha + D_s beta")
    print("Quantity 2: physical defect      = (D_r alpha + D_s beta) / J")
    print()

    header = (
        "ndivs | K     | hmin       | "
        "||cons||_inf | rate | ||cons||_L2(ref) | rate | "
        "||phys||_inf | rate | ||phys||_L2 | rate | "
        "|int cons| | |int phys|"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row.ndivs:<5d} | "
            f"{row.n_elements:<5d} | "
            f"{row.hmin:<10.4e} | "
            f"{row.conservative_linf:<12.4e} | "
            f"{_fmt_rate(row.conservative_linf_rate):>4s} | "
            f"{row.conservative_l2_ref:<16.4e} | "
            f"{_fmt_rate(row.conservative_l2_ref_rate):>4s} | "
            f"{row.physical_linf:<12.4e} | "
            f"{_fmt_rate(row.physical_linf_rate):>4s} | "
            f"{row.physical_l2:<12.4e} | "
            f"{_fmt_rate(row.physical_l2_rate):>4s} | "
            f"{row.conservative_weighted_mean:<10.4e} | "
            f"{row.physical_weighted_mean:<10.4e}"
        )

    print()


def _normalize_expr_option_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    expr_options = {"--alpha0", "--u0"}
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check convergence of the metric-divergence defect "
            "D_r alpha + D_s beta on the octahedron-based sphere mesh."
        )
    )

    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[4],
        help="Polynomial orders to check. Default: 4.",
    )

    parser.add_argument(
        "--ndivs",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Mesh subdivision counts. Default: 1 2 4 8.",
    )

    parser.add_argument(
        "--table",
        type=str,
        default="table1",
        choices=["table1", "table2"],
        help="Triangle quadrature table. Default: table1.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Sphere radius. Default: 1.0.",
    )

    parser.add_argument(
        "--alpha0",
        type=str,
        default="-pi/4",
        help="Rotation-axis parameter. Supports expressions such as -pi/4. Default: -pi/4.",
    )

    parser.add_argument(
        "--u0",
        type=str,
        default="2*pi/10",
        help="Angular speed scale. Supports expressions such as 2*pi/10. Default: 2*pi/10.",
    )

    parser.add_argument(
        "--no-project-velocity",
        action="store_true",
        help="Do not project the velocity field onto the tangent plane.",
    )

    if argv is None:
        argv = sys.argv[1:]

    return parser.parse_args(_normalize_expr_option_args(list(argv)))


def main() -> int:
    args = parse_args()

    alpha0 = parse_float_expr(args.alpha0)
    u0 = parse_float_expr(args.u0)
    project_velocity = not args.no_project_velocity

    ndivs_list = sorted(set(int(n) for n in args.ndivs))

    if any(n < 1 for n in ndivs_list):
        raise ValueError("All ndivs values must be positive integers.")

    for order in args.orders:
        rows = [
            compute_metric_divergence_row(
                order=order,
                table=args.table,
                ndivs=ndivs,
                radius=args.radius,
                alpha0=alpha0,
                u0=u0,
                project_velocity=project_velocity,
            )
            for ndivs in ndivs_list
        ]

        rows_with_rates = attach_rates(rows)

        print(f"order = {order}, table = {args.table}")
        print(f"alpha0 = {alpha0:.16e}, u0 = {u0:.16e}, radius = {args.radius:.16e}")
        print(f"project_velocity = {project_velocity}")

        print_table(rows_with_rates)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
