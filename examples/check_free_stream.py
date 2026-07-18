from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


try:
    from check_metric_divergence import (
        _normalize_expr_option_args,
        convergence_rate,
        parse_float_expr,
        rotation_axis_from_alpha0,
    )
except ModuleNotFoundError:
    from examples.check_metric_divergence import (
        _normalize_expr_option_args,
        convergence_rate,
        parse_float_expr,
        rotation_axis_from_alpha0,
    )
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache, full_rhs
from simplex_dg.time import manifold_integral, minimum_face_length
from simplex_dg.trace import build_trace_cache


@dataclass(frozen=True)
class FreeStreamRow:
    order: int
    table: str
    volume_form: str
    flux_type: str
    lf_alpha: float
    ndivs: int
    n_elements: int
    n_points_per_element: int
    hmin: float
    constant_linf: float
    constant_physical_l2: float
    constant_global_integral: float
    constant_linf_rate: float | None = None
    constant_physical_l2_rate: float | None = None


def compute_free_stream_row(
    *,
    order: int,
    table: str,
    ndivs: int,
    radius: float,
    alpha0: float,
    u0: float,
    flux_type: str,
    lf_alpha: float,
    volume_form: str,
    project_velocity: bool,
) -> FreeStreamRow:
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=radius)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)

    axis = rotation_axis_from_alpha0(alpha0)
    omega = tuple(float(u0) * a for a in axis)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=omega,
        flux_type=flux_type,
        lf_alpha=lf_alpha,
        project_velocity=project_velocity,
        volume_form=volume_form,
        validate=True,
    )

    ones = np.ones((mesh.elements.shape[0], ref.rs.shape[0]))
    rhs = full_rhs(ones, full, use_numba=False)

    return FreeStreamRow(
        order=order,
        table=table,
        volume_form=full.volume_form,
        flux_type=full.surface.flux_type,
        lf_alpha=full.surface.lf_alpha,
        ndivs=ndivs,
        n_elements=mesh.elements.shape[0],
        n_points_per_element=ref.rs.shape[0],
        hmin=minimum_face_length(ref, geom),
        constant_linf=float(np.max(np.abs(rhs))),
        constant_physical_l2=float(
            np.sqrt(np.sum(ref.area * ref.weights[None, :] * geom.sqrt_g * rhs * rhs))
        ),
        constant_global_integral=abs(manifold_integral(rhs, ref, geom)),
    )


def attach_rates(rows: list[FreeStreamRow]) -> list[FreeStreamRow]:
    if not rows:
        return []

    rows_sorted = sorted(rows, key=lambda row: row.hmin, reverse=True)
    out: list[FreeStreamRow] = []
    prev: FreeStreamRow | None = None

    for row in rows_sorted:
        if prev is None:
            out.append(row)
        else:
            out.append(
                FreeStreamRow(
                    order=row.order,
                    table=row.table,
                    volume_form=row.volume_form,
                    flux_type=row.flux_type,
                    lf_alpha=row.lf_alpha,
                    ndivs=row.ndivs,
                    n_elements=row.n_elements,
                    n_points_per_element=row.n_points_per_element,
                    hmin=row.hmin,
                    constant_linf=row.constant_linf,
                    constant_physical_l2=row.constant_physical_l2,
                    constant_global_integral=row.constant_global_integral,
                    constant_linf_rate=convergence_rate(
                        prev.constant_linf,
                        row.constant_linf,
                        prev.hmin,
                        row.hmin,
                    ),
                    constant_physical_l2_rate=convergence_rate(
                        prev.constant_physical_l2,
                        row.constant_physical_l2,
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


def print_table(rows: list[FreeStreamRow]) -> None:
    print()
    print("Free-stream diagnostic")
    print("Quantity: full RHS applied to the constant state q = 1")
    print()

    header = (
        "ndivs | K     | hmin       | "
        "||R(1)||_inf | rate | ||R(1)||_L2 | rate | |int R(1)|"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row.ndivs:<5d} | "
            f"{row.n_elements:<5d} | "
            f"{row.hmin:<10.4e} | "
            f"{row.constant_linf:<12.4e} | "
            f"{_fmt_rate(row.constant_linf_rate):>4s} | "
            f"{row.constant_physical_l2:<12.4e} | "
            f"{_fmt_rate(row.constant_physical_l2_rate):>4s} | "
            f"{row.constant_global_integral:<10.4e}"
        )

    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the free-stream residual R_h(1) on the sphere mesh."
    )
    parser.add_argument("--table", type=str, default="table1", choices=["table1", "table2"])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--ndivs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--alpha0", type=str, default="-pi/4")
    parser.add_argument("--u0", type=str, default="2*pi/10")
    parser.add_argument("--flux-type", type=str, default="central", choices=["central", "upwind", "lf"])
    parser.add_argument("--lf-alpha", type=float, default=1.0)
    parser.add_argument("--volume-form", type=str, default="conservative", choices=["conservative", "split"])
    parser.add_argument("--no-project-velocity", action="store_true")

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

    rows = [
        compute_free_stream_row(
            order=args.order,
            table=args.table,
            ndivs=ndivs,
            radius=args.radius,
            alpha0=alpha0,
            u0=u0,
            flux_type=args.flux_type,
            lf_alpha=args.lf_alpha,
            volume_form=args.volume_form,
            project_velocity=project_velocity,
        )
        for ndivs in ndivs_list
    ]

    rows_with_rates = attach_rates(rows)

    print(f"table = {args.table}, order = {args.order}")
    print(f"volume_form = {args.volume_form}, flux_type = {args.flux_type}, lf_alpha = {args.lf_alpha:.16e}")
    print(f"alpha0 = {alpha0:.16e}, u0 = {u0:.16e}, radius = {args.radius:.16e}")
    print(f"project_velocity = {project_velocity}")
    print_table(rows_with_rates)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
