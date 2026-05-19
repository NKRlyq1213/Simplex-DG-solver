from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from simplex_dg.diagnostics import (
    ConvergenceRow,
    error_report,
    format_convergence_table,
    write_convergence_csv,
)
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.problems import exact_gaussian_solid_body, gaussian_on_sphere
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache, full_rhs_split
from simplex_dg.time import (
    cfl_dt_from_geometry,
    integrate_lsrk54,
    manifold_integral,
    manifold_l2_norm,
    minimum_face_length,
)
from simplex_dg.trace import build_trace_cache


def run_one_level(
    *,
    level: int,
    order: int,
    table: str,
    cfl: float,
    tf: float,
    sigma: float,
    radius: float,
    flux_type: str,
    use_numba: bool,
) -> ConvergenceRow:
    omega = (0.0, 0.0, 1.0)
    center0 = (radius, 0.0, 0.0)

    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(level=level, radius=radius)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=omega,
        flux_type=flux_type,
        constant_preserving=True,
    )

    q0 = gaussian_on_sphere(
        X=geom.X,
        center=center0,
        radius=radius,
        sigma=sigma,
        amplitude=1.0,
    )

    dt_raw = cfl_dt_from_geometry(
        ref=ref,
        geom=geom,
        max_speed=full.volume.max_speed,
        cfl=cfl,
    )

    nsteps = max(1, int(np.ceil(tf / dt_raw)))
    dt = float(tf / nsteps)

    def rhs(t, q):
        return full_rhs_split(q, full, use_numba=use_numba)

    result = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=0.0,
        tf=tf,
        dt=dt,
        monitor=None,
    )

    q_exact = exact_gaussian_solid_body(
        X=geom.X,
        t=tf,
        radius=radius,
        sigma=sigma,
        amplitude=1.0,
        center0=center0,
        omega=omega,
    )

    rep = error_report(result.q, q_exact, ref, geom)

    mass0 = manifold_integral(q0, ref, geom)
    massf = manifold_integral(result.q, ref, geom)

    l20 = manifold_l2_norm(q0, ref, geom)
    l2f = manifold_l2_norm(result.q, ref, geom)

    return ConvergenceRow(
        level=level,
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
        mass_drift=massf - mass0,
        l2_norm_drift=l2f - l20,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian solid-body advection convergence runner.")

    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--table", type=str, default="table1")
    parser.add_argument("--cfl", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--flux", type=str, default="upwind", choices=["upwind", "central", "lf"])
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/convergence/gaussian_sphere_convergence.csv")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows: list[ConvergenceRow] = []

    print("Gaussian convergence run")
    print("------------------------")
    print(f"levels : {args.levels}")
    print(f"order  : {args.order}")
    print(f"table  : {args.table}")
    print(f"cfl    : {args.cfl}")
    print(f"tf     : {args.tf}")
    print(f"sigma  : {args.sigma}")
    print(f"flux   : {args.flux}")
    print()

    t0 = time.perf_counter()

    for level in args.levels:
        start = time.perf_counter()

        row = run_one_level(
            level=level,
            order=args.order,
            table=args.table,
            cfl=args.cfl,
            tf=args.tf,
            sigma=args.sigma,
            radius=args.radius,
            flux_type=args.flux,
            use_numba=not args.no_numba,
        )

        rows.append(row)

        elapsed = time.perf_counter() - start

        print(
            f"level={level}, K={row.n_elements}, DOFs={row.total_dofs}, "
            f"L2={row.l2_error:.6e}, rel={row.relative_l2_error:.6e}, "
            f"Linf={row.linf_error:.6e}, mass drift={row.mass_drift:+.6e}, "
            f"time={elapsed:.2f}s"
        )

    print()
    print(format_convergence_table(rows))

    output = Path(args.output)
    write_convergence_csv(output, rows)

    elapsed_total = time.perf_counter() - t0

    print()
    print(f"CSV written to: {output}")
    print(f"total elapsed: {elapsed_total:.2f}s")


if __name__ == "__main__":
    main()