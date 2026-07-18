# examples/check_projected_sbp.py
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


from simplex_dg.reference import build_reference_cache


# Reference-face derivatives with respect to the edge parameter t in [0, 1].
#
# These must match simplex_dg.reference.quadrature.reference_edge_nodes:
#
# face 1: r =  1 - 2t, s = -1 + 2t
# face 2: r = -1,      s =  1 - 2t
# face 3: r = -1 + 2t, s = -1
#
# Therefore:
#   n_r dS =  ds/dt dt
#   n_s dS = -dr/dt dt
FACE_DRDT = np.array([-2.0, 0.0, 2.0], dtype=float)
FACE_DSDT = np.array([2.0, -2.0, 0.0], dtype=float)


@dataclass(frozen=True)
class ProjectedSBPReport:
    order: int
    table: str
    n_volume_points: int
    n_poly_basis: int
    n_face_points: int

    abs_error_r: float
    abs_error_s: float
    rel_error_r: float
    rel_error_s: float

    lhs_norm_r: float
    lhs_norm_s: float
    rhs_norm_r: float
    rhs_norm_s: float

    max_abs_entry_r: float
    max_abs_entry_s: float

    passed: bool


def _safe_relative_error(abs_error: float, reference_norm: float) -> float:
    return abs_error / max(reference_norm, np.finfo(float).tiny)


def _matrix_inf_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord=np.inf))


def _max_abs_entry(A: np.ndarray) -> float:
    return float(np.max(np.abs(A)))


def projected_sbp_boundary_matrices(ref) -> tuple[np.ndarray, np.ndarray]:
    """Build the projected SBP boundary matrices.

    The matrices are

        B_r = sum_gamma (E^gamma)^T W_b^gamma (ds/dt)_gamma E^gamma,

        B_s = sum_gamma (E^gamma)^T W_b^gamma ( -dr/dt)_gamma E^gamma.

    Here E^gamma is the projected boundary trace matrix V^gamma P.
    In the current code base, it is stored as ref.face_interp[face_id].
    """
    n_points = ref.rs.shape[0]

    boundary_r = np.zeros((n_points, n_points), dtype=float)
    boundary_s = np.zeros((n_points, n_points), dtype=float)

    for face_id in (1, 2, 3):
        f = face_id - 1

        E = np.asarray(ref.face_interp[face_id], dtype=float)
        wb = np.asarray(ref.edge_rules[face_id].weights, dtype=float)

        if E.ndim != 2:
            raise ValueError(f"face_interp[{face_id}] must be a 2D matrix.")

        if E.shape[1] != n_points:
            raise ValueError(
                f"face_interp[{face_id}] has wrong number of columns: "
                f"got {E.shape[1]}, expected {n_points}."
            )

        if E.shape[0] != wb.size:
            raise ValueError(
                f"edge weight size mismatch on face {face_id}: "
                f"E has {E.shape[0]} rows, weights have size {wb.size}."
            )

        drdt = FACE_DRDT[f]
        dsdt = FACE_DSDT[f]

        # Avoid forming diag(wb) explicitly:
        #
        #   E.T @ diag(wb * dsdt) @ E
        #   E.T @ diag(wb * (-drdt)) @ E
        #
        # Since wb is a 1D vector, (wb * factor)[:, None] * E scales each row.
        boundary_r += E.T @ ((wb * dsdt)[:, None] * E)
        boundary_s += E.T @ ((wb * (-drdt))[:, None] * E)

    return boundary_r, boundary_s


def projected_sbp_lhs_matrices(ref) -> tuple[np.ndarray, np.ndarray]:
    """Build the left-hand side matrices of the projected SBP relation.

    The quadrature weights stored in ref.weights are normalized with respect to
    the reference triangle rule. The reference area factor is stored separately
    as ref.area. Therefore the matrix identity checked here is

        ref.area * (W Dr + Dr.T W) = boundary_r,

        ref.area * (W Ds + Ds.T W) = boundary_s.

    If one defines W in the report as already including |T|, then this is the
    same as

        W Dr + Dr.T W = boundary_r,

        W Ds + Ds.T W = boundary_s.
    """
    W = np.diag(np.asarray(ref.weights, dtype=float))

    lhs_r = ref.area * (W @ ref.Dr + ref.Dr.T @ W)
    lhs_s = ref.area * (W @ ref.Ds + ref.Ds.T @ W)

    return lhs_r, lhs_s


def check_projected_sbp(
    *,
    order: int,
    table: str = "table1",
    n_face: int | None = None,
    tol: float = 1.0e-10,
) -> ProjectedSBPReport:
    """Check the projected SBP identities for one polynomial order."""
    ref = build_reference_cache(
        order=order,
        table=table,
        n_face=n_face,
        validate=True,
    )

    lhs_r, lhs_s = projected_sbp_lhs_matrices(ref)
    rhs_r, rhs_s = projected_sbp_boundary_matrices(ref)

    residual_r = lhs_r - rhs_r
    residual_s = lhs_s - rhs_s

    abs_error_r = _matrix_inf_norm(residual_r)
    abs_error_s = _matrix_inf_norm(residual_s)

    rhs_norm_r = _matrix_inf_norm(rhs_r)
    rhs_norm_s = _matrix_inf_norm(rhs_s)

    rel_error_r = _safe_relative_error(abs_error_r, rhs_norm_r)
    rel_error_s = _safe_relative_error(abs_error_s, rhs_norm_s)

    passed = (abs_error_r <= tol) and (abs_error_s <= tol)

    n_face_points = ref.edge_rules[1].n_points

    return ProjectedSBPReport(
        order=order,
        table=table,
        n_volume_points=ref.rs.shape[0],
        n_poly_basis=ref.V.shape[1],
        n_face_points=n_face_points,
        abs_error_r=abs_error_r,
        abs_error_s=abs_error_s,
        rel_error_r=rel_error_r,
        rel_error_s=rel_error_s,
        lhs_norm_r=_matrix_inf_norm(lhs_r),
        lhs_norm_s=_matrix_inf_norm(lhs_s),
        rhs_norm_r=rhs_norm_r,
        rhs_norm_s=rhs_norm_s,
        max_abs_entry_r=_max_abs_entry(residual_r),
        max_abs_entry_s=_max_abs_entry(residual_s),
        passed=passed,
    )


def format_report(report: ProjectedSBPReport) -> str:
    status = "PASS" if report.passed else "FAIL"

    return (
        f"[{status}] "
        f"order={report.order}, table={report.table}, "
        f"Nq={report.n_volume_points}, Np={report.n_poly_basis}, "
        f"Nf={report.n_face_points}\n"
        f"  r-direction:\n"
        f"    ||LHS_r - RHS_r||_inf = {report.abs_error_r:.6e}\n"
        f"    relative error        = {report.rel_error_r:.6e}\n"
        f"    max abs entry error   = {report.max_abs_entry_r:.6e}\n"
        f"    ||LHS_r||_inf         = {report.lhs_norm_r:.6e}\n"
        f"    ||RHS_r||_inf         = {report.rhs_norm_r:.6e}\n"
        f"  s-direction:\n"
        f"    ||LHS_s - RHS_s||_inf = {report.abs_error_s:.6e}\n"
        f"    relative error        = {report.rel_error_s:.6e}\n"
        f"    max abs entry error   = {report.max_abs_entry_s:.6e}\n"
        f"    ||LHS_s||_inf         = {report.lhs_norm_s:.6e}\n"
        f"    ||RHS_s||_inf         = {report.rhs_norm_s:.6e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the projected SBP identities "
            "area*(W D + D^T W) = sum E^T W_b N E."
        )
    )

    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Polynomial orders to check. Default: 1 2 3 4.",
    )

    parser.add_argument(
        "--table",
        type=str,
        default="table1",
        choices=["table1", "table2"],
        help="Triangle quadrature table. Default: table1.",
    )

    parser.add_argument(
        "--n-face",
        type=int,
        default=None,
        help=(
            "Number of face quadrature points. "
            "Default: order + 1, consistent with build_reference_cache."
        ),
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-10,
        help="Absolute infinity-norm tolerance. Default: 1e-10.",
    )

    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Print diagnostics but do not return a nonzero exit code on failure.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    all_passed = True

    for order in args.orders:
        report = check_projected_sbp(
            order=order,
            table=args.table,
            n_face=args.n_face,
            tol=args.tol,
        )

        print(format_report(report))
        print()

        all_passed = all_passed and report.passed

    if (not all_passed) and (not args.no_fail):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())