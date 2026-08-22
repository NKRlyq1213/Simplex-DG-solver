from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import json
import operator
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.problems import gaussian_on_sphere
from simplex_dg.reference import build_reference_cache
from simplex_dg.reference.quadrature import REFERENCE_AREA
from simplex_dg.rhs import (
    build_full_rhs_cache,
    common_projected_line_velocity,
    full_rhs,
    projected_line_velocity,
)
from simplex_dg.time import minimum_face_length
from simplex_dg.trace import build_trace_cache, gather_neighbor_traces, pair_face_traces


TABLE = "table1"
SBP_VARIANT = "projected"
VOLUME_FORM = "split"
FLUX_TYPE = "central"
RELATIVE_EPS = 1.0e-30


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


FACE_FIELDS = [
    "ndivs",
    "state",
    "element",
    "face",
    "neighbor_element",
    "neighbor_face",
    "n_face_points",
    "max_abs_aM",
    "max_abs_aP",
    "max_abs_metric_mismatch",
    "relative_metric_mismatch",
    "weighted_l2_metric_mismatch",
    "R_geo_face",
    "I_direct_face",
    "interface_formula_error",
]


SUMMARY_FIELDS = [
    "order",
    "ndivs",
    "n_elements",
    "n_points_per_element",
    "total_dofs",
    "state",
    "hmin",
    "max_abs_metric_mismatch",
    "max_relative_metric_mismatch",
    "global_metric_l2",
    "R_geo",
    "abs_R_geo",
    "I_direct_global",
    "interface_formula_error",
    "max_face_interface_formula_error",
    "dE_dt_actual",
    "physical_volume_rate",
    "energy_defect",
    "balance_error",
    "abs_balance_error",
    "relative_balance_error",
    "dM_dt",
    "abs_mass_rate",
    "max_common_velocity_reconstruction_error",
    "metric_mismatch_rate",
    "R_geo_abs_rate",
    "face_csv",
]


@dataclass(frozen=True)
class DiscreteObjects:
    ref: Any
    mesh: Any
    conn: Any
    geom: Any
    trace: Any
    full: Any
    hmin: float


def parse_float_expr(value: str | float | int | None) -> float | None:
    """Parse CLI expressions such as 1.0, pi/4, and -pi/4."""
    if value is None:
        return None

    if isinstance(value, (float, int)):
        return float(value)

    s = str(value).strip()

    try:
        return float(s)
    except ValueError:
        pass

    node = ast.parse(s, mode="eval").body

    def eval_node(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)

        if isinstance(n, ast.Name) and n.id == "pi":
            return float(np.pi)

        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
            return float(_ALLOWED_UNARYOPS[type(n.op)](eval_node(n.operand)))

        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return float(_ALLOWED_BINOPS[type(n.op)](eval_node(n.left), eval_node(n.right)))

        raise ValueError(f"Unsupported numeric expression: {value!r}")

    return float(eval_node(node))


def _normalize_expr_option_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    expr_options = {
        "--radius",
        "--sigma",
        "--sigma-physical",
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


def rotation_axis_from_alpha0(alpha0: float) -> tuple[float, float, float]:
    """Match examples/step9_gaussian_convergence.py."""
    return (
        -float(np.sin(alpha0)),
        0.0,
        float(np.cos(alpha0)),
    )


def resolve_sigma_physical(
    *,
    radius: float,
    sigma_angle: float,
    sigma_physical: float | None,
) -> float:
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    sigma = float(radius) * float(sigma_angle) if sigma_physical is None else float(sigma_physical)

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    return sigma


def build_discrete_objects(
    *,
    order: int,
    ndivs: int,
    radius: float,
    alpha0: float,
    u0: float,
) -> DiscreteObjects:
    axis = rotation_axis_from_alpha0(alpha0)
    omega = tuple(float(u0) * component for component in axis)

    ref = build_reference_cache(
        order=order,
        table=TABLE,
        sbp_variant=SBP_VARIANT,
    )
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=radius)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=omega,
        flux_type=FLUX_TYPE,
        lf_alpha=0.0,
        volume_form=VOLUME_FORM,
    )

    return DiscreteObjects(
        ref=ref,
        mesh=mesh,
        conn=conn,
        geom=geom,
        trace=trace,
        full=full,
        hmin=minimum_face_length(ref, geom),
    )


def polynomial_state(ref: Any, n_elements: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    coeffs = rng.normal(loc=0.0, scale=0.1, size=(n_elements, ref.V.shape[1]))
    return coeffs @ np.asarray(ref.V, dtype=float).T


def build_states(
    objects: DiscreteObjects,
    *,
    radius: float,
    sigma_physical: float,
    amplitude: float,
) -> list[tuple[str, np.ndarray]]:
    geom = objects.geom

    q_constant = np.ones_like(geom.sqrt_g, dtype=float)
    q_gaussian = gaussian_on_sphere(
        X=geom.X,
        center=(radius, 0.0, 0.0),
        radius=radius,
        sigma=sigma_physical,
        amplitude=amplitude,
    )
    q_polynomial = polynomial_state(objects.ref, objects.mesh.elements.shape[0])

    return [
        ("constant", q_constant),
        ("gaussian", q_gaussian),
        ("polynomial", q_polynomial),
    ]


def unique_internal_faces(trace: Any) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    faces: list[tuple[int, int, int, int]] = []

    for k in range(trace.n_elements):
        for f in range(trace.n_faces):
            if bool(trace.is_boundary[k, f]):
                continue

            neighbor = int(trace.neighbor_elements[k, f])
            neighbor_face = int(trace.neighbor_faces[k, f])
            pair = tuple(sorted(((int(k), int(f)), (neighbor, neighbor_face))))

            if pair in seen:
                continue

            seen.add(pair)
            owner_k, owner_f = pair[0]
            owner_neighbor = int(trace.neighbor_elements[owner_k, owner_f])
            owner_neighbor_face = int(trace.neighbor_faces[owner_k, owner_f])
            faces.append((owner_k, owner_f, owner_neighbor, owner_neighbor_face))

    return faces


def observed_rate(
    old_error: float,
    new_error: float,
    old_h: float,
    new_h: float,
) -> float | None:
    if not all(np.isfinite(v) for v in (old_error, new_error, old_h, new_h)):
        return None

    if old_error <= 0.0 or new_error <= 0.0:
        return None

    if old_h <= 0.0 or new_h <= 0.0 or np.isclose(old_h, new_h):
        return None

    return float(np.log(old_error / new_error) / np.log(old_h / new_h))


def diagnose_state(
    *,
    objects: DiscreteObjects,
    q: np.ndarray,
    state: str,
    order: int,
    ndivs: int,
    face_csv_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ref = objects.ref
    geom = objects.geom
    trace = objects.trace
    full = objects.full

    traces = pair_face_traces(q, trace, use_numba=False)
    qM = traces.qM
    qP = traces.qP

    aM = projected_line_velocity(full.volume, full.surface)
    aP = gather_neighbor_traces(
        aM,
        trace,
        boundary_value=np.nan,
        use_numba=False,
    )

    a_common_existing = common_projected_line_velocity(
        full.volume,
        full.surface,
        trace,
        use_numba=False,
    )
    a_common_direct = 0.5 * (aM - aP)

    internal_mask = ~trace.is_boundary
    if np.any(internal_mask):
        common_error = float(np.nanmax(np.abs((a_common_existing - a_common_direct)[internal_mask])))
    else:
        common_error = 0.0

    face_rows: list[dict[str, Any]] = []

    for element, face, neighbor_element, neighbor_face in unique_internal_faces(trace):
        weights = np.asarray(trace.face_weights[face], dtype=float)

        qM_f = np.asarray(qM[element, face], dtype=float)
        qP_f = np.asarray(qP[element, face], dtype=float)
        aM_f = np.asarray(aM[element, face], dtype=float)
        aP_f = np.asarray(aP[element, face], dtype=float)

        mismatch = aM_f + aP_f
        max_abs_aM = float(np.max(np.abs(aM_f)))
        max_abs_aP = float(np.max(np.abs(aP_f)))
        max_abs_mismatch = float(np.max(np.abs(mismatch)))
        relative_mismatch = max_abs_mismatch / max(max_abs_aM, max_abs_aP, RELATIVE_EPS)
        weighted_l2_mismatch = float(np.sqrt(np.sum(weights * mismatch * mismatch)))

        R_geo_face = float(0.25 * np.sum(weights * mismatch * (qM_f * qM_f + qP_f * qP_f)))
        a_common = 0.5 * (aM_f - aP_f)
        fstar = 0.5 * a_common * (qM_f + qP_f)
        I_direct_face = float(
            np.sum(
                weights
                * (
                    0.5 * aM_f * qM_f * qM_f
                    + 0.5 * aP_f * qP_f * qP_f
                    + (qP_f - qM_f) * fstar
                )
            )
        )
        interface_formula_error = abs(I_direct_face - R_geo_face)

        face_rows.append(
            {
                "ndivs": int(ndivs),
                "state": state,
                "element": int(element),
                "face": int(face),
                "neighbor_element": int(neighbor_element),
                "neighbor_face": int(neighbor_face),
                "n_face_points": int(trace.n_face_points),
                "max_abs_aM": max_abs_aM,
                "max_abs_aP": max_abs_aP,
                "max_abs_metric_mismatch": max_abs_mismatch,
                "relative_metric_mismatch": float(relative_mismatch),
                "weighted_l2_metric_mismatch": weighted_l2_mismatch,
                "R_geo_face": R_geo_face,
                "I_direct_face": I_direct_face,
                "interface_formula_error": float(interface_formula_error),
            }
        )

    rhs = full_rhs(q, full, use_numba=False)
    volume_weights = ref.weights[None, :]

    dE_dt_actual = float(REFERENCE_AREA * np.sum(volume_weights * geom.sqrt_g * q * rhs))
    physical_volume_rate = float(
        -REFERENCE_AREA
        * np.sum(
            volume_weights
            * 0.5
            * q
            * q
            * (full.volume.Dr_alpha + full.volume.Ds_beta)
        )
    )
    energy_defect = dE_dt_actual - physical_volume_rate

    dM_dt = float(REFERENCE_AREA * np.sum(volume_weights * geom.sqrt_g * rhs))

    R_geo = float(sum(row["R_geo_face"] for row in face_rows))
    I_direct_global = float(sum(row["I_direct_face"] for row in face_rows))
    balance_error = energy_defect - R_geo
    relative_balance_error = abs(balance_error) / max(abs(energy_defect), abs(R_geo), RELATIVE_EPS)

    summary = {
        "order": int(order),
        "ndivs": int(ndivs),
        "n_elements": int(objects.mesh.elements.shape[0]),
        "n_points_per_element": int(ref.rs.shape[0]),
        "total_dofs": int(objects.mesh.elements.shape[0] * ref.rs.shape[0]),
        "state": state,
        "hmin": float(objects.hmin),
        "max_abs_metric_mismatch": float(
            max((row["max_abs_metric_mismatch"] for row in face_rows), default=0.0)
        ),
        "max_relative_metric_mismatch": float(
            max((row["relative_metric_mismatch"] for row in face_rows), default=0.0)
        ),
        "global_metric_l2": float(
            np.sqrt(sum(row["weighted_l2_metric_mismatch"] ** 2 for row in face_rows))
        ),
        "R_geo": R_geo,
        "abs_R_geo": abs(R_geo),
        "I_direct_global": I_direct_global,
        "interface_formula_error": abs(I_direct_global - R_geo),
        "max_face_interface_formula_error": float(
            max((row["interface_formula_error"] for row in face_rows), default=0.0)
        ),
        "dE_dt_actual": dE_dt_actual,
        "physical_volume_rate": physical_volume_rate,
        "energy_defect": energy_defect,
        "balance_error": balance_error,
        "abs_balance_error": abs(balance_error),
        "relative_balance_error": float(relative_balance_error),
        "dM_dt": dM_dt,
        "abs_mass_rate": abs(dM_dt),
        "max_common_velocity_reconstruction_error": common_error,
        "metric_mismatch_rate": None,
        "R_geo_abs_rate": None,
        "face_csv": face_csv_name,
    }

    return summary, face_rows


def attach_rates(rows: list[dict[str, Any]]) -> None:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_state.setdefault(str(row["state"]), []).append(row)

    for state_rows in by_state.values():
        state_rows.sort(key=lambda row: float(row["hmin"]), reverse=True)
        previous: dict[str, Any] | None = None

        for row in state_rows:
            if previous is not None:
                row["metric_mismatch_rate"] = observed_rate(
                    float(previous["max_abs_metric_mismatch"]),
                    float(row["max_abs_metric_mismatch"]),
                    float(previous["hmin"]),
                    float(row["hmin"]),
                )
                row["R_geo_abs_rate"] = observed_rate(
                    abs(float(previous["R_geo"])),
                    abs(float(row["R_geo"])),
                    float(previous["hmin"]),
                    float(row["hmin"]),
                )

            previous = row


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, np.generic):
        return value.item()

    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if value is None:
                    out[key] = ""
            writer.writerow(out)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_console_table(rows: list[dict[str, Any]]) -> None:
    print()
    print("ndiv | state      | max|aM+aP|  | R_geo        | E'-R_phys   | balance err | |M'|")
    print("-----+------------+-------------+--------------+-------------+-------------+-------------")

    for row in sorted(rows, key=lambda r: (int(r["ndivs"]), str(r["state"]))):
        print(
            f"{int(row['ndivs']):>4d} | "
            f"{str(row['state']):<10s} | "
            f"{float(row['max_abs_metric_mismatch']): .4e} | "
            f"{float(row['R_geo']): .4e} | "
            f"{float(row['energy_defect']): .4e} | "
            f"{float(row['balance_error']): .4e} | "
            f"{float(row['abs_mass_rate']): .4e}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pure diagnostic for projected split central interface metric mismatch "
            "in Simplex-DG-solver."
        )
    )
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--ndivs", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--radius", type=parse_float_expr, default=1.0)
    parser.add_argument("--alpha0", type=parse_float_expr, default=-np.pi / 4.0)
    parser.add_argument("--u0", type=parse_float_expr, default=2.0 * np.pi / 10.0)
    parser.add_argument("--sigma", type=parse_float_expr, default=0.35)
    parser.add_argument("--sigma-physical", type=parse_float_expr, default=None)
    parser.add_argument("--amplitude", type=parse_float_expr, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/projected_split_interface_metric",
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

    if args.order <= 0:
        parser.error("order must be positive.")

    if args.radius <= 0.0:
        parser.error("radius must be positive.")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sigma_physical = resolve_sigma_physical(
        radius=float(args.radius),
        sigma_angle=float(args.sigma),
        sigma_physical=args.sigma_physical,
    )
    output_dir = ROOT / args.output_dir

    print("Projected split central interface metric diagnostic")
    print("---------------------------------------------------")
    print(f"table         : {TABLE}")
    print(f"sbp_variant   : {SBP_VARIANT}")
    print(f"volume_form   : {VOLUME_FORM}")
    print(f"flux_type     : {FLUX_TYPE}")
    print(f"order         : {args.order}")
    print(f"ndivs         : {args.ndivs}")
    print(f"radius        : {args.radius}")
    print(f"alpha0        : {args.alpha0}")
    print(f"u0            : {args.u0}")
    print(f"sigma_angle   : {args.sigma}")
    print(f"sigma_physical: {sigma_physical}")
    print(f"amplitude     : {args.amplitude}")
    print(f"output_dir    : {output_dir}")
    print()

    summary_rows: list[dict[str, Any]] = []
    face_files: list[str] = []

    for ndivs in args.ndivs:
        objects = build_discrete_objects(
            order=int(args.order),
            ndivs=int(ndivs),
            radius=float(args.radius),
            alpha0=float(args.alpha0),
            u0=float(args.u0),
        )
        states = build_states(
            objects,
            radius=float(args.radius),
            sigma_physical=sigma_physical,
            amplitude=float(args.amplitude),
        )

        for state, q in states:
            face_csv_name = f"faces_ndiv{ndivs}_{state}.csv"
            summary, face_rows = diagnose_state(
                objects=objects,
                q=q,
                state=state,
                order=int(args.order),
                ndivs=int(ndivs),
                face_csv_name=face_csv_name,
            )

            write_csv(output_dir / face_csv_name, face_rows, FACE_FIELDS)
            face_files.append(face_csv_name)
            summary_rows.append(summary)

            print(
                f"ndivs={ndivs}, state={state:<10s}, "
                f"max|aM+aP|={summary['max_abs_metric_mismatch']:.6e}, "
                f"R_geo={summary['R_geo']:.6e}, "
                f"Edefect={summary['energy_defect']:.6e}, "
                f"balance={summary['balance_error']:.6e}, "
                f"|M'|={summary['abs_mass_rate']:.6e}"
            )

    attach_rates(summary_rows)

    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_json(
        output_dir / "summary.json",
        {
            "configuration": {
                "table": TABLE,
                "sbp_variant": SBP_VARIANT,
                "volume_form": VOLUME_FORM,
                "flux_type": FLUX_TYPE,
                "order": int(args.order),
                "ndivs": [int(v) for v in args.ndivs],
                "radius": float(args.radius),
                "alpha0": float(args.alpha0),
                "u0": float(args.u0),
                "omega": [
                    float(args.u0) * component
                    for component in rotation_axis_from_alpha0(float(args.alpha0))
                ],
                "sigma_angle": float(args.sigma),
                "sigma_physical": sigma_physical,
                "amplitude": float(args.amplitude),
                "reference_area": float(REFERENCE_AREA),
                "use_numba": False,
            },
            "summary": summary_rows,
            "face_files": face_files,
        },
    )

    print_console_table(summary_rows)

    print()
    print(f"summary CSV written to : {output_dir / 'summary.csv'}")
    print(f"summary JSON written to: {output_dir / 'summary.json'}")
    print(f"face CSV files written : {len(face_files)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
