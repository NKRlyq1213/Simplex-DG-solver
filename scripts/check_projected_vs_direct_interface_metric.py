from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import operator
from pathlib import Path
import subprocess
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
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache, projected_line_velocity
from simplex_dg.time import minimum_face_length
from simplex_dg.trace import build_trace_cache, gather_neighbor_traces


TABLE = "table1"
SBP_VARIANT = "projected"
VOLUME_FORM = "split"
FLUX_TYPE = "central"
ROUND_OFF_TOL = 1.0e-12
RELATIVE_EPS = 1.0e-30

FACE_CONORMAL_INTERPRETATION = "unit co-normal"
DIRECT_COEFFICIENT_FORMULA = "geom.face_jacobian * full.surface.normal_velocity"
DIRECT_COEFFICIENT_REASON = (
    "src/simplex_dg/geometry/sphere.py::map_reference_face_to_sphere_element "
    "normalizes face_conormal to unit length, and validate_geometry_cache checks "
    "||geom.face_conormal|| = 1. src/simplex_dg/rhs/surface.py::build_surface_rhs_cache "
    "sets normal_velocity = sum(face_velocity * geom.face_conormal, axis=3), so "
    "normal_velocity is u dot unit_conormal and must be multiplied by face_jacobian "
    "to obtain the oriented physical line coefficient."
)


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
    "element",
    "face",
    "neighbor_element",
    "neighbor_face",
    "n_face_points",
    "max_abs_projected_M",
    "max_abs_projected_P",
    "max_abs_projected_mismatch",
    "weighted_l2_projected_mismatch",
    "relative_projected_mismatch",
    "max_abs_direct_M",
    "max_abs_direct_P",
    "max_abs_direct_mismatch",
    "weighted_l2_direct_mismatch",
    "relative_direct_mismatch",
    "max_abs_projected_minus_direct_M",
    "max_abs_projected_minus_direct_P",
    "relative_projected_direct_difference",
    "max_abs_decomposition_error",
    "max_face_coordinate_mismatch",
    "max_face_conormal_mismatch",
    "max_face_jacobian_mismatch",
]


SUMMARY_FIELDS = [
    "order",
    "ndivs",
    "n_elements",
    "n_points_per_element",
    "total_dofs",
    "hmin",
    "max_abs_projected_M",
    "max_abs_direct_M",
    "max_abs_projected_minus_direct_M",
    "relative_projected_direct_M_difference",
    "max_projected_mismatch",
    "global_projected_l2",
    "max_relative_projected_mismatch",
    "max_direct_mismatch",
    "global_direct_l2",
    "max_relative_direct_mismatch",
    "max_projected_direct_difference",
    "max_relative_projected_direct_difference",
    "max_decomposition_error",
    "max_face_coordinate_mismatch",
    "max_face_conormal_mismatch",
    "max_face_jacobian_mismatch",
    "projected_mismatch_rate",
    "direct_mismatch_rate",
    "projected_direct_difference_rate",
    "direct_coefficient_formula",
    "direct_coefficient_interpretation",
    "classification",
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


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    commit = result.stdout.strip()
    return commit or "unknown"


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
            owner_element, owner_face = pair[0]
            owner_neighbor = int(trace.neighbor_elements[owner_element, owner_face])
            owner_neighbor_face = int(trace.neighbor_faces[owner_element, owner_face])
            faces.append((owner_element, owner_face, owner_neighbor, owner_neighbor_face))

    return faces


def gather_neighbor_face_array(values: np.ndarray, trace: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    expected = (trace.n_elements, trace.n_faces, trace.n_face_points)

    if values.shape[:3] != expected:
        raise ValueError(f"values leading dimensions must be {expected}.")

    gathered = np.full_like(values, np.nan, dtype=float)

    for k in range(trace.n_elements):
        for f in range(trace.n_faces):
            if bool(trace.is_boundary[k, f]):
                continue

            neighbor = int(trace.neighbor_elements[k, f])
            neighbor_face = int(trace.neighbor_faces[k, f])
            face_values = values[neighbor, neighbor_face]

            if bool(trace.face_flip[k, f]):
                face_values = face_values[::-1, ...]

            gathered[k, f] = face_values

    return gathered


def weighted_l2(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights * values * values)))


def relative(max_abs_error: float, *scales: float) -> float:
    return float(max_abs_error / max(*(abs(float(scale)) for scale in scales), RELATIVE_EPS))


def observed_rate(
    old_error: float,
    new_error: float,
    old_h: float,
    new_h: float,
    *,
    roundoff_tol: float = ROUND_OFF_TOL,
) -> float:
    if not all(np.isfinite(v) for v in (old_error, new_error, old_h, new_h)):
        return float("nan")

    if old_error <= roundoff_tol or new_error <= roundoff_tol:
        return float("nan")

    if old_h <= 0.0 or new_h <= 0.0 or np.isclose(old_h, new_h):
        return float("nan")

    return float(np.log(old_error / new_error) / np.log(old_h / new_h))


def direct_line_coefficient(objects: DiscreteObjects) -> np.ndarray:
    return np.asarray(objects.geom.face_jacobian, dtype=float) * np.asarray(
        objects.full.surface.normal_velocity,
        dtype=float,
    )


def classify_rows(rows: list[dict[str, Any]]) -> str:
    max_projected = max(float(row["max_projected_mismatch"]) for row in rows)
    max_direct = max(float(row["max_direct_mismatch"]) for row in rows)

    if max_direct <= ROUND_OFF_TOL and max_projected > ROUND_OFF_TOL:
        return "Case A: Direct compatible, projected incompatible"

    if max_direct > ROUND_OFF_TOL:
        return "Case B: Direct also incompatible"

    return "Case C: Both projected and direct are roundoff compatible"


def diagnose_ndiv(
    *,
    objects: DiscreteObjects,
    order: int,
    ndivs: int,
    face_csv_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    geom = objects.geom
    trace = objects.trace
    full = objects.full

    a_projected_M = projected_line_velocity(full.volume, full.surface)
    a_projected_P = gather_neighbor_traces(
        a_projected_M,
        trace,
        boundary_value=np.nan,
        use_numba=False,
    )

    a_direct_M = direct_line_coefficient(objects)
    a_direct_P = gather_neighbor_traces(
        a_direct_M,
        trace,
        boundary_value=np.nan,
        use_numba=False,
    )

    X_face_P = gather_neighbor_face_array(geom.X_face, trace)
    conormal_P = gather_neighbor_face_array(geom.face_conormal, trace)
    face_jacobian_P = gather_neighbor_traces(
        geom.face_jacobian,
        trace,
        boundary_value=np.nan,
        use_numba=False,
    )

    face_rows: list[dict[str, Any]] = []
    worst_direct: dict[str, Any] = {
        "abs_direct_mismatch": -1.0,
    }

    for element, face, neighbor_element, neighbor_face in unique_internal_faces(trace):
        weights = np.asarray(trace.face_weights[face], dtype=float)

        apM = np.asarray(a_projected_M[element, face], dtype=float)
        apP = np.asarray(a_projected_P[element, face], dtype=float)
        adM = np.asarray(a_direct_M[element, face], dtype=float)
        adP = np.asarray(a_direct_P[element, face], dtype=float)

        eps_projected = apM + apP
        eps_direct = adM + adP
        delta_projection_M = apM - adM
        delta_projection_P = apP - adP
        decomposition = eps_projected - (
            eps_direct + delta_projection_M + delta_projection_P
        )

        max_abs_projected_M = float(np.max(np.abs(apM)))
        max_abs_projected_P = float(np.max(np.abs(apP)))
        max_abs_direct_M = float(np.max(np.abs(adM)))
        max_abs_direct_P = float(np.max(np.abs(adP)))
        max_abs_projected_mismatch = float(np.max(np.abs(eps_projected)))
        max_abs_direct_mismatch = float(np.max(np.abs(eps_direct)))
        max_abs_projected_minus_direct_M = float(np.max(np.abs(delta_projection_M)))
        max_abs_projected_minus_direct_P = float(np.max(np.abs(delta_projection_P)))
        max_abs_projected_direct = max(
            max_abs_projected_minus_direct_M,
            max_abs_projected_minus_direct_P,
        )

        X_M = np.asarray(geom.X_face[element, face], dtype=float)
        X_P = np.asarray(X_face_P[element, face], dtype=float)
        conormal_M = np.asarray(geom.face_conormal[element, face], dtype=float)
        conormal_neighbor = np.asarray(conormal_P[element, face], dtype=float)
        Jf_M = np.asarray(geom.face_jacobian[element, face], dtype=float)
        Jf_P = np.asarray(face_jacobian_P[element, face], dtype=float)

        max_face_coordinate_mismatch = float(np.max(np.abs(X_M - X_P)))
        max_face_conormal_mismatch = float(np.max(np.abs(conormal_M + conormal_neighbor)))
        max_face_jacobian_mismatch = float(np.max(np.abs(Jf_M - Jf_P)))

        direct_abs = np.abs(eps_direct)
        direct_node = int(np.argmax(direct_abs))
        direct_value = float(direct_abs[direct_node])

        if direct_value > float(worst_direct["abs_direct_mismatch"]):
            worst_direct = {
                "abs_direct_mismatch": direct_value,
                "element": int(element),
                "face": int(face),
                "neighbor_element": int(neighbor_element),
                "neighbor_face": int(neighbor_face),
                "face_node_index": direct_node,
                "a_direct_M": float(adM[direct_node]),
                "a_direct_P": float(adP[direct_node]),
                "a_projected_M": float(apM[direct_node]),
                "a_projected_P": float(apP[direct_node]),
                "X_face_M": X_M[direct_node].tolist(),
                "X_face_P": X_P[direct_node].tolist(),
                "face_jacobian_M": float(Jf_M[direct_node]),
                "face_jacobian_P": float(Jf_P[direct_node]),
                "face_conormal_M": conormal_M[direct_node].tolist(),
                "face_conormal_P": conormal_neighbor[direct_node].tolist(),
            }

        face_rows.append(
            {
                "ndivs": int(ndivs),
                "element": int(element),
                "face": int(face),
                "neighbor_element": int(neighbor_element),
                "neighbor_face": int(neighbor_face),
                "n_face_points": int(trace.n_face_points),
                "max_abs_projected_M": max_abs_projected_M,
                "max_abs_projected_P": max_abs_projected_P,
                "max_abs_projected_mismatch": max_abs_projected_mismatch,
                "weighted_l2_projected_mismatch": weighted_l2(eps_projected, weights),
                "relative_projected_mismatch": relative(
                    max_abs_projected_mismatch,
                    max_abs_projected_M,
                    max_abs_projected_P,
                ),
                "max_abs_direct_M": max_abs_direct_M,
                "max_abs_direct_P": max_abs_direct_P,
                "max_abs_direct_mismatch": max_abs_direct_mismatch,
                "weighted_l2_direct_mismatch": weighted_l2(eps_direct, weights),
                "relative_direct_mismatch": relative(
                    max_abs_direct_mismatch,
                    max_abs_direct_M,
                    max_abs_direct_P,
                ),
                "max_abs_projected_minus_direct_M": max_abs_projected_minus_direct_M,
                "max_abs_projected_minus_direct_P": max_abs_projected_minus_direct_P,
                "relative_projected_direct_difference": relative(
                    max_abs_projected_direct,
                    max_abs_projected_M,
                    max_abs_projected_P,
                    max_abs_direct_M,
                    max_abs_direct_P,
                ),
                "max_abs_decomposition_error": float(np.max(np.abs(decomposition))),
                "max_face_coordinate_mismatch": max_face_coordinate_mismatch,
                "max_face_conormal_mismatch": max_face_conormal_mismatch,
                "max_face_jacobian_mismatch": max_face_jacobian_mismatch,
            }
        )

    max_abs_projected_M_global = float(
        max(
            max(row["max_abs_projected_M"], row["max_abs_projected_P"])
            for row in face_rows
        )
    )
    max_abs_direct_M_global = float(
        max(max(row["max_abs_direct_M"], row["max_abs_direct_P"]) for row in face_rows)
    )
    max_abs_projected_minus_direct_M_global = float(
        max(
            max(
                row["max_abs_projected_minus_direct_M"],
                row["max_abs_projected_minus_direct_P"],
            )
            for row in face_rows
        )
    )

    summary = {
        "order": int(order),
        "ndivs": int(ndivs),
        "n_elements": int(objects.mesh.elements.shape[0]),
        "n_points_per_element": int(objects.ref.rs.shape[0]),
        "total_dofs": int(objects.mesh.elements.shape[0] * objects.ref.rs.shape[0]),
        "hmin": float(objects.hmin),
        "max_abs_projected_M": max_abs_projected_M_global,
        "max_abs_direct_M": max_abs_direct_M_global,
        "max_abs_projected_minus_direct_M": max_abs_projected_minus_direct_M_global,
        "relative_projected_direct_M_difference": relative(
            max_abs_projected_minus_direct_M_global,
            max_abs_projected_M_global,
            max_abs_direct_M_global,
        ),
        "max_projected_mismatch": float(
            max(row["max_abs_projected_mismatch"] for row in face_rows)
        ),
        "global_projected_l2": float(
            np.sqrt(sum(row["weighted_l2_projected_mismatch"] ** 2 for row in face_rows))
        ),
        "max_relative_projected_mismatch": float(
            max(row["relative_projected_mismatch"] for row in face_rows)
        ),
        "max_direct_mismatch": float(
            max(row["max_abs_direct_mismatch"] for row in face_rows)
        ),
        "global_direct_l2": float(
            np.sqrt(sum(row["weighted_l2_direct_mismatch"] ** 2 for row in face_rows))
        ),
        "max_relative_direct_mismatch": float(
            max(row["relative_direct_mismatch"] for row in face_rows)
        ),
        "max_projected_direct_difference": max_abs_projected_minus_direct_M_global,
        "max_relative_projected_direct_difference": float(
            max(row["relative_projected_direct_difference"] for row in face_rows)
        ),
        "max_decomposition_error": float(
            max(row["max_abs_decomposition_error"] for row in face_rows)
        ),
        "max_face_coordinate_mismatch": float(
            max(row["max_face_coordinate_mismatch"] for row in face_rows)
        ),
        "max_face_conormal_mismatch": float(
            max(row["max_face_conormal_mismatch"] for row in face_rows)
        ),
        "max_face_jacobian_mismatch": float(
            max(row["max_face_jacobian_mismatch"] for row in face_rows)
        ),
        "projected_mismatch_rate": float("nan"),
        "direct_mismatch_rate": float("nan"),
        "projected_direct_difference_rate": float("nan"),
        "direct_coefficient_formula": DIRECT_COEFFICIENT_FORMULA,
        "direct_coefficient_interpretation": FACE_CONORMAL_INTERPRETATION,
        "classification": "",
        "face_csv": face_csv_name,
    }

    return summary, face_rows, worst_direct


def attach_rates(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: float(row["hmin"]), reverse=True)
    previous: dict[str, Any] | None = None

    for row in rows:
        if previous is not None:
            row["projected_mismatch_rate"] = observed_rate(
                float(previous["max_projected_mismatch"]),
                float(row["max_projected_mismatch"]),
                float(previous["hmin"]),
                float(row["hmin"]),
            )
            row["direct_mismatch_rate"] = observed_rate(
                float(previous["max_direct_mismatch"]),
                float(row["max_direct_mismatch"]),
                float(previous["hmin"]),
                float(row["hmin"]),
            )
            row["projected_direct_difference_rate"] = observed_rate(
                float(previous["max_projected_direct_difference"]),
                float(row["max_projected_direct_difference"]),
                float(previous["hmin"]),
                float(row["hmin"]),
            )

        previous = row


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        return [json_safe(val) for val in value]

    if isinstance(value, tuple):
        return [json_safe(val) for val in value]

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_console_table(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        "ndiv | max proj mismatch | max direct mismatch | "
        "max proj-direct | decomposition err"
    )
    print("-----+-------------------+---------------------+-----------------+------------------")

    for row in sorted(rows, key=lambda item: int(item["ndivs"])):
        print(
            f"{int(row['ndivs']):>4d} | "
            f"{float(row['max_projected_mismatch']): .6e} | "
            f"{float(row['max_direct_mismatch']): .6e} | "
            f"{float(row['max_projected_direct_difference']): .6e} | "
            f"{float(row['max_decomposition_error']): .6e}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare projected and direct interface line velocity compatibility."
    )
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--ndivs", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--radius", type=parse_float_expr, default=1.0)
    parser.add_argument("--alpha0", type=parse_float_expr, default=-np.pi / 4.0)
    parser.add_argument("--u0", type=parse_float_expr, default=2.0 * np.pi / 10.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/projected_vs_direct_interface_metric",
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
    output_dir = ROOT / args.output_dir

    print("Projected vs direct interface metric diagnostic")
    print("-----------------------------------------------")
    print(f"table         : {TABLE}")
    print(f"sbp_variant   : {SBP_VARIANT}")
    print(f"volume_form   : {VOLUME_FORM}")
    print(f"flux_type     : {FLUX_TYPE}")
    print(f"order         : {args.order}")
    print(f"ndivs         : {args.ndivs}")
    print(f"radius        : {args.radius}")
    print(f"alpha0        : {args.alpha0}")
    print(f"u0            : {args.u0}")
    print(f"direct formula: {DIRECT_COEFFICIENT_FORMULA}")
    print(f"output_dir    : {output_dir}")
    print()

    summary_rows: list[dict[str, Any]] = []
    worst_interfaces: dict[str, Any] = {}
    face_files: list[str] = []

    for ndivs in args.ndivs:
        objects = build_discrete_objects(
            order=int(args.order),
            ndivs=int(ndivs),
            radius=float(args.radius),
            alpha0=float(args.alpha0),
            u0=float(args.u0),
        )
        face_csv_name = f"faces_ndiv{ndivs}.csv"
        summary, face_rows, worst_direct = diagnose_ndiv(
            objects=objects,
            order=int(args.order),
            ndivs=int(ndivs),
            face_csv_name=face_csv_name,
        )

        write_csv(output_dir / face_csv_name, face_rows, FACE_FIELDS)
        face_files.append(face_csv_name)
        summary_rows.append(summary)
        worst_interfaces[str(ndivs)] = worst_direct

        print(
            f"ndivs={ndivs}, "
            f"max_projected={summary['max_projected_mismatch']:.6e}, "
            f"max_direct={summary['max_direct_mismatch']:.6e}, "
            f"proj-direct={summary['max_projected_direct_difference']:.6e}, "
            f"decomp={summary['max_decomposition_error']:.6e}, "
            f"X={summary['max_face_coordinate_mismatch']:.6e}, "
            f"conormal={summary['max_face_conormal_mismatch']:.6e}, "
            f"Jf={summary['max_face_jacobian_mismatch']:.6e}"
        )

    attach_rates(summary_rows)
    classification = classify_rows(summary_rows)

    for row in summary_rows:
        row["classification"] = classification

    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_json(
        output_dir / "summary.json",
        {
            "git_commit": current_git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "table": TABLE,
                "sbp_variant": SBP_VARIANT,
                "volume_form": VOLUME_FORM,
                "flux_type": FLUX_TYPE,
                "order": int(args.order),
                "ndivs": [int(value) for value in args.ndivs],
                "radius": float(args.radius),
                "alpha0": float(args.alpha0),
                "u0": float(args.u0),
                "omega": [
                    float(args.u0) * component
                    for component in rotation_axis_from_alpha0(float(args.alpha0))
                ],
                "use_numba": False,
                "roundoff_tolerance": ROUND_OFF_TOL,
            },
            "direct_coefficient_scaling_interpretation": DIRECT_COEFFICIENT_REASON,
            "face_conormal_interpretation": FACE_CONORMAL_INTERPRETATION,
            "face_conormal_source": (
                "src/simplex_dg/geometry/sphere.py::map_reference_face_to_sphere_element "
                "and validate_geometry_cache"
            ),
            "normal_velocity_source": (
                "src/simplex_dg/rhs/surface.py::build_surface_rhs_cache"
            ),
            "direct_coefficient_formula": DIRECT_COEFFICIENT_FORMULA,
            "final_classification": classification,
            "summary": summary_rows,
            "worst_direct_interfaces": worst_interfaces,
            "face_files": face_files,
        },
    )

    print_console_table(summary_rows)
    print()
    print(f"classification       : {classification}")
    print(f"summary CSV written  : {output_dir / 'summary.csv'}")
    print(f"summary JSON written : {output_dir / 'summary.json'}")
    print(f"face CSV files written: {len(face_files)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
