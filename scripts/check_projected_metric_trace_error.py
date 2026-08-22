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
from simplex_dg.rhs.surface import _FACE_DRDT, _FACE_DSDT
from simplex_dg.time import minimum_face_length
from simplex_dg.trace import build_trace_cache, gather_neighbor_traces


TABLE = "table1"
SBP_VARIANT = "projected"
VOLUME_FORM = "split"
FLUX_TYPE = "central"
ROUND_OFF_TOL = 1.0e-12
RELATIVE_EPS = 1.0e-30
DIRECT_COEFFICIENT_FORMULA = "geom.face_jacobian * full.surface.normal_velocity"
FINAL_CLASSIFICATION = "Projection-induced interface metric incompatibility confirmed"


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


SINGLE_SIDE_FIELDS = [
    "order",
    "ndivs",
    "element",
    "face",
    "reference_face_id",
    "face_node",
    "R_alpha",
    "R_beta",
    "term_alpha",
    "term_beta",
    "a_projected",
    "a_direct",
    "delta",
    "abs_delta",
    "relative_delta",
    "delta_alpha_trace",
    "delta_beta_trace",
]


INTERFACE_FIELDS = [
    "order",
    "ndivs",
    "element",
    "face",
    "reference_face_id",
    "neighbor_element",
    "neighbor_face",
    "neighbor_reference_face_id",
    "max_abs_projected_mismatch",
    "max_abs_direct_mismatch",
    "max_abs_delta_M",
    "max_abs_delta_P",
    "max_abs_delta_M_plus_delta_P",
    "max_abs_projected_minus_delta_sum",
    "max_abs_decomposition_error",
]


FACE_PAIR_FIELDS = [
    "order",
    "ndivs",
    "face_pair",
    "face",
    "neighbor_face",
    "reference_face_id",
    "neighbor_reference_face_id",
    "count",
    "max_projected_mismatch",
    "mean_abs_projected_mismatch",
    "weighted_l2_projected_mismatch",
    "max_abs_delta_M",
    "max_abs_delta_P",
]


SUMMARY_FIELDS = [
    "order",
    "ndivs",
    "n_elements",
    "n_points_per_element",
    "total_dofs",
    "hmin",
    "max_projected_reconstruction_error",
    "max_face1_combined_error",
    "l2_face1_combined_error",
    "relative_face1_combined_error",
    "face1_rate",
    "max_face2_alpha_error",
    "l2_face2_alpha_error",
    "relative_face2_alpha_error",
    "face2_alpha_rate",
    "max_face3_beta_error",
    "l2_face3_beta_error",
    "relative_face3_beta_error",
    "face3_beta_rate",
    "max_single_side_projected_direct_difference",
    "single_side_projected_direct_difference_rate",
    "max_projected_interface_mismatch",
    "max_direct_interface_mismatch",
    "max_delta_sum",
    "max_projected_minus_delta_sum",
    "max_interface_decomposition_error",
    "projected_interface_mismatch_rate",
    "dominant_face_error_type",
    "empirical_convergence_interpretation",
    "classification",
    "single_side_csv",
    "interface_csv",
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


def validate_positive_increasing(values: list[int], name: str) -> list[int]:
    if not values:
        raise ValueError(f"{name} must not be empty.")

    out = [int(value) for value in values]

    if any(value < 1 for value in out):
        raise ValueError(f"{name} must contain only positive integers.")

    if len(set(out)) != len(out):
        raise ValueError(f"{name} must be unique.")

    if any(curr <= prev for prev, curr in zip(out, out[1:])):
        raise ValueError(f"{name} must be strictly increasing.")

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


def direct_line_coefficient(objects: DiscreteObjects) -> np.ndarray:
    return np.asarray(objects.geom.face_jacobian, dtype=float) * np.asarray(
        objects.full.surface.normal_velocity,
        dtype=float,
    )


def reconstruct_projected_coefficient(
    objects: DiscreteObjects,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trace = objects.trace
    volume = objects.full.volume
    alpha = np.asarray(volume.alpha, dtype=float)
    beta = np.asarray(volume.beta, dtype=float)

    expected = (trace.n_elements, trace.n_faces, trace.n_face_points)
    alpha_R = np.empty(expected, dtype=float)
    beta_R = np.empty(expected, dtype=float)
    term_alpha = np.empty(expected, dtype=float)
    term_beta = np.empty(expected, dtype=float)
    manual = np.empty(expected, dtype=float)

    for f in range(trace.n_faces):
        alpha_R[:, f, :] = alpha @ trace.face_interp[f].T
        beta_R[:, f, :] = beta @ trace.face_interp[f].T

        # Production surface.py defines a = dsdt * R(alpha) - drdt * R(beta).
        dsdt = float(_FACE_DSDT[f])
        drdt = float(_FACE_DRDT[f])
        term_alpha[:, f, :] = dsdt * alpha_R[:, f, :]
        term_beta[:, f, :] = -drdt * beta_R[:, f, :]
        manual[:, f, :] = term_alpha[:, f, :] + term_beta[:, f, :]

    existing = projected_line_velocity(objects.full.volume, objects.full.surface)
    return manual, existing, alpha_R, beta_R, term_alpha, term_beta


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


def weighted_l2_face(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights[None, :] * values * values)))


def weighted_l2_one(values: np.ndarray, weights: np.ndarray) -> float:
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
    if not all(np.isfinite(value) for value in (old_error, new_error, old_h, new_h)):
        return float("nan")

    if old_error <= roundoff_tol or new_error <= roundoff_tol:
        return float("nan")

    if old_h <= 0.0 or new_h <= 0.0 or np.isclose(old_h, new_h):
        return float("nan")

    return float(np.log(old_error / new_error) / np.log(old_h / new_h))


def classify_rate_sequence(rates: list[float]) -> str:
    finite = [float(rate) for rate in rates if np.isfinite(rate)]

    if not finite:
        return "no clear rate"

    tail = finite[-2:] if len(finite) >= 2 else finite
    mean_tail = float(np.mean(tail))

    if 4.5 <= mean_tail <= 5.5 and all(3.75 <= rate <= 6.25 for rate in tail):
        return "approximately fifth order"

    if 3.5 <= mean_tail < 4.5 and all(2.75 <= rate <= 5.25 for rate in tail):
        return "approximately fourth order"

    if len(finite) >= 2:
        return "mixed/pre-asymptotic"

    return "no clear rate"


def single_side_rows(
    *,
    order: int,
    ndivs: int,
    objects: DiscreteObjects,
    alpha_R: np.ndarray,
    beta_R: np.ndarray,
    term_alpha: np.ndarray,
    term_beta: np.ndarray,
    a_projected: np.ndarray,
    a_direct: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delta = a_projected - a_direct

    for element in range(objects.trace.n_elements):
        for face in range(objects.trace.n_faces):
            for node in range(objects.trace.n_face_points):
                value_projected = float(a_projected[element, face, node])
                value_direct = float(a_direct[element, face, node])
                value_delta = float(delta[element, face, node])
                row = {
                    "order": int(order),
                    "ndivs": int(ndivs),
                    "element": int(element),
                    "face": int(face),
                    "reference_face_id": int(face + 1),
                    "face_node": int(node),
                    "R_alpha": float(alpha_R[element, face, node]),
                    "R_beta": float(beta_R[element, face, node]),
                    "term_alpha": float(term_alpha[element, face, node]),
                    "term_beta": float(term_beta[element, face, node]),
                    "a_projected": value_projected,
                    "a_direct": value_direct,
                    "delta": value_delta,
                    "abs_delta": abs(value_delta),
                    "relative_delta": relative(
                        abs(value_delta),
                        value_projected,
                        value_direct,
                    ),
                    "delta_alpha_trace": float("nan"),
                    "delta_beta_trace": float("nan"),
                }

                if face == 1:
                    row["delta_alpha_trace"] = value_delta
                elif face == 2:
                    row["delta_beta_trace"] = value_delta

                rows.append(row)

    return rows


def interface_diagnostics(
    *,
    order: int,
    ndivs: int,
    objects: DiscreteObjects,
    a_projected: np.ndarray,
    a_direct: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace = objects.trace
    delta = a_projected - a_direct
    a_projected_P = gather_neighbor_traces(
        a_projected,
        trace,
        boundary_value=np.nan,
        use_numba=False,
    )
    a_direct_P = gather_neighbor_traces(
        a_direct,
        trace,
        boundary_value=np.nan,
        use_numba=False,
    )

    rows: list[dict[str, Any]] = []
    pair_accum: dict[tuple[int, int], dict[str, Any]] = {}

    for element, face, neighbor_element, neighbor_face in unique_internal_faces(trace):
        weights = np.asarray(trace.face_weights[face], dtype=float)
        apM = np.asarray(a_projected[element, face], dtype=float)
        adM = np.asarray(a_direct[element, face], dtype=float)
        apP = np.asarray(a_projected_P[element, face], dtype=float)
        adP = np.asarray(a_direct_P[element, face], dtype=float)

        delta_M = apM - adM
        delta_P = apP - adP
        epsilon_projected = apM + apP
        epsilon_direct = adM + adP
        delta_sum = delta_M + delta_P
        projected_minus_delta_sum = epsilon_projected - delta_sum
        decomposition = epsilon_projected - (epsilon_direct + delta_sum)

        rows.append(
            {
                "order": int(order),
                "ndivs": int(ndivs),
                "element": int(element),
                "face": int(face),
                "reference_face_id": int(face + 1),
                "neighbor_element": int(neighbor_element),
                "neighbor_face": int(neighbor_face),
                "neighbor_reference_face_id": int(neighbor_face + 1),
                "max_abs_projected_mismatch": float(np.max(np.abs(epsilon_projected))),
                "max_abs_direct_mismatch": float(np.max(np.abs(epsilon_direct))),
                "max_abs_delta_M": float(np.max(np.abs(delta_M))),
                "max_abs_delta_P": float(np.max(np.abs(delta_P))),
                "max_abs_delta_M_plus_delta_P": float(np.max(np.abs(delta_sum))),
                "max_abs_projected_minus_delta_sum": float(
                    np.max(np.abs(projected_minus_delta_sum))
                ),
                "max_abs_decomposition_error": float(np.max(np.abs(decomposition))),
            }
        )

        key = (int(face), int(neighbor_face))
        if key not in pair_accum:
            pair_accum[key] = {
                "order": int(order),
                "ndivs": int(ndivs),
                "face_pair": f"{face + 1}<->{neighbor_face + 1}",
                "face": int(face),
                "neighbor_face": int(neighbor_face),
                "reference_face_id": int(face + 1),
                "neighbor_reference_face_id": int(neighbor_face + 1),
                "count": 0,
                "_sum_abs_projected": 0.0,
                "_n_values": 0,
                "_sum_w_projected_sq": 0.0,
                "max_projected_mismatch": 0.0,
                "max_abs_delta_M": 0.0,
                "max_abs_delta_P": 0.0,
            }

        acc = pair_accum[key]
        acc["count"] += 1
        acc["_sum_abs_projected"] += float(np.sum(np.abs(epsilon_projected)))
        acc["_n_values"] += int(epsilon_projected.size)
        acc["_sum_w_projected_sq"] += float(np.sum(weights * epsilon_projected * epsilon_projected))
        acc["max_projected_mismatch"] = max(
            float(acc["max_projected_mismatch"]),
            float(np.max(np.abs(epsilon_projected))),
        )
        acc["max_abs_delta_M"] = max(
            float(acc["max_abs_delta_M"]),
            float(np.max(np.abs(delta_M))),
        )
        acc["max_abs_delta_P"] = max(
            float(acc["max_abs_delta_P"]),
            float(np.max(np.abs(delta_P))),
        )

    pair_rows: list[dict[str, Any]] = []
    for acc in sorted(pair_accum.values(), key=lambda row: (row["face"], row["neighbor_face"])):
        out = dict(acc)
        out["mean_abs_projected_mismatch"] = (
            float(out["_sum_abs_projected"]) / max(int(out["_n_values"]), 1)
        )
        out["weighted_l2_projected_mismatch"] = float(np.sqrt(float(out["_sum_w_projected_sq"])))
        out.pop("_sum_abs_projected")
        out.pop("_n_values")
        out.pop("_sum_w_projected_sq")
        pair_rows.append(out)

    return rows, pair_rows


def summarize_ndiv(
    *,
    order: int,
    ndivs: int,
    objects: DiscreteObjects,
    reconstruction_error: float,
    a_projected: np.ndarray,
    a_direct: np.ndarray,
    interface_rows: list[dict[str, Any]],
    single_side_csv: str,
    interface_csv: str,
) -> dict[str, Any]:
    delta = a_projected - a_direct
    face_errors: dict[int, dict[str, float]] = {}

    for face in range(objects.trace.n_faces):
        weights = np.asarray(objects.trace.face_weights[face], dtype=float)
        delta_face = delta[:, face, :]
        max_error = float(np.max(np.abs(delta_face)))
        l2_error = weighted_l2_face(delta_face, weights)
        max_projected = float(np.max(np.abs(a_projected[:, face, :])))
        max_direct = float(np.max(np.abs(a_direct[:, face, :])))
        face_errors[face] = {
            "max": max_error,
            "l2": l2_error,
            "relative": relative(max_error, max_projected, max_direct),
        }

    max_values = {
        "face1_combined": face_errors[0]["max"],
        "face2_alpha": face_errors[1]["max"],
        "face3_beta": face_errors[2]["max"],
    }
    dominant_face_error_type = max(max_values, key=max_values.get)

    max_direct_interface = float(
        max(row["max_abs_direct_mismatch"] for row in interface_rows)
    )
    max_projected_interface = float(
        max(row["max_abs_projected_mismatch"] for row in interface_rows)
    )
    max_delta_sum = float(
        max(row["max_abs_delta_M_plus_delta_P"] for row in interface_rows)
    )
    max_projected_minus_delta_sum = float(
        max(row["max_abs_projected_minus_delta_sum"] for row in interface_rows)
    )
    max_decomposition = float(
        max(row["max_abs_decomposition_error"] for row in interface_rows)
    )

    return {
        "order": int(order),
        "ndivs": int(ndivs),
        "n_elements": int(objects.mesh.elements.shape[0]),
        "n_points_per_element": int(objects.ref.rs.shape[0]),
        "total_dofs": int(objects.mesh.elements.shape[0] * objects.ref.rs.shape[0]),
        "hmin": float(objects.hmin),
        "max_projected_reconstruction_error": float(reconstruction_error),
        "max_face1_combined_error": face_errors[0]["max"],
        "l2_face1_combined_error": face_errors[0]["l2"],
        "relative_face1_combined_error": face_errors[0]["relative"],
        "face1_rate": float("nan"),
        "max_face2_alpha_error": face_errors[1]["max"],
        "l2_face2_alpha_error": face_errors[1]["l2"],
        "relative_face2_alpha_error": face_errors[1]["relative"],
        "face2_alpha_rate": float("nan"),
        "max_face3_beta_error": face_errors[2]["max"],
        "l2_face3_beta_error": face_errors[2]["l2"],
        "relative_face3_beta_error": face_errors[2]["relative"],
        "face3_beta_rate": float("nan"),
        "max_single_side_projected_direct_difference": float(np.max(np.abs(delta))),
        "single_side_projected_direct_difference_rate": float("nan"),
        "max_projected_interface_mismatch": max_projected_interface,
        "max_direct_interface_mismatch": max_direct_interface,
        "max_delta_sum": max_delta_sum,
        "max_projected_minus_delta_sum": max_projected_minus_delta_sum,
        "max_interface_decomposition_error": max_decomposition,
        "projected_interface_mismatch_rate": float("nan"),
        "dominant_face_error_type": dominant_face_error_type,
        "empirical_convergence_interpretation": "",
        "classification": FINAL_CLASSIFICATION,
        "single_side_csv": single_side_csv,
        "interface_csv": interface_csv,
    }


def attach_rates(rows: list[dict[str, Any]]) -> None:
    by_order: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_order.setdefault(int(row["order"]), []).append(row)

    for order_rows in by_order.values():
        order_rows.sort(key=lambda row: float(row["hmin"]), reverse=True)
        previous: dict[str, Any] | None = None

        for row in order_rows:
            if previous is not None:
                row["face1_rate"] = observed_rate(
                    float(previous["max_face1_combined_error"]),
                    float(row["max_face1_combined_error"]),
                    float(previous["hmin"]),
                    float(row["hmin"]),
                )
                row["face2_alpha_rate"] = observed_rate(
                    float(previous["max_face2_alpha_error"]),
                    float(row["max_face2_alpha_error"]),
                    float(previous["hmin"]),
                    float(row["hmin"]),
                )
                row["face3_beta_rate"] = observed_rate(
                    float(previous["max_face3_beta_error"]),
                    float(row["max_face3_beta_error"]),
                    float(previous["hmin"]),
                    float(row["hmin"]),
                )
                row["single_side_projected_direct_difference_rate"] = observed_rate(
                    float(previous["max_single_side_projected_direct_difference"]),
                    float(row["max_single_side_projected_direct_difference"]),
                    float(previous["hmin"]),
                    float(row["hmin"]),
                )
                row["projected_interface_mismatch_rate"] = observed_rate(
                    float(previous["max_projected_interface_mismatch"]),
                    float(row["max_projected_interface_mismatch"]),
                    float(previous["hmin"]),
                    float(row["hmin"]),
                )

            previous = row

        rates = {
            "face1": [float(row["face1_rate"]) for row in order_rows],
            "face2_alpha": [float(row["face2_alpha_rate"]) for row in order_rows],
            "face3_beta": [float(row["face3_beta_rate"]) for row in order_rows],
            "interface": [float(row["projected_interface_mismatch_rate"]) for row in order_rows],
        }
        interpretation = {
            key: classify_rate_sequence(value)
            for key, value in rates.items()
        }
        interpretation_text = ", ".join(f"{key}: {value}" for key, value in interpretation.items())

        for row in order_rows:
            row["empirical_convergence_interpretation"] = interpretation_text


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


def output_names(*, order: int, ndivs: int, n_orders: int) -> tuple[str, str]:
    if n_orders == 1:
        return f"single_side_ndiv{ndivs}.csv", f"interfaces_ndiv{ndivs}.csv"

    return (
        f"single_side_order{order}_ndiv{ndivs}.csv",
        f"interfaces_order{order}_ndiv{ndivs}.csv",
    )


def print_console_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        "order ndiv | face1 err   | face2 alpha | face3 beta  | "
        "proj interface | direct mismatch"
    )
    print("-----------+-------------+-------------+-------------+----------------+----------------")

    for row in sorted(rows, key=lambda item: (int(item["order"]), int(item["ndivs"]))):
        print(
            f"{int(row['order']):>5d} {int(row['ndivs']):>4d} | "
            f"{float(row['max_face1_combined_error']): .4e} | "
            f"{float(row['max_face2_alpha_error']): .4e} | "
            f"{float(row['max_face3_beta_error']): .4e} | "
            f"{float(row['max_projected_interface_mismatch']): .4e} | "
            f"{float(row['max_direct_interface_mismatch']): .4e}"
        )

    print()
    print("order ndiv | face1 rate | alpha rate | beta rate | interface rate")
    print("-----------+------------+------------+-----------+---------------")

    for row in sorted(rows, key=lambda item: (int(item["order"]), int(item["ndivs"]))):
        def fmt(value: Any) -> str:
            value_f = float(value)
            return "nan" if not np.isfinite(value_f) else f"{value_f:.3f}"

        print(
            f"{int(row['order']):>5d} {int(row['ndivs']):>4d} | "
            f"{fmt(row['face1_rate']):>10s} | "
            f"{fmt(row['face2_alpha_rate']):>10s} | "
            f"{fmt(row['face3_beta_rate']):>9s} | "
            f"{fmt(row['projected_interface_mismatch_rate']):>13s}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose projected metric trace error against direct surface coefficient."
    )
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument(
        "--orders",
        nargs="+",
        type=int,
        default=None,
        help="Optional order sweep. If omitted, --order is used.",
    )
    parser.add_argument("--ndivs", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--radius", type=parse_float_expr, default=1.0)
    parser.add_argument("--alpha0", type=parse_float_expr, default=-np.pi / 4.0)
    parser.add_argument("--u0", type=parse_float_expr, default=2.0 * np.pi / 10.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/projected_metric_trace_error",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(_normalize_expr_option_args(list(argv)))

    try:
        args.ndivs = validate_positive_increasing(args.ndivs, "ndivs")
    except ValueError as exc:
        parser.error(str(exc))

    if args.orders is None:
        args.orders = [int(args.order)]
    else:
        try:
            args.orders = validate_positive_increasing(args.orders, "orders")
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
    n_orders = len(args.orders)

    print("Projected metric trace error diagnostic")
    print("---------------------------------------")
    print(f"table         : {TABLE}")
    print(f"sbp_variant   : {SBP_VARIANT}")
    print(f"volume_form   : {VOLUME_FORM}")
    print(f"flux_type     : {FLUX_TYPE}")
    print(f"orders        : {args.orders}")
    print(f"ndivs         : {args.ndivs}")
    print(f"radius        : {args.radius}")
    print(f"alpha0        : {args.alpha0}")
    print(f"u0            : {args.u0}")
    print(f"direct formula: {DIRECT_COEFFICIENT_FORMULA}")
    print(f"output_dir    : {output_dir}")
    print()

    summary_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    output_files: list[str] = []

    for order in args.orders:
        for ndivs in args.ndivs:
            objects = build_discrete_objects(
                order=int(order),
                ndivs=int(ndivs),
                radius=float(args.radius),
                alpha0=float(args.alpha0),
                u0=float(args.u0),
            )

            manual_projected, existing_projected, alpha_R, beta_R, term_alpha, term_beta = (
                reconstruct_projected_coefficient(objects)
            )
            reconstruction_error = float(np.max(np.abs(manual_projected - existing_projected)))
            a_direct = direct_line_coefficient(objects)
            delta = existing_projected - a_direct

            single_name, interface_name = output_names(
                order=int(order),
                ndivs=int(ndivs),
                n_orders=n_orders,
            )
            single_rows = single_side_rows(
                order=int(order),
                ndivs=int(ndivs),
                objects=objects,
                alpha_R=alpha_R,
                beta_R=beta_R,
                term_alpha=term_alpha,
                term_beta=term_beta,
                a_projected=existing_projected,
                a_direct=a_direct,
            )
            interface_rows, pair_rows = interface_diagnostics(
                order=int(order),
                ndivs=int(ndivs),
                objects=objects,
                a_projected=existing_projected,
                a_direct=a_direct,
            )
            summary = summarize_ndiv(
                order=int(order),
                ndivs=int(ndivs),
                objects=objects,
                reconstruction_error=reconstruction_error,
                a_projected=existing_projected,
                a_direct=a_direct,
                interface_rows=interface_rows,
                single_side_csv=single_name,
                interface_csv=interface_name,
            )

            write_csv(output_dir / single_name, single_rows, SINGLE_SIDE_FIELDS)
            write_csv(output_dir / interface_name, interface_rows, INTERFACE_FIELDS)
            summary_rows.append(summary)
            all_pair_rows.extend(pair_rows)
            output_files.extend([single_name, interface_name])

            print(
                f"order={order}, ndivs={ndivs}, "
                f"recon={reconstruction_error:.3e}, "
                f"face1={summary['max_face1_combined_error']:.6e}, "
                f"face2_alpha={summary['max_face2_alpha_error']:.6e}, "
                f"face3_beta={summary['max_face3_beta_error']:.6e}, "
                f"single={np.max(np.abs(delta)):.6e}, "
                f"interface={summary['max_projected_interface_mismatch']:.6e}, "
                f"direct={summary['max_direct_interface_mismatch']:.6e}, "
                f"decomp={summary['max_interface_decomposition_error']:.3e}"
            )

    attach_rates(summary_rows)
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "face_pair_summary.csv", all_pair_rows, FACE_PAIR_FIELDS)
    write_json(
        output_dir / "summary.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(),
            "configuration": {
                "table": TABLE,
                "sbp_variant": SBP_VARIANT,
                "volume_form": VOLUME_FORM,
                "flux_type": FLUX_TYPE,
                "orders": [int(order) for order in args.orders],
                "ndivs": [int(ndiv) for ndiv in args.ndivs],
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
            "direct_coefficient_formula": DIRECT_COEFFICIENT_FORMULA,
            "face_formulas": {
                "face1": "a_projected = 2*R(alpha) + 2*R(beta)",
                "face2": "a_projected = -2*R(alpha)",
                "face3": "a_projected = -2*R(beta)",
                "production_constants": {
                    "FACE_DRDT": [float(value) for value in _FACE_DRDT],
                    "FACE_DSDT": [float(value) for value in _FACE_DSDT],
                },
            },
            "summary": summary_rows,
            "face_pair_summary": all_pair_rows,
            "empirical_convergence_interpretation": {
                str(order): next(
                    (
                        row["empirical_convergence_interpretation"]
                        for row in summary_rows
                        if int(row["order"]) == int(order)
                    ),
                    "",
                )
                for order in args.orders
            },
            "dominant_face_error_type": {
                f"order{row['order']}_ndiv{row['ndivs']}": row["dominant_face_error_type"]
                for row in summary_rows
            },
            "direct_geometry_remains_roundoff_compatible": all(
                float(row["max_direct_interface_mismatch"]) <= ROUND_OFF_TOL
                for row in summary_rows
            ),
            "projected_mismatch_explained_by_single_side_projection_errors": all(
                float(row["max_interface_decomposition_error"]) <= ROUND_OFF_TOL
                for row in summary_rows
            ),
            "classification": FINAL_CLASSIFICATION,
            "output_files": output_files + ["summary.csv", "summary.json", "face_pair_summary.csv"],
        },
    )

    print_console_summary(summary_rows)
    print()
    print(f"classification          : {FINAL_CLASSIFICATION}")
    print(f"summary CSV written     : {output_dir / 'summary.csv'}")
    print(f"summary JSON written    : {output_dir / 'summary.json'}")
    print(f"face-pair CSV written   : {output_dir / 'face_pair_summary.csv'}")
    print(f"single/interface files  : {len(output_files)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
