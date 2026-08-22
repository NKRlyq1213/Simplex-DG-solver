from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.problems import gaussian_on_sphere
from simplex_dg.reference import build_reference_cache, vandermonde2d
from simplex_dg.rhs import (
    build_surface_rhs_cache,
    build_volume_rhs_cache,
    common_projected_line_velocity,
    projected_interior_line_flux,
    projected_line_velocity,
)
from simplex_dg.rhs.surface import _FACE_DRDT, _FACE_DSDT
from simplex_dg.trace import build_trace_cache, evaluate_face_traces


DEFAULT_TABLE = "table2"
DEFAULT_SBP_VARIANT = "projected"
DEFAULT_ORDER = 4
DEFAULT_NDIVS = (4, 8, 16, 32, 64)
DEFAULT_OUTPUT_DIR = "results/interface_normal_orientation"
DEFAULT_RADIUS = 1.0
DEFAULT_SIGMA = 0.35
DEFAULT_AMPLITUDE = 1.0
DEFAULT_RANDOM_SEED = 20260812
RELATIVE_EPS = 1.0e-300


MATRIX_FIELDNAMES = [
    "table",
    "sbp_variant",
    "order",
    "face",
    "projection_linf",
    "projection_frobenius",
    "Rgamma_minus_VgammaP_linf",
    "Rgamma_minus_VgammaP_frobenius",
]

INTERFACE_FIELDNAMES = [
    "ndiv",
    "state",
    "element_minus",
    "face_minus",
    "element_plus",
    "face_plus",
    "face_flip",
    "face_node",
    "x_match_error",
    "conormal_opposite_error",
    "face_jacobian_error",
    "face_jacobian_relative_error",
    "a_minus",
    "a_plus_outward",
    "a_plus_to_minus",
    "m_pm",
    "m_code",
    "m_difference",
    "Rq_minus",
    "Rq_plus",
    "Fq_minus",
    "Fq_plus_outward",
    "Fq_plus_to_minus",
    "epsilon_minus",
    "epsilon_plus",
    "literal_minus_normal_plus_value",
    "literal_vs_oriented_plus_difference",
    "m_old_literal_subtracted",
    "m_old_literal_subtracted_minus_m_pm",
    "m_literal_as_minus_orientation",
    "m_literal_as_minus_orientation_minus_m_pm",
    "epsilon_plus_wrong_outward_sign",
]

SUMMARY_FIELDNAMES = [
    "ndiv",
    "state",
    "num_interfaces",
    "max_Rgamma_minus_VgammaP",
    "max_face_point_error",
    "max_conormal_opposite_error",
    "max_face_jacobian_error",
    "max_face_jacobian_relative_error",
    "max_abs_a_minus_plus_a_plus",
    "max_abs_m_code_minus_m_pm",
    "max_abs_epsilon_minus",
    "max_abs_epsilon_plus",
    "max_literal_vs_oriented_plus_difference",
    "mean_literal_vs_oriented_plus_difference",
    "max_abs_m_old_literal_subtracted_minus_m_pm",
    "max_abs_m_literal_as_minus_orientation_minus_m_pm",
]


@dataclass(frozen=True)
class CaseObjects:
    ref: Any
    mesh: Any
    conn: Any
    geom: Any
    trace: Any
    volume: Any
    surface: Any


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
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


def manual_modal_projection(ref: Any) -> np.ndarray:
    V = np.asarray(ref.V, dtype=float)
    weights = np.asarray(ref.weights, dtype=float)
    mass = float(ref.area) * (V.T @ (weights[:, None] * V))
    rhs = float(ref.area) * (V.T * weights[None, :])
    return np.linalg.solve(mass, rhs)


def matrix_trace_checks(
    *,
    table: str,
    sbp_variant: str,
    orders: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for order in orders:
        ref = build_reference_cache(
            order=int(order),
            table=table,
            sbp_variant=sbp_variant,
            validate=True,
        )
        P_manual = manual_modal_projection(ref)
        P_diff = np.asarray(ref.projection, dtype=float) - P_manual

        for face_id in (1, 2, 3):
            edge = ref.edge_rules[face_id]
            V_face = vandermonde2d(ref.order, edge.rs[:, 0], edge.rs[:, 1])
            R_manual = V_face @ P_manual
            R_code = np.asarray(ref.face_interp[face_id], dtype=float)
            R_diff = R_code - R_manual
            rows.append(
                {
                    "table": table,
                    "sbp_variant": sbp_variant,
                    "order": int(order),
                    "face": int(face_id),
                    "projection_linf": float(np.max(np.abs(P_diff))),
                    "projection_frobenius": float(np.linalg.norm(P_diff, ord="fro")),
                    "Rgamma_minus_VgammaP_linf": float(np.max(np.abs(R_diff))),
                    "Rgamma_minus_VgammaP_frobenius": float(np.linalg.norm(R_diff, ord="fro")),
                }
            )

    return rows


def build_case(
    *,
    order: int,
    table: str,
    sbp_variant: str,
    ndiv: int,
    radius: float,
    omega: tuple[float, float, float],
) -> CaseObjects:
    ref = build_reference_cache(
        order=order,
        table=table,
        sbp_variant=sbp_variant,
        validate=True,
    )
    mesh = build_octa_sphere_mesh(ndivs=ndiv, radius=radius)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)
    volume = build_volume_rhs_cache(
        ref=ref,
        geom=geom,
        omega=omega,
        project_velocity=True,
        validate=True,
    )
    surface = build_surface_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=omega,
        flux_type="central",
        lf_alpha=1.0,
        project_velocity=True,
        validate=True,
    )
    return CaseObjects(ref=ref, mesh=mesh, conn=conn, geom=geom, trace=trace, volume=volume, surface=surface)


def nodal_project(values: np.ndarray, ref: Any) -> np.ndarray:
    projector = np.asarray(ref.V @ ref.projection, dtype=float)
    values = np.asarray(values, dtype=float)
    return values @ projector.T


def polynomial_state(ref: Any, n_elements: int) -> np.ndarray:
    n_modes = ref.V.shape[1]
    coeff = np.zeros(n_modes, dtype=float)

    for j in range(n_modes):
        sign = -1.0 if j % 2 else 1.0
        coeff[j] = sign / float(j + 2)

    coeff[0] += 1.0
    values = coeff @ ref.V.T
    return np.broadcast_to(values, (n_elements, ref.V.shape[0])).copy()


def random_modal_state(ref: Any, n_elements: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_modes = ref.V.shape[1]
    scale = 1.0 / np.sqrt(np.arange(1, n_modes + 1, dtype=float))
    coeffs = rng.standard_normal((n_elements, n_modes)) * scale[None, :]
    return coeffs @ ref.V.T


def build_states(
    *,
    objects: CaseObjects,
    radius: float,
    sigma: float,
    amplitude: float,
    random_seed: int,
) -> list[tuple[str, np.ndarray]]:
    geom = objects.geom
    ref = objects.ref
    K = objects.mesh.elements.shape[0]

    q_constant = np.ones((K, ref.rs.shape[0]), dtype=float)
    q_polynomial = polynomial_state(ref, K)
    q_gaussian_raw = gaussian_on_sphere(
        X=geom.X,
        center=(radius, 0.0, 0.0),
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
    )
    q_gaussian = nodal_project(q_gaussian_raw, ref)
    q_random = random_modal_state(ref, K, seed=random_seed)

    return [
        ("constant", q_constant),
        ("polynomial", q_polynomial),
        ("projected_gaussian", q_gaussian),
        ("random_modal", q_random),
    ]


def trace_alpha_beta_faces(objects: CaseObjects) -> tuple[np.ndarray, np.ndarray]:
    volume = objects.volume
    surface = objects.surface
    expected = (surface.n_elements, surface.n_faces, surface.n_face_points)
    alpha_face = np.empty(expected, dtype=float)
    beta_face = np.empty(expected, dtype=float)

    for f in range(surface.n_faces):
        alpha_face[:, f, :] = volume.alpha @ surface.face_interp[f].T
        beta_face[:, f, :] = volume.beta @ surface.face_interp[f].T

    return alpha_face, beta_face


def aligned_face(values: np.ndarray, k: int, f: int, nbr: int, nbr_f: int, flip: bool) -> np.ndarray:
    out = values[nbr, nbr_f]
    if flip:
        out = out[::-1]
    return out


def init_summary(
    *,
    ndiv: int,
    state: str,
    num_interfaces: int,
    matrix_max: float,
) -> dict[str, Any]:
    return {
        "ndiv": int(ndiv),
        "state": state,
        "num_interfaces": int(num_interfaces),
        "max_Rgamma_minus_VgammaP": float(matrix_max),
        "max_face_point_error": 0.0,
        "max_conormal_opposite_error": 0.0,
        "max_face_jacobian_error": 0.0,
        "max_face_jacobian_relative_error": 0.0,
        "max_abs_a_minus_plus_a_plus": 0.0,
        "max_abs_m_code_minus_m_pm": 0.0,
        "max_abs_epsilon_minus": 0.0,
        "max_abs_epsilon_plus": 0.0,
        "max_literal_vs_oriented_plus_difference": 0.0,
        "mean_literal_vs_oriented_plus_difference": 0.0,
        "max_abs_m_old_literal_subtracted_minus_m_pm": 0.0,
        "max_abs_m_literal_as_minus_orientation_minus_m_pm": 0.0,
        "_literal_abs_sum": 0.0,
        "_literal_count": 0,
    }


def update_max(summary: dict[str, Any], key: str, value: float) -> None:
    summary[key] = max(float(summary[key]), abs(float(value)))


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    count = int(summary.pop("_literal_count"))
    total = float(summary.pop("_literal_abs_sum"))
    summary["mean_literal_vs_oriented_plus_difference"] = total / count if count else 0.0
    return summary


def run_ndiv_diagnostic(
    *,
    objects: CaseObjects,
    ndiv: int,
    output_dir: Path,
    matrix_max: float,
    radius: float,
    sigma: float,
    amplitude: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    trace = objects.trace
    geom = objects.geom
    volume = objects.volume
    surface = objects.surface
    interior_faces = np.asarray(objects.conn.interior_faces, dtype=int)
    num_interfaces = int(interior_faces.shape[0])

    a_outward = projected_line_velocity(volume, surface)
    m_code_all = common_projected_line_velocity(volume, surface, trace, use_numba=False)
    alpha_face, beta_face = trace_alpha_beta_faces(objects)
    states = build_states(
        objects=objects,
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
        random_seed=random_seed + int(ndiv),
    )

    path = output_dir / f"interfaces_ndiv{ndiv}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERFACE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for state_name, q in states:
            qM = evaluate_face_traces(q, trace, use_numba=False)
            Fq_outward = projected_interior_line_flux(q, volume, surface)
            summary = init_summary(
                ndiv=ndiv,
                state=state_name,
                num_interfaces=num_interfaces,
                matrix_max=matrix_max,
            )

            for k_raw, f_raw, nbr_raw, nbr_f_raw in interior_faces:
                k = int(k_raw)
                f = int(f_raw)
                nbr = int(nbr_raw)
                nbr_f = int(nbr_f_raw)
                face_flip = bool(trace.face_flip[k, f])

                x_minus = geom.X_face[k, f]
                x_plus = aligned_face(geom.X_face, k, f, nbr, nbr_f, face_flip)
                conormal_minus = geom.face_conormal[k, f]
                conormal_plus = aligned_face(geom.face_conormal, k, f, nbr, nbr_f, face_flip)
                jac_minus = geom.face_jacobian[k, f]
                jac_plus = aligned_face(geom.face_jacobian, k, f, nbr, nbr_f, face_flip)

                a_minus = a_outward[k, f]
                a_plus_outward = aligned_face(a_outward, k, f, nbr, nbr_f, face_flip)
                a_plus_to_minus = -a_plus_outward
                m_pm = 0.5 * (a_minus - a_plus_outward)
                m_code = m_code_all[k, f]

                alpha_plus = aligned_face(alpha_face, k, f, nbr, nbr_f, face_flip)
                beta_plus = aligned_face(beta_face, k, f, nbr, nbr_f, face_flip)
                drdt_minus = float(_FACE_DRDT[f])
                dsdt_minus = float(_FACE_DSDT[f])
                literal_plus = dsdt_minus * alpha_plus - drdt_minus * beta_plus

                Rq_minus = qM[k, f]
                Rq_plus = aligned_face(qM, k, f, nbr, nbr_f, face_flip)
                Fq_minus = Fq_outward[k, f]
                Fq_plus_outward = aligned_face(Fq_outward, k, f, nbr, nbr_f, face_flip)
                Fq_plus_to_minus = -Fq_plus_outward

                epsilon_minus = Fq_minus - m_pm * Rq_minus
                epsilon_plus = Fq_plus_to_minus - m_pm * Rq_plus
                epsilon_plus_wrong = Fq_plus_outward - m_pm * Rq_plus

                m_old_literal_subtracted = 0.5 * (a_minus - literal_plus)
                m_literal_as_minus_orientation = 0.5 * (a_minus + literal_plus)

                x_error = np.linalg.norm(x_minus - x_plus, axis=1)
                conormal_error = np.linalg.norm(conormal_minus + conormal_plus, axis=1)
                jac_error = np.abs(jac_minus - jac_plus)
                jac_rel_error = jac_error / np.maximum.reduce(
                    [np.abs(jac_minus), np.abs(jac_plus), np.full_like(jac_error, RELATIVE_EPS)]
                )
                literal_diff = literal_plus - a_plus_to_minus

                update_max(summary, "max_face_point_error", float(np.max(x_error)))
                update_max(summary, "max_conormal_opposite_error", float(np.max(conormal_error)))
                update_max(summary, "max_face_jacobian_error", float(np.max(jac_error)))
                update_max(summary, "max_face_jacobian_relative_error", float(np.max(jac_rel_error)))
                update_max(summary, "max_abs_a_minus_plus_a_plus", float(np.max(np.abs(a_minus + a_plus_outward))))
                update_max(summary, "max_abs_m_code_minus_m_pm", float(np.max(np.abs(m_code - m_pm))))
                update_max(summary, "max_abs_epsilon_minus", float(np.max(np.abs(epsilon_minus))))
                update_max(summary, "max_abs_epsilon_plus", float(np.max(np.abs(epsilon_plus))))
                update_max(
                    summary,
                    "max_literal_vs_oriented_plus_difference",
                    float(np.max(np.abs(literal_diff))),
                )
                update_max(
                    summary,
                    "max_abs_m_old_literal_subtracted_minus_m_pm",
                    float(np.max(np.abs(m_old_literal_subtracted - m_pm))),
                )
                update_max(
                    summary,
                    "max_abs_m_literal_as_minus_orientation_minus_m_pm",
                    float(np.max(np.abs(m_literal_as_minus_orientation - m_pm))),
                )
                summary["_literal_abs_sum"] += float(np.sum(np.abs(literal_diff)))
                summary["_literal_count"] += int(literal_diff.size)

                for node in range(surface.n_face_points):
                    writer.writerow(
                        {
                            "ndiv": int(ndiv),
                            "state": state_name,
                            "element_minus": k,
                            "face_minus": f + 1,
                            "element_plus": nbr,
                            "face_plus": nbr_f + 1,
                            "face_flip": int(face_flip),
                            "face_node": int(node),
                            "x_match_error": float(x_error[node]),
                            "conormal_opposite_error": float(conormal_error[node]),
                            "face_jacobian_error": float(jac_error[node]),
                            "face_jacobian_relative_error": float(jac_rel_error[node]),
                            "a_minus": float(a_minus[node]),
                            "a_plus_outward": float(a_plus_outward[node]),
                            "a_plus_to_minus": float(a_plus_to_minus[node]),
                            "m_pm": float(m_pm[node]),
                            "m_code": float(m_code[node]),
                            "m_difference": float(m_code[node] - m_pm[node]),
                            "Rq_minus": float(Rq_minus[node]),
                            "Rq_plus": float(Rq_plus[node]),
                            "Fq_minus": float(Fq_minus[node]),
                            "Fq_plus_outward": float(Fq_plus_outward[node]),
                            "Fq_plus_to_minus": float(Fq_plus_to_minus[node]),
                            "epsilon_minus": float(epsilon_minus[node]),
                            "epsilon_plus": float(epsilon_plus[node]),
                            "literal_minus_normal_plus_value": float(literal_plus[node]),
                            "literal_vs_oriented_plus_difference": float(literal_diff[node]),
                            "m_old_literal_subtracted": float(m_old_literal_subtracted[node]),
                            "m_old_literal_subtracted_minus_m_pm": float(
                                m_old_literal_subtracted[node] - m_pm[node]
                            ),
                            "m_literal_as_minus_orientation": float(m_literal_as_minus_orientation[node]),
                            "m_literal_as_minus_orientation_minus_m_pm": float(
                                m_literal_as_minus_orientation[node] - m_pm[node]
                            ),
                            "epsilon_plus_wrong_outward_sign": float(epsilon_plus_wrong[node]),
                        }
                    )

            summary_rows.append(finalize_summary(summary))

    return summary_rows


def aggregate_by_ndiv(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    weighted_literal: dict[int, tuple[float, int]] = {}

    for row in summary_rows:
        ndiv = int(row["ndiv"])
        if ndiv not in grouped:
            grouped[ndiv] = {
                key: (0.0 if key not in ("ndiv", "state", "num_interfaces") else row.get(key, "all_states"))
                for key in SUMMARY_FIELDNAMES
            }
            grouped[ndiv]["ndiv"] = ndiv
            grouped[ndiv]["state"] = "all_states"
            grouped[ndiv]["num_interfaces"] = int(row["num_interfaces"])
            weighted_literal[ndiv] = (0.0, 0)

        out = grouped[ndiv]
        for key in SUMMARY_FIELDNAMES:
            if key in ("ndiv", "state", "num_interfaces", "mean_literal_vs_oriented_plus_difference"):
                continue
            out[key] = max(float(out[key]), abs(float(row[key])))

        total, count = weighted_literal[ndiv]
        count_i = int(row["num_interfaces"]) * 5
        total += float(row["mean_literal_vs_oriented_plus_difference"]) * count_i
        count += count_i
        weighted_literal[ndiv] = (total, count)

    for ndiv, (total, count) in weighted_literal.items():
        grouped[ndiv]["mean_literal_vs_oriented_plus_difference"] = total / count if count else 0.0

    return [grouped[ndiv] for ndiv in sorted(grouped)]


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    print("")
    print("Mesh-level maxima over all tested states")
    print("ndiv | R=VP        | x match     | conormal    | J face      | aM+aP      | m code      | eps-       | eps+       | literal")
    print("-----+-------------+-------------+-------------+-------------+------------+-------------+------------+------------+------------")
    for row in rows:
        print(
            f"{int(row['ndiv']):>4d} | "
            f"{float(row['max_Rgamma_minus_VgammaP']):.3e} | "
            f"{float(row['max_face_point_error']):.3e} | "
            f"{float(row['max_conormal_opposite_error']):.3e} | "
            f"{float(row['max_face_jacobian_error']):.3e} | "
            f"{float(row['max_abs_a_minus_plus_a_plus']):.3e} | "
            f"{float(row['max_abs_m_code_minus_m_pm']):.3e} | "
            f"{float(row['max_abs_epsilon_minus']):.3e} | "
            f"{float(row['max_abs_epsilon_plus']):.3e} | "
            f"{float(row['max_literal_vs_oriented_plus_difference']):.3e}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only checks for interface normal orientation and projected face traces."
    )
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER)
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE, choices=["table1", "table2"])
    parser.add_argument("--sbp", type=str, default=DEFAULT_SBP_VARIANT, choices=["projected"])
    parser.add_argument("--ndivs", nargs="+", type=int, default=list(DEFAULT_NDIVS))
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--omega", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.order <= 0:
        parser.error("--order must be positive.")
    if any(int(ndiv) <= 0 for ndiv in args.ndivs):
        parser.error("--ndivs values must be positive.")
    if len(set(int(ndiv) for ndiv in args.ndivs)) != len(args.ndivs):
        parser.error("--ndivs values must be unique.")
    if args.ndivs != sorted(args.ndivs):
        parser.error("--ndivs values must be increasing.")
    if args.radius <= 0.0:
        parser.error("--radius must be positive.")
    if args.sigma <= 0.0:
        parser.error("--sigma must be positive.")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ROOT / args.output_dir
    omega = tuple(float(v) for v in args.omega)

    matrix_rows = matrix_trace_checks(
        table=args.table,
        sbp_variant=args.sbp,
        orders=(1, 2, 3, 4),
    )
    matrix_max = max(float(row["Rgamma_minus_VgammaP_linf"]) for row in matrix_rows)
    write_csv(output_dir / "matrix_Rgamma_checks.csv", matrix_rows, MATRIX_FIELDNAMES)

    print("Interface normal orientation diagnostic")
    print("---------------------------------------")
    print(f"table       : {args.table}")
    print(f"sbp_variant : {args.sbp}")
    print(f"order       : {args.order}")
    print(f"ndivs       : {[int(v) for v in args.ndivs]}")
    print(f"omega       : {omega}")
    print(f"output_dir  : {output_dir}")
    print(f"matrix max  : {matrix_max:.6e}")

    all_summary_rows: list[dict[str, Any]] = []

    for ndiv in args.ndivs:
        ndiv_int = int(ndiv)
        print(f"running ndiv={ndiv_int} ...", flush=True)
        objects = build_case(
            order=args.order,
            table=args.table,
            sbp_variant=args.sbp,
            ndiv=ndiv_int,
            radius=float(args.radius),
            omega=omega,
        )
        rows = run_ndiv_diagnostic(
            objects=objects,
            ndiv=ndiv_int,
            output_dir=output_dir,
            matrix_max=matrix_max,
            radius=float(args.radius),
            sigma=float(args.sigma),
            amplitude=float(args.amplitude),
            random_seed=int(args.random_seed),
        )
        all_summary_rows.extend(rows)
        print(f"finished ndiv={ndiv_int}: {objects.mesh.elements.shape[0]} elements, {len(rows)} state summaries", flush=True)

    write_csv(output_dir / "summary.csv", all_summary_rows, SUMMARY_FIELDNAMES)
    mesh_rows = aggregate_by_ndiv(all_summary_rows)
    write_csv(output_dir / "summary_by_ndiv.csv", mesh_rows, SUMMARY_FIELDNAMES)

    metadata = {
        "table": args.table,
        "sbp_variant": args.sbp,
        "order": int(args.order),
        "matrix_orders": [1, 2, 3, 4],
        "ndivs": [int(v) for v in args.ndivs],
        "radius": float(args.radius),
        "sigma": float(args.sigma),
        "amplitude": float(args.amplitude),
        "random_seed": int(args.random_seed),
        "omega": list(omega),
        "states": ["constant", "polynomial", "projected_gaussian", "random_modal"],
        "orientation_definitions": {
            "n_r": "ds/dt for the local reference face",
            "n_s": "dr/dt for the local reference face",
            "a_outward": "n_r * R(alpha) - n_s * R(beta), computed independently on each element",
            "a_plus_to_minus": "-a_plus_outward after face-node reordering",
            "m_pm": "0.5 * (a_minus - a_plus_outward)",
            "epsilon_minus": "Fq_minus - m_pm * Rq_minus",
            "epsilon_plus": "(-Fq_plus_outward) - m_pm * Rq_plus",
            "literal_plus": "n_r_minus * R_plus(alpha_plus) - n_s_minus * R_plus(beta_plus)",
        },
        "matrix_checks": matrix_rows,
        "summary_by_state": all_summary_rows,
        "summary_by_ndiv": mesh_rows,
    }
    write_json(output_dir / "summary.json", metadata)
    print_summary_table(mesh_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
