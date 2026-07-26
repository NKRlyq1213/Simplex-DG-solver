from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

import numpy as np

from simplex_dg.rhs import (
    SurfaceRHSCache,
    VolumeRHSCache,
    projected_interior_line_flux,
    projected_line_velocity,
)
from simplex_dg.trace import TraceCache, evaluate_face_traces


@dataclass(frozen=True)
class ProjectedProductResidualArrays:
    q_face: np.ndarray
    projected_flux: np.ndarray
    projected_velocity: np.ndarray
    residual: np.ndarray
    residual_r: np.ndarray
    residual_s: np.ndarray
    alpha_q_face: np.ndarray
    alpha_face: np.ndarray
    beta_q_face: np.ndarray
    beta_face: np.ndarray


@dataclass(frozen=True)
class ProjectedProductResidualReport:
    absolute_weighted_l2: float
    relative_weighted_l2: float
    absolute_linf: float
    relative_linf: float
    r_absolute_weighted_l2: float
    s_absolute_weighted_l2: float
    reference_flux_weighted_l2: float


@dataclass(frozen=True)
class ProjectedProductConvergenceRow:
    ndivs: int
    order: int
    n_elements: int
    total_dofs: int
    hmin: float
    absolute_weighted_l2: float
    relative_weighted_l2: float
    absolute_linf: float
    relative_linf: float
    r_absolute_weighted_l2: float
    s_absolute_weighted_l2: float
    reference_flux_weighted_l2: float
    relative_weighted_l2_rate: float | None = None
    relative_linf_rate: float | None = None


def observed_rate(
    prev_error: float,
    curr_error: float,
    prev_resolution: float,
    curr_resolution: float,
) -> float | None:
    prev_error = float(prev_error)
    curr_error = float(curr_error)
    prev_resolution = float(prev_resolution)
    curr_resolution = float(curr_resolution)

    if not np.isfinite(prev_error) or not np.isfinite(curr_error):
        return None

    if not np.isfinite(prev_resolution) or not np.isfinite(curr_resolution):
        return None

    if prev_error <= 0.0 or curr_error <= 0.0:
        return None

    if (
        prev_resolution <= 0.0
        or curr_resolution <= 0.0
        or np.isclose(prev_resolution, curr_resolution)
    ):
        return None

    scale = max(abs(prev_error), abs(curr_error), 1.0)
    if min(abs(prev_error), abs(curr_error)) <= 100.0 * np.finfo(float).eps * scale:
        return None

    return float(np.log(prev_error / curr_error) / np.log(prev_resolution / curr_resolution))


def _sorted_rows(
    rows: list[ProjectedProductConvergenceRow],
    rate_basis: str,
) -> list[ProjectedProductConvergenceRow]:
    if rate_basis == "hmin":
        return sorted(rows, key=lambda row: row.hmin, reverse=True)

    if rate_basis == "ndiv":
        return sorted(rows, key=lambda row: row.ndivs)

    raise ValueError("rate_basis must be 'hmin' or 'ndiv'.")


def _rate_resolution(
    row: ProjectedProductConvergenceRow,
    rate_basis: str,
    *,
    position: str,
) -> float:
    if rate_basis == "hmin":
        return float(row.hmin)

    if rate_basis == "ndiv":
        value = float(row.ndivs)
        if position == "prev":
            return 1.0 / value
        if position == "curr":
            return 1.0 / value
        raise ValueError("position must be 'prev' or 'curr'.")

    raise ValueError("rate_basis must be 'hmin' or 'ndiv'.")


def _validate_volume_surface_trace(
    volume: VolumeRHSCache,
    surface: SurfaceRHSCache,
    trace: TraceCache,
) -> None:
    if volume.n_elements != surface.n_elements or volume.n_elements != trace.n_elements:
        raise ValueError("volume, surface, and trace must have the same number of elements.")

    if volume.n_points != surface.n_points or volume.n_points != trace.n_points:
        raise ValueError("volume, surface, and trace must have the same number of volume points.")

    if surface.n_faces != trace.n_faces or surface.n_face_points != trace.n_face_points:
        raise ValueError("surface and trace must have the same face dimensions.")


def projected_product_residual_arrays(
    q: np.ndarray,
    volume: VolumeRHSCache,
    surface: SurfaceRHSCache,
    trace: TraceCache,
    *,
    use_numba: bool | None = None,
) -> ProjectedProductResidualArrays:
    _validate_volume_surface_trace(volume, surface, trace)

    q = np.asarray(q, dtype=float)
    expected_vol = (volume.n_elements, volume.n_points)

    if q.shape != expected_vol:
        raise ValueError(f"q must have shape {expected_vol}.")

    q_face = evaluate_face_traces(q, trace, use_numba=use_numba)

    alpha_q_face = evaluate_face_traces(volume.alpha * q, trace, use_numba=use_numba)
    alpha_face = evaluate_face_traces(volume.alpha, trace, use_numba=use_numba)
    beta_q_face = evaluate_face_traces(volume.beta * q, trace, use_numba=use_numba)
    beta_face = evaluate_face_traces(volume.beta, trace, use_numba=use_numba)

    residual_r = alpha_q_face - alpha_face * q_face
    residual_s = beta_q_face - beta_face * q_face

    projected_flux = projected_interior_line_flux(q=q, volume=volume, cache=surface)
    projected_velocity = projected_line_velocity(volume=volume, cache=surface)
    residual = projected_flux - projected_velocity * q_face

    return ProjectedProductResidualArrays(
        q_face=q_face,
        projected_flux=projected_flux,
        projected_velocity=projected_velocity,
        residual=residual,
        residual_r=residual_r,
        residual_s=residual_s,
        alpha_q_face=alpha_q_face,
        alpha_face=alpha_face,
        beta_q_face=beta_q_face,
        beta_face=beta_face,
    )


def _weighted_boundary_l2(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    return float(np.sqrt(np.sum(weights[None, :, :] * values * values)))


def projected_product_residual_report(
    q: np.ndarray,
    volume: VolumeRHSCache,
    surface: SurfaceRHSCache,
    trace: TraceCache,
) -> ProjectedProductResidualReport:
    arrays = projected_product_residual_arrays(
        q=q,
        volume=volume,
        surface=surface,
        trace=trace,
    )

    weights = np.asarray(trace.face_weights, dtype=float)
    tiny = np.finfo(float).tiny

    absolute_weighted_l2 = _weighted_boundary_l2(arrays.residual, weights)
    reference_flux_weighted_l2 = _weighted_boundary_l2(arrays.projected_flux, weights)
    absolute_linf = float(np.max(np.abs(arrays.residual)))
    flux_linf = float(np.max(np.abs(arrays.projected_flux)))

    report = ProjectedProductResidualReport(
        absolute_weighted_l2=absolute_weighted_l2,
        relative_weighted_l2=absolute_weighted_l2 / max(reference_flux_weighted_l2, tiny),
        absolute_linf=absolute_linf,
        relative_linf=absolute_linf / max(flux_linf, tiny),
        r_absolute_weighted_l2=_weighted_boundary_l2(arrays.residual_r, weights),
        s_absolute_weighted_l2=_weighted_boundary_l2(arrays.residual_s, weights),
        reference_flux_weighted_l2=reference_flux_weighted_l2,
    )

    if not all(
        np.isfinite(value)
        for value in (
            report.absolute_weighted_l2,
            report.relative_weighted_l2,
            report.absolute_linf,
            report.relative_linf,
            report.r_absolute_weighted_l2,
            report.s_absolute_weighted_l2,
            report.reference_flux_weighted_l2,
        )
    ):
        raise ValueError("Projected-product residual report contains non-finite values.")

    return report


def attach_rates(
    rows: list[ProjectedProductConvergenceRow],
    rate_basis: str = "hmin",
) -> list[ProjectedProductConvergenceRow]:
    if not rows:
        return []

    rows_sorted = _sorted_rows(rows, rate_basis)
    out: list[ProjectedProductConvergenceRow] = []
    prev: ProjectedProductConvergenceRow | None = None

    for row in rows_sorted:
        if prev is None:
            out.append(row)
        else:
            row_dict = asdict(row)
            prev_resolution = _rate_resolution(prev, rate_basis, position="prev")
            curr_resolution = _rate_resolution(row, rate_basis, position="curr")
            row_dict["relative_weighted_l2_rate"] = observed_rate(
                prev.relative_weighted_l2,
                row.relative_weighted_l2,
                prev_resolution,
                curr_resolution,
            )
            row_dict["relative_linf_rate"] = observed_rate(
                prev.relative_linf,
                row.relative_linf,
                prev_resolution,
                curr_resolution,
            )
            out.append(ProjectedProductConvergenceRow(**row_dict))

        prev = row

    return out


def rows_to_dicts(
    rows: list[ProjectedProductConvergenceRow],
    rate_basis: str = "hmin",
) -> list[dict[str, float | int | str]]:
    dicts: list[dict[str, float | int | str]] = []

    for row in attach_rates(rows, rate_basis=rate_basis):
        row_dict = asdict(row)
        if row_dict["relative_weighted_l2_rate"] is None:
            row_dict["relative_weighted_l2_rate"] = ""
        if row_dict["relative_linf_rate"] is None:
            row_dict["relative_linf_rate"] = ""
        dicts.append(row_dict)

    return dicts


def write_csv(
    path: str | Path,
    rows: list[ProjectedProductConvergenceRow],
    rate_basis: str = "hmin",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dicts = rows_to_dicts(rows, rate_basis=rate_basis)

    if not dicts:
        raise ValueError("Cannot write an empty projected-product convergence table.")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dicts[0].keys()))
        writer.writeheader()
        writer.writerows(dicts)

    return path
