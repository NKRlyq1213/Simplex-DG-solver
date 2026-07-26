from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import csv
from pathlib import Path

import numpy as np

from simplex_dg.geometry import GeometryCache
from simplex_dg.reference import ReferenceCache
from simplex_dg.rhs import VolumeRHSCache, apply_reference_operator
from simplex_dg.time import minimum_face_length


def nodal_projection_matrix(ref: ReferenceCache) -> np.ndarray:
    """Return the nodal projector P = V (M^{-1} V^T W)."""
    return np.asarray(ref.V @ ref.projection, dtype=float)


def project_nodal_samples(values: np.ndarray, ref: ReferenceCache) -> np.ndarray:
    """Project nodal samples onto the discrete polynomial space on each element."""
    values = np.asarray(values, dtype=float)
    projector = nodal_projection_matrix(ref)

    if values.ndim == 1:
        if values.shape[0] != projector.shape[0]:
            raise ValueError("values has incompatible length.")
        return projector @ values

    if values.ndim == 2:
        if values.shape[1] != projector.shape[0]:
            raise ValueError("values has incompatible trailing dimension.")
        return values @ projector.T

    raise ValueError("values must have shape (Np,) or (K, Np).")


def projection_closure_linf(values: np.ndarray, ref: ReferenceCache) -> float:
    values = np.asarray(values, dtype=float)
    closed = project_nodal_samples(values, ref)
    return float(np.max(np.abs(values - closed)))


def weighted_reference_l2(values: np.ndarray, ref: ReferenceCache) -> float:
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] != ref.weights.size:
        raise ValueError("values must have shape (K, Np).")

    return float(np.sqrt(np.sum(ref.area * ref.weights[None, :] * values * values)))


def weighted_physical_l2_divided_by_j(
    values: np.ndarray,
    sqrt_g: np.ndarray,
    ref: ReferenceCache,
) -> float:
    values = np.asarray(values, dtype=float)
    sqrt_g = np.asarray(sqrt_g, dtype=float)

    if values.shape != sqrt_g.shape:
        raise ValueError("values and sqrt_g must have the same shape.")

    if np.any(sqrt_g <= 0.0):
        raise ValueError("sqrt_g must be positive.")

    return float(np.sqrt(np.sum(ref.area * ref.weights[None, :] * (values * values) / sqrt_g)))


def weighted_energy_residual(
    q_h: np.ndarray,
    tau_sum: np.ndarray,
    ref: ReferenceCache,
) -> float:
    q_h = np.asarray(q_h, dtype=float)
    tau_sum = np.asarray(tau_sum, dtype=float)

    if q_h.shape != tau_sum.shape:
        raise ValueError("q_h and tau_sum must have the same shape.")

    return float(0.5 * np.sum(ref.area * ref.weights[None, :] * q_h * tau_sum))


def discrete_energy(
    q_h: np.ndarray,
    sqrt_g: np.ndarray,
    ref: ReferenceCache,
) -> float:
    q_h = np.asarray(q_h, dtype=float)
    sqrt_g = np.asarray(sqrt_g, dtype=float)

    if q_h.shape != sqrt_g.shape:
        raise ValueError("q_h and sqrt_g must have the same shape.")

    return float(0.5 * np.sum(ref.area * ref.weights[None, :] * sqrt_g * q_h * q_h))


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
    rows: list[LeibnizDefectRow],
    rate_basis: str,
) -> list[LeibnizDefectRow]:
    if rate_basis == "hmin":
        return sorted(rows, key=lambda row: row.hmin, reverse=True)

    if rate_basis == "ndiv":
        return sorted(rows, key=lambda row: row.ndivs)

    raise ValueError("rate_basis must be 'hmin' or 'ndiv'.")


def _rate_resolution(
    row: LeibnizDefectRow,
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


def _row_rate_fields() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(LeibnizDefectRow)
        if field.name.endswith("_rate")
    )


def _base_error_fields() -> tuple[str, ...]:
    return tuple(name[: -len("_rate")] for name in _row_rate_fields())


@dataclass(frozen=True)
class LeibnizDefects:
    Dr_q: np.ndarray
    Ds_q: np.ndarray
    Dr_alpha_q: np.ndarray
    Ds_beta_q: np.ndarray
    tau_r: np.ndarray
    tau_s: np.ndarray
    tau_sum: np.ndarray


def compute_leibniz_defect_component(
    D: np.ndarray,
    coeff: np.ndarray,
    D_coeff: np.ndarray,
    q_h: np.ndarray,
) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=float)
    D_coeff = np.asarray(D_coeff, dtype=float)
    q_h = np.asarray(q_h, dtype=float)

    if coeff.shape != q_h.shape or D_coeff.shape != q_h.shape:
        raise ValueError("coeff, D_coeff, and q_h must have the same shape.")

    D_q = apply_reference_operator(D, q_h)
    D_coeff_q = apply_reference_operator(D, coeff * q_h)

    return D_coeff_q - coeff * D_q - q_h * D_coeff


def compute_leibniz_defects(
    q_h: np.ndarray,
    volume: VolumeRHSCache,
) -> LeibnizDefects:
    q_h = np.asarray(q_h, dtype=float)
    expected = (volume.n_elements, volume.n_points)

    if q_h.shape != expected:
        raise ValueError(f"q_h must have shape {expected}.")

    Dr_q = apply_reference_operator(volume.Dr, q_h)
    Ds_q = apply_reference_operator(volume.Ds, q_h)
    Dr_alpha_q = apply_reference_operator(volume.Dr, volume.alpha * q_h)
    Ds_beta_q = apply_reference_operator(volume.Ds, volume.beta * q_h)

    tau_r = Dr_alpha_q - volume.alpha * Dr_q - q_h * volume.Dr_alpha
    tau_s = Ds_beta_q - volume.beta * Ds_q - q_h * volume.Ds_beta

    return LeibnizDefects(
        Dr_q=Dr_q,
        Ds_q=Ds_q,
        Dr_alpha_q=Dr_alpha_q,
        Ds_beta_q=Ds_beta_q,
        tau_r=tau_r,
        tau_s=tau_s,
        tau_sum=tau_r + tau_s,
    )


@dataclass(frozen=True)
class LeibnizDefectRow:
    order: int
    table: str
    state: str
    ndivs: int
    n_elements: int
    n_points_per_element: int
    total_dofs: int
    hmin: float

    q_projection_closure_linf: float

    tau_r_linf: float
    tau_r_l2_ref: float
    tau_s_linf: float
    tau_s_l2_ref: float
    tau_sum_linf: float
    tau_sum_l2_ref: float

    physical_tau_r_linf: float
    physical_tau_r_l2: float
    physical_tau_s_linf: float
    physical_tau_s_l2: float
    physical_tau_sum_linf: float
    physical_tau_sum_l2: float

    energy: float
    abs_energy_residual: float
    relative_energy_residual: float

    tau_r_linf_rate: float | None = None
    tau_r_l2_ref_rate: float | None = None
    tau_s_linf_rate: float | None = None
    tau_s_l2_ref_rate: float | None = None
    tau_sum_linf_rate: float | None = None
    tau_sum_l2_ref_rate: float | None = None

    physical_tau_r_linf_rate: float | None = None
    physical_tau_r_l2_rate: float | None = None
    physical_tau_s_linf_rate: float | None = None
    physical_tau_s_l2_rate: float | None = None
    physical_tau_sum_linf_rate: float | None = None
    physical_tau_sum_l2_rate: float | None = None

    abs_energy_residual_rate: float | None = None
    relative_energy_residual_rate: float | None = None


@dataclass(frozen=True)
class LeibnizExperimentResult:
    row: LeibnizDefectRow
    q_raw: np.ndarray
    q_h: np.ndarray
    defects: LeibnizDefects


def compute_leibniz_defect_row(
    *,
    order: int,
    table: str,
    state: str,
    ref: ReferenceCache,
    geom: GeometryCache,
    volume: VolumeRHSCache,
    q_raw: np.ndarray,
    ndivs: int,
) -> LeibnizExperimentResult:
    q_raw = np.asarray(q_raw, dtype=float)

    if state == "projected-gaussian":
        q_h = project_nodal_samples(q_raw, ref)
    elif state == "raw-gaussian":
        q_h = q_raw.copy()
    else:
        raise ValueError(f"Unsupported state representation: {state!r}")

    defects = compute_leibniz_defects(q_h, volume)
    tau_r_over_j = defects.tau_r / volume.sqrt_g
    tau_s_over_j = defects.tau_s / volume.sqrt_g
    tau_sum_over_j = defects.tau_sum / volume.sqrt_g

    energy = discrete_energy(q_h, volume.sqrt_g, ref)
    abs_energy_residual = abs(weighted_energy_residual(q_h, defects.tau_sum, ref))
    energy_denom = max(abs(energy), np.finfo(float).tiny)

    row = LeibnizDefectRow(
        order=order,
        table=table,
        state=state,
        ndivs=ndivs,
        n_elements=volume.n_elements,
        n_points_per_element=volume.n_points,
        total_dofs=volume.n_elements * volume.n_points,
        hmin=minimum_face_length(ref, geom),
        q_projection_closure_linf=projection_closure_linf(q_h, ref),
        tau_r_linf=float(np.max(np.abs(defects.tau_r))),
        tau_r_l2_ref=weighted_reference_l2(defects.tau_r, ref),
        tau_s_linf=float(np.max(np.abs(defects.tau_s))),
        tau_s_l2_ref=weighted_reference_l2(defects.tau_s, ref),
        tau_sum_linf=float(np.max(np.abs(defects.tau_sum))),
        tau_sum_l2_ref=weighted_reference_l2(defects.tau_sum, ref),
        physical_tau_r_linf=float(np.max(np.abs(tau_r_over_j))),
        physical_tau_r_l2=weighted_physical_l2_divided_by_j(defects.tau_r, volume.sqrt_g, ref),
        physical_tau_s_linf=float(np.max(np.abs(tau_s_over_j))),
        physical_tau_s_l2=weighted_physical_l2_divided_by_j(defects.tau_s, volume.sqrt_g, ref),
        physical_tau_sum_linf=float(np.max(np.abs(tau_sum_over_j))),
        physical_tau_sum_l2=weighted_physical_l2_divided_by_j(defects.tau_sum, volume.sqrt_g, ref),
        energy=energy,
        abs_energy_residual=abs_energy_residual,
        relative_energy_residual=abs_energy_residual / energy_denom,
    )

    return LeibnizExperimentResult(
        row=row,
        q_raw=q_raw,
        q_h=q_h,
        defects=defects,
    )


def attach_rates(
    rows: list[LeibnizDefectRow],
    rate_basis: str = "hmin",
) -> list[LeibnizDefectRow]:
    if not rows:
        return []

    rows_sorted = _sorted_rows(rows, rate_basis)
    prev: LeibnizDefectRow | None = None
    out: list[LeibnizDefectRow] = []

    error_fields = _base_error_fields()

    for row in rows_sorted:
        if prev is None:
            out.append(row)
            prev = row
            continue

        row_dict = asdict(row)
        prev_resolution = _rate_resolution(prev, rate_basis, position="prev")
        curr_resolution = _rate_resolution(row, rate_basis, position="curr")

        for field_name in error_fields:
            row_dict[f"{field_name}_rate"] = observed_rate(
                getattr(prev, field_name),
                getattr(row, field_name),
                prev_resolution,
                curr_resolution,
            )

        out.append(LeibnizDefectRow(**row_dict))
        prev = row

    return out


def rows_to_dicts(
    rows: list[LeibnizDefectRow],
    rate_basis: str = "hmin",
) -> list[dict[str, float | int | str]]:
    dicts: list[dict[str, float | int | str]] = []

    for row in attach_rates(rows, rate_basis=rate_basis):
        row_dict = asdict(row)
        for field_name in _row_rate_fields():
            if row_dict[field_name] is None:
                row_dict[field_name] = ""
        dicts.append(row_dict)

    return dicts


def write_csv(
    path: str | Path,
    rows: list[LeibnizDefectRow],
    rate_basis: str = "hmin",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dicts = rows_to_dicts(rows, rate_basis=rate_basis)

    if not dicts:
        raise ValueError("Cannot write an empty Leibniz-defect table.")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dicts[0].keys()))
        writer.writeheader()
        writer.writerows(dicts)

    return path


def _fmt_rate(rate: float | None) -> str:
    if rate is None:
        return "-"
    return f"{rate:.2f}"


def format_reference_table(
    rows: list[LeibnizDefectRow],
    rate_basis: str = "hmin",
) -> str:
    lines = [
        "Table A: reference-form defects",
        (
            "ndivs | K     | hmin       | "
            "||tau_r||inf | rate | ||tau_r||L2ref | rate | "
            "||tau_s||inf | rate | ||tau_s||L2ref | rate | "
            "||tau_sum||inf | rate | ||tau_sum||L2ref | rate"
        ),
    ]
    lines.append("-" * len(lines[-1]))

    for row in attach_rates(rows, rate_basis=rate_basis):
        lines.append(
            f"{row.ndivs:<5d} | "
            f"{row.n_elements:<5d} | "
            f"{row.hmin:<10.4e} | "
            f"{row.tau_r_linf:<12.4e} | "
            f"{_fmt_rate(row.tau_r_linf_rate):>4s} | "
            f"{row.tau_r_l2_ref:<14.4e} | "
            f"{_fmt_rate(row.tau_r_l2_ref_rate):>4s} | "
            f"{row.tau_s_linf:<12.4e} | "
            f"{_fmt_rate(row.tau_s_linf_rate):>4s} | "
            f"{row.tau_s_l2_ref:<14.4e} | "
            f"{_fmt_rate(row.tau_s_l2_ref_rate):>4s} | "
            f"{row.tau_sum_linf:<14.4e} | "
            f"{_fmt_rate(row.tau_sum_linf_rate):>4s} | "
            f"{row.tau_sum_l2_ref:<16.4e} | "
            f"{_fmt_rate(row.tau_sum_l2_ref_rate):>4s}"
        )

    return "\n".join(lines)


def format_physical_table(
    rows: list[LeibnizDefectRow],
    rate_basis: str = "hmin",
) -> str:
    lines = [
        "Table B: physical defects and energy residual",
        (
            "ndivs | K     | hmin       | "
            "||tau_r/J||inf | rate | ||tau_r/J||L2 | rate | "
            "||tau_s/J||inf | rate | ||tau_s/J||L2 | rate | "
            "||tau_sum/J||inf | rate | ||tau_sum/J||L2 | rate | "
            "|epsilon_tau| | |epsilon_tau|/E"
        ),
    ]
    lines.append("-" * len(lines[-1]))

    for row in attach_rates(rows, rate_basis=rate_basis):
        lines.append(
            f"{row.ndivs:<5d} | "
            f"{row.n_elements:<5d} | "
            f"{row.hmin:<10.4e} | "
            f"{row.physical_tau_r_linf:<14.4e} | "
            f"{_fmt_rate(row.physical_tau_r_linf_rate):>4s} | "
            f"{row.physical_tau_r_l2:<14.4e} | "
            f"{_fmt_rate(row.physical_tau_r_l2_rate):>4s} | "
            f"{row.physical_tau_s_linf:<14.4e} | "
            f"{_fmt_rate(row.physical_tau_s_linf_rate):>4s} | "
            f"{row.physical_tau_s_l2:<14.4e} | "
            f"{_fmt_rate(row.physical_tau_s_l2_rate):>4s} | "
            f"{row.physical_tau_sum_linf:<16.4e} | "
            f"{_fmt_rate(row.physical_tau_sum_linf_rate):>4s} | "
            f"{row.physical_tau_sum_l2:<16.4e} | "
            f"{_fmt_rate(row.physical_tau_sum_l2_rate):>4s} | "
            f"{row.abs_energy_residual:<12.4e} | "
            f"{row.relative_energy_residual:<12.4e}"
        )

    return "\n".join(lines)
