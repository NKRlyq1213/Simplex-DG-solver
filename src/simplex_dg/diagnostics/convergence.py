from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConvergenceRow:
    level: int
    order: int
    n_elements: int
    n_points_per_element: int
    total_dofs: int
    dt: float
    tf: float
    nsteps: int
    hmin: float
    l2_error: float
    relative_l2_error: float
    linf_error: float
    mass_drift: float
    l2_norm_drift: float


def estimate_log2_rates(values: list[float]) -> list[float | None]:
    if len(values) == 0:
        return []

    rates: list[float | None] = [None]

    for i in range(1, len(values)):
        prev = float(values[i - 1])
        curr = float(values[i])

        if prev <= 0.0 or curr <= 0.0:
            rates.append(None)
        else:
            rates.append(float(np.log(prev / curr) / np.log(2.0)))

    return rates


def rows_to_dicts_with_rates(rows: list[ConvergenceRow]) -> list[dict[str, float | int | str]]:
    l2_rates = estimate_log2_rates([r.l2_error for r in rows])
    rel_rates = estimate_log2_rates([r.relative_l2_error for r in rows])
    linf_rates = estimate_log2_rates([r.linf_error for r in rows])

    out: list[dict[str, float | int | str]] = []

    for row, l2_rate, rel_rate, linf_rate in zip(rows, l2_rates, rel_rates, linf_rates):
        d = asdict(row)
        d["l2_rate"] = "" if l2_rate is None else l2_rate
        d["relative_l2_rate"] = "" if rel_rate is None else rel_rate
        d["linf_rate"] = "" if linf_rate is None else linf_rate
        out.append(d)

    return out


def write_convergence_csv(path: str | Path, rows: list[ConvergenceRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dicts = rows_to_dicts_with_rates(rows)

    if not dicts:
        raise ValueError("Cannot write empty convergence table.")

    fieldnames = list(dicts[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dicts)


def format_convergence_table(rows: list[ConvergenceRow]) -> str:
    dicts = rows_to_dicts_with_rates(rows)

    headers = [
        "level",
        "K",
        "DOFs",
        "dt",
        "steps",
        "hmin",
        "L2 err",
        "rate",
        "rel L2",
        "Linf",
        "mass drift",
        "L2 drift",
    ]

    lines = []
    lines.append(
        f"{headers[0]:>5} {headers[1]:>6} {headers[2]:>8} "
        f"{headers[3]:>11} {headers[4]:>7} {headers[5]:>11} "
        f"{headers[6]:>12} {headers[7]:>8} {headers[8]:>12} "
        f"{headers[9]:>12} {headers[10]:>12} {headers[11]:>12}"
    )

    for d in dicts:
        rate = d["l2_rate"]
        rate_s = "" if rate == "" else f"{float(rate):.3f}"

        lines.append(
            f"{int(d['level']):5d} "
            f"{int(d['n_elements']):6d} "
            f"{int(d['total_dofs']):8d} "
            f"{float(d['dt']):11.4e} "
            f"{int(d['nsteps']):7d} "
            f"{float(d['hmin']):11.4e} "
            f"{float(d['l2_error']):12.4e} "
            f"{rate_s:>8} "
            f"{float(d['relative_l2_error']):12.4e} "
            f"{float(d['linf_error']):12.4e} "
            f"{float(d['mass_drift']):12.4e} "
            f"{float(d['l2_norm_drift']):12.4e}"
        )

    return "\n".join(lines)