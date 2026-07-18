from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConvergenceRow:
    ndivs: int
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
    initial_mass: float = 0.0
    final_mass: float = 0.0
    absolute_mass_drift: float = 0.0
    relative_mass_drift: float = 0.0
    initial_energy: float = 0.0
    final_energy: float = 0.0
    absolute_energy_drift: float = 0.0
    relative_energy_drift: float = 0.0
    q_min: float = 0.0
    q_max: float = 0.0
    undershoot: float = 0.0
    overshoot: float = 0.0
    elapsed_seconds: float = 0.0


def estimate_convergence_rates(values: list[float], hmins: list[float]) -> list[float | None]:
    if len(values) != len(hmins):
        raise ValueError("values and hmins must have the same length.")

    if len(values) == 0:
        return []

    rates: list[float | None] = [None]

    for i in range(1, len(values)):
        prev = float(values[i - 1])
        curr = float(values[i])
        h_prev = float(hmins[i - 1])
        h_curr = float(hmins[i])

        if prev <= 0.0 or curr <= 0.0 or h_prev <= 0.0 or h_curr <= 0.0 or np.isclose(h_prev, h_curr):
            rates.append(None)
        else:
            rates.append(float(np.log(prev / curr) / np.log(h_prev / h_curr)))

    return rates


def rows_to_dicts_with_rates(rows: list[ConvergenceRow]) -> list[dict[str, float | int | str]]:
    hmins = [r.hmin for r in rows]
    l2_rates = estimate_convergence_rates([r.l2_error for r in rows], hmins)
    rel_rates = estimate_convergence_rates([r.relative_l2_error for r in rows], hmins)
    linf_rates = estimate_convergence_rates([r.linf_error for r in rows], hmins)

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
        "ndivs",
        "K",
        "Nq",
        "DOFs",
        "dt",
        "steps",
        "hmin",
        "L2 err",
        "rate",
        "rel L2",
        "rate",
        "Linf",
        "rate",
        "rel mass",
        "rel energy",
        "qmin",
        "qmax",
    ]

    lines = []
    lines.append(
        f"{headers[0]:>5} {headers[1]:>6} {headers[2]:>4} {headers[3]:>8} "
        f"{headers[4]:>11} {headers[5]:>7} {headers[6]:>11} "
        f"{headers[7]:>12} {headers[8]:>8} {headers[9]:>12} {headers[10]:>8} "
        f"{headers[11]:>12} {headers[12]:>8} {headers[13]:>12} "
        f"{headers[14]:>12} {headers[15]:>10} {headers[16]:>10}"
    )

    for d in dicts:
        rate = d["l2_rate"]
        rate_s = "" if rate == "" else f"{float(rate):.3f}"
        rel_rate = d["relative_l2_rate"]
        rel_rate_s = "" if rel_rate == "" else f"{float(rel_rate):.3f}"
        linf_rate = d["linf_rate"]
        linf_rate_s = "" if linf_rate == "" else f"{float(linf_rate):.3f}"

        lines.append(
            f"{int(d['ndivs']):5d} "
            f"{int(d['n_elements']):6d} "
            f"{int(d['n_points_per_element']):4d} "
            f"{int(d['total_dofs']):8d} "
            f"{float(d['dt']):11.4e} "
            f"{int(d['nsteps']):7d} "
            f"{float(d['hmin']):11.4e} "
            f"{float(d['l2_error']):12.4e} "
            f"{rate_s:>8} "
            f"{float(d['relative_l2_error']):12.4e} "
            f"{rel_rate_s:>8} "
            f"{float(d['linf_error']):12.4e} "
            f"{linf_rate_s:>8} "
            f"{float(d['relative_mass_drift']):12.4e} "
            f"{float(d['relative_energy_drift']):12.4e} "
            f"{float(d['q_min']):10.4e} "
            f"{float(d['q_max']):10.4e}"
        )

    return "\n".join(lines)
