from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RK4A = np.array(
    [
        0.0,
        -567301805773.0 / 1357537059087.0,
        -2404267990393.0 / 2016746695238.0,
        -3550918686646.0 / 2091501179385.0,
        -1275806237668.0 / 842570457699.0,
    ],
    dtype=float,
)

RK4B = np.array(
    [
        1432997174477.0 / 9575080441755.0,
        5161836677717.0 / 13612068292357.0,
        1720146321549.0 / 2090206949498.0,
        3134564353537.0 / 4481467310338.0,
        2277821191437.0 / 14882151754819.0,
    ],
    dtype=float,
)

RK4C = np.array(
    [
        0.0,
        1432997174477.0 / 9575080441755.0,
        2526269341429.0 / 6820363962896.0,
        2006345519317.0 / 3224310063776.0,
        2802321613138.0 / 2924317926251.0,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class TimeIntegrationResult:
    q: np.ndarray
    t: float
    nsteps: int
    history: list[dict[str, float]]


def final_time_tolerance(tf: float, t0: float = 0.0) -> float:
    return 1e-14 * max(1.0, abs(float(tf)), abs(float(t0)))


def lsrk54_step(
    rhs,
    t: float,
    q: np.ndarray,
    dt: float,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    q_in = np.asarray(q, dtype=float)

    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    if out is None:
        q_out = q_in.copy()
    else:
        q_out = np.asarray(out, dtype=float)
        if q_out.shape != q_in.shape:
            raise ValueError("out must have the same shape as q.")
        q_out[...] = q_in

    res = np.zeros_like(q_out)

    for stage in range(5):
        t_stage = float(t) + RK4C[stage] * dt
        dqdt = np.asarray(rhs(t_stage, q_out), dtype=float)

        if dqdt.shape != q_out.shape:
            raise ValueError("rhs(t, q) must return the same shape as q.")

        res *= RK4A[stage]
        res += dt * dqdt
        q_out += RK4B[stage] * res

    return q_out


def integrate_lsrk54(
    rhs,
    q0: np.ndarray,
    t0: float,
    tf: float,
    dt: float,
    *,
    max_steps: int = 10_000_000,
    monitor=None,
    monitor_every: int = 1,
) -> TimeIntegrationResult:
    if tf < t0:
        raise ValueError("Require tf >= t0.")

    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    if monitor_every <= 0:
        raise ValueError("monitor_every must be positive.")

    q = np.asarray(q0, dtype=float).copy()
    t = float(t0)
    tf = float(tf)
    tol = final_time_tolerance(tf, t0)

    history: list[dict[str, float]] = []

    if monitor is not None:
        history.append(monitor(t, q))

    nsteps = 0

    while t < tf - tol:
        if nsteps >= max_steps:
            raise RuntimeError("Maximum number of time steps exceeded.")

        dt_step = min(float(dt), tf - t)

        q = lsrk54_step(rhs, t, q, dt_step)
        t += dt_step
        nsteps += 1

        if monitor is not None and (nsteps % monitor_every == 0 or t >= tf - tol):
            history.append(monitor(t, q))

        if not np.all(np.isfinite(q)):
            raise FloatingPointError("Non-finite state encountered during integration.")

    if abs(t - tf) <= tol:
        t = tf

    return TimeIntegrationResult(
        q=q,
        t=t,
        nsteps=nsteps,
        history=history,
    )