from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np

from simplex_dg.mesh.connectivity import ConnectivityCache
from simplex_dg.reference.operators import ReferenceCache


try:
    _numba = importlib.import_module("numba")
    njit = _numba.njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class TraceCache:
    n_elements: int
    n_points: int
    n_faces: int
    n_face_points: int

    face_interp: np.ndarray
    face_weights: np.ndarray

    neighbor_elements: np.ndarray
    neighbor_faces: np.ndarray
    is_boundary: np.ndarray
    face_flip: np.ndarray


@dataclass(frozen=True)
class FaceTraces:
    qM: np.ndarray
    qP: np.ndarray


def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _evaluate_faces_kernel(q, face_interp, qM):
        K = q.shape[0]
        Np = q.shape[1]
        n_faces = face_interp.shape[0]
        Nf = face_interp.shape[1]

        for k in range(K):
            for f in range(n_faces):
                for i in range(Nf):
                    acc = 0.0
                    for j in range(Np):
                        acc += face_interp[f, i, j] * q[k, j]
                    qM[k, f, i] = acc


    @njit(cache=True)
    def _gather_neighbor_faces_kernel(qM, neighbor_elements, neighbor_faces, is_boundary, face_flip, boundary_value, qP):
        K = qM.shape[0]
        n_faces = qM.shape[1]
        Nf = qM.shape[2]

        for k in range(K):
            for f in range(n_faces):
                if is_boundary[k, f]:
                    for i in range(Nf):
                        qP[k, f, i] = boundary_value
                else:
                    nbr = neighbor_elements[k, f]
                    nbr_f = neighbor_faces[k, f]

                    if face_flip[k, f]:
                        for i in range(Nf):
                            qP[k, f, i] = qM[nbr, nbr_f, Nf - 1 - i]
                    else:
                        for i in range(Nf):
                            qP[k, f, i] = qM[nbr, nbr_f, i]
else:
    _evaluate_faces_kernel = None
    _gather_neighbor_faces_kernel = None


def build_trace_cache(
    ref: ReferenceCache,
    conn: ConnectivityCache,
    validate: bool = True,
) -> TraceCache:
    EToE = np.asarray(conn.EToE, dtype=int)
    EToF = np.asarray(conn.EToF, dtype=int)
    is_boundary = np.asarray(conn.is_boundary, dtype=bool)
    face_flip = np.asarray(conn.face_flip, dtype=bool)

    K = EToE.shape[0]
    Np = ref.rs.shape[0]
    n_faces = 3
    Nf = ref.edge_rules[1].n_points

    face_interp = np.zeros((n_faces, Nf, Np), dtype=float)
    face_weights = np.zeros((n_faces, Nf), dtype=float)

    for face_id in (1, 2, 3):
        f = face_id - 1
        face_interp[f] = np.asarray(ref.face_interp[face_id], dtype=float)
        face_weights[f] = np.asarray(ref.edge_rules[face_id].weights, dtype=float)

    cache = TraceCache(
        n_elements=K,
        n_points=Np,
        n_faces=n_faces,
        n_face_points=Nf,
        face_interp=face_interp,
        face_weights=face_weights,
        neighbor_elements=EToE,
        neighbor_faces=EToF,
        is_boundary=is_boundary,
        face_flip=face_flip,
    )

    if validate:
        validate_trace_cache(cache)

    return cache


def validate_trace_cache(cache: TraceCache) -> None:
    K = cache.n_elements
    Np = cache.n_points
    n_faces = cache.n_faces
    Nf = cache.n_face_points

    if cache.face_interp.shape != (n_faces, Nf, Np):
        raise ValueError("face_interp must have shape (3, Nf, Np).")

    if cache.face_weights.shape != (n_faces, Nf):
        raise ValueError("face_weights must have shape (3, Nf).")

    if cache.neighbor_elements.shape != (K, n_faces):
        raise ValueError("neighbor_elements must have shape (K, 3).")

    if cache.neighbor_faces.shape != (K, n_faces):
        raise ValueError("neighbor_faces must have shape (K, 3).")

    if cache.is_boundary.shape != (K, n_faces):
        raise ValueError("is_boundary must have shape (K, 3).")

    if cache.face_flip.shape != (K, n_faces):
        raise ValueError("face_flip must have shape (K, 3).")

    if np.any(cache.face_weights <= 0.0):
        raise ValueError("face_weights must be positive.")


def evaluate_face_traces(
    q: np.ndarray,
    trace: TraceCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    q = np.asarray(q, dtype=float)

    if q.shape != (trace.n_elements, trace.n_points):
        raise ValueError(f"q must have shape {(trace.n_elements, trace.n_points)}.")

    if out is None:
        qM = np.empty((trace.n_elements, trace.n_faces, trace.n_face_points), dtype=float)
    else:
        qM = np.asarray(out, dtype=float)
        if qM.shape != (trace.n_elements, trace.n_faces, trace.n_face_points):
            raise ValueError("out has wrong shape.")

    if _should_use_numba(use_numba):
        _evaluate_faces_kernel(
            q,
            trace.face_interp,
            qM,
        )
        return qM

    for f in range(trace.n_faces):
        qM[:, f, :] = q @ trace.face_interp[f].T

    return qM


def gather_neighbor_traces(
    qM: np.ndarray,
    trace: TraceCache,
    boundary_value: float = np.nan,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    qM = np.asarray(qM, dtype=float)

    expected = (trace.n_elements, trace.n_faces, trace.n_face_points)

    if qM.shape != expected:
        raise ValueError(f"qM must have shape {expected}.")

    if out is None:
        qP = np.empty_like(qM)
    else:
        qP = np.asarray(out, dtype=float)
        if qP.shape != expected:
            raise ValueError("out has wrong shape.")

    if _should_use_numba(use_numba):
        _gather_neighbor_faces_kernel(
            qM,
            trace.neighbor_elements,
            trace.neighbor_faces,
            trace.is_boundary,
            trace.face_flip,
            float(boundary_value),
            qP,
        )
        return qP

    qP.fill(float(boundary_value))

    interior = ~trace.is_boundary

    if np.any(interior):
        k_idx, f_idx = np.where(interior)
        nbr = trace.neighbor_elements[k_idx, f_idx]
        nbr_f = trace.neighbor_faces[k_idx, f_idx]

        qP[k_idx, f_idx, :] = qM[nbr, nbr_f, :]

        flip_mask = trace.face_flip[k_idx, f_idx]

        if np.any(flip_mask):
            kf = k_idx[flip_mask]
            ff = f_idx[flip_mask]
            qP[kf, ff, :] = qP[kf, ff, ::-1]

    return qP


def pair_face_traces(
    q: np.ndarray,
    trace: TraceCache,
    boundary_value: float = np.nan,
    use_numba: bool | None = None,
) -> FaceTraces:
    qM = evaluate_face_traces(q, trace, use_numba=use_numba)
    qP = gather_neighbor_traces(
        qM,
        trace,
        boundary_value=boundary_value,
        use_numba=use_numba,
    )

    return FaceTraces(qM=qM, qP=qP)


def interior_trace_mismatch(
    traces: FaceTraces,
    trace: TraceCache,
) -> np.ndarray:
    qM = np.asarray(traces.qM, dtype=float)
    qP = np.asarray(traces.qP, dtype=float)

    interior = ~trace.is_boundary

    if not np.any(interior):
        return np.zeros(0, dtype=float)

    return np.abs(qM[interior] - qP[interior])


def max_interior_trace_mismatch(
    traces: FaceTraces,
    trace: TraceCache,
) -> float:
    mismatch = interior_trace_mismatch(traces, trace)

    if mismatch.size == 0:
        return 0.0

    return float(np.max(mismatch))


def check_constant_trace_consistency(
    trace: TraceCache,
    value: float = 1.0,
    use_numba: bool | None = None,
) -> float:
    q = np.full((trace.n_elements, trace.n_points), float(value), dtype=float)
    traces = pair_face_traces(q, trace, use_numba=use_numba)
    return max_interior_trace_mismatch(traces, trace)