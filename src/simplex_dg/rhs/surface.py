from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np

from simplex_dg.geometry import GeometryCache
from simplex_dg.reference import ReferenceCache
from simplex_dg.reference.basis import vandermonde2d
from simplex_dg.rhs.volume import project_to_tangent, solid_body_rotation_velocity
from simplex_dg.trace import FaceTraces, TraceCache


try:
    _numba = importlib.import_module("numba")
    njit = _numba.njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False


_FLUX_TO_ID = {
    "central": 0,
    "upwind": 1,
    "lf": 2,
    "lax_friedrichs": 2,
    "lax-friedrichs": 2,
}


@dataclass(frozen=True)
class SurfaceRHSCache:
    n_elements: int
    n_points: int
    n_faces: int
    n_face_points: int

    lift: np.ndarray
    sqrt_g: np.ndarray

    face_jacobian: np.ndarray
    face_velocity: np.ndarray
    normal_velocity: np.ndarray

    flux_type: str
    flux_id: int


def _should_use_numba(use_numba: bool | None) -> bool:
    if use_numba is None:
        return _NUMBA_AVAILABLE
    return bool(use_numba) and _NUMBA_AVAILABLE


def flux_id_from_name(flux_type: str) -> int:
    key = flux_type.lower().strip()

    if key not in _FLUX_TO_ID:
        raise ValueError("flux_type must be 'central', 'upwind', or 'lf'.")

    return _FLUX_TO_ID[key]


def build_lift_matrices(ref: ReferenceCache, trace: TraceCache) -> np.ndarray:
    lift = np.zeros(
        (trace.n_faces, trace.n_points, trace.n_face_points),
        dtype=float,
    )

    for face_id in (1, 2, 3):
        f = face_id - 1
        edge = ref.edge_rules[face_id]
        V_face = vandermonde2d(ref.order, edge.rs[:, 0], edge.rs[:, 1])

        # Maps face quadrature values to volume nodal values:
        #
        # nodal_lift = V_volume M^{-1} V_face^T W_face
        #
        # Geometry-dependent face_jacobian is not included here. It is applied
        # at runtime because it is element-dependent.
        lift[f] = (ref.V @ ref.Minv @ V_face.T) * edge.weights[None, :]

    return lift


def compute_face_velocity(
    geom: GeometryCache,
    velocity_face: np.ndarray | None = None,
    omega: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    project_velocity: bool = True,
) -> np.ndarray:
    if velocity_face is None:
        omega_arr = np.asarray(omega, dtype=float).reshape(3)
        u = np.cross(omega_arr, geom.X_face)
    else:
        u = np.asarray(velocity_face, dtype=float)

    if u.shape != geom.X_face.shape:
        raise ValueError("velocity_face must have shape (K, 3, Nf, 3).")

    if project_velocity:
        u = project_to_tangent(u, geom.face_normal)

    return u


def build_surface_rhs_cache(
    ref: ReferenceCache,
    geom: GeometryCache,
    trace: TraceCache,
    velocity_face: np.ndarray | None = None,
    omega: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    flux_type: str = "upwind",
    project_velocity: bool = True,
    validate: bool = True,
) -> SurfaceRHSCache:
    lift = build_lift_matrices(ref, trace)

    face_velocity = compute_face_velocity(
        geom=geom,
        velocity_face=velocity_face,
        omega=omega,
        project_velocity=project_velocity,
    )

    normal_velocity = np.sum(face_velocity * geom.face_conormal, axis=3)

    cache = SurfaceRHSCache(
        n_elements=trace.n_elements,
        n_points=trace.n_points,
        n_faces=trace.n_faces,
        n_face_points=trace.n_face_points,
        lift=lift,
        sqrt_g=np.asarray(geom.sqrt_g, dtype=float),
        face_jacobian=np.asarray(geom.face_jacobian, dtype=float),
        face_velocity=face_velocity,
        normal_velocity=normal_velocity,
        flux_type=flux_type.lower().strip(),
        flux_id=flux_id_from_name(flux_type),
    )

    if validate:
        validate_surface_rhs_cache(cache, geom, trace)

    return cache


def validate_surface_rhs_cache(
    cache: SurfaceRHSCache,
    geom: GeometryCache,
    trace: TraceCache,
    tol: float = 1e-10,
) -> None:
    K = cache.n_elements
    Np = cache.n_points
    n_faces = cache.n_faces
    Nf = cache.n_face_points

    if cache.lift.shape != (n_faces, Np, Nf):
        raise ValueError("lift must have shape (3, Np, Nf).")

    if cache.sqrt_g.shape != (K, Np):
        raise ValueError("sqrt_g must have shape (K, Np).")

    if cache.face_jacobian.shape != (K, n_faces, Nf):
        raise ValueError("face_jacobian must have shape (K, 3, Nf).")

    if cache.face_velocity.shape != (K, n_faces, Nf, 3):
        raise ValueError("face_velocity must have shape (K, 3, Nf, 3).")

    if cache.normal_velocity.shape != (K, n_faces, Nf):
        raise ValueError("normal_velocity must have shape (K, 3, Nf).")

    if np.any(cache.sqrt_g <= 0.0):
        raise ValueError("sqrt_g must be positive.")

    if np.any(cache.face_jacobian <= 0.0):
        raise ValueError("face_jacobian must be positive.")

    tangent_error = np.max(np.abs(np.sum(cache.face_velocity * geom.face_normal, axis=3)))

    if tangent_error > tol:
        raise ValueError(f"face_velocity is not tangent: max error = {tangent_error}.")

    if cache.flux_id not in (0, 1, 2):
        raise ValueError("Invalid flux_id.")


def numerical_flux(
    qM: np.ndarray,
    qP: np.ndarray,
    normal_velocity: np.ndarray,
    flux_id: int,
) -> np.ndarray:
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float)
    un = np.asarray(normal_velocity, dtype=float)

    if not (qM.shape == qP.shape == un.shape):
        raise ValueError("qM, qP, and normal_velocity must have the same shape.")

    if flux_id == 0:
        return 0.5 * un * (qM + qP)

    if flux_id == 1:
        return np.where(un >= 0.0, un * qM, un * qP)

    if flux_id == 2:
        return 0.5 * un * (qM + qP) - 0.5 * np.abs(un) * (qP - qM)

    raise ValueError("Invalid flux_id.")


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _surface_lift_correction_kernel(
        qM,
        qP,
        lift,
        sqrt_g,
        face_jacobian,
        normal_velocity,
        flux_id,
        out,
    ):
        K = qM.shape[0]
        n_faces = qM.shape[1]
        Nf = qM.shape[2]
        Np = sqrt_g.shape[1]

        for k in range(K):
            for i in range(Np):
                acc = 0.0

                for f in range(n_faces):
                    for j in range(Nf):
                        un = normal_velocity[k, f, j]
                        qm = qM[k, f, j]
                        qp = qP[k, f, j]

                        flux_m = un * qm

                        if flux_id == 0:
                            flux_star = 0.5 * un * (qm + qp)
                        elif flux_id == 1:
                            if un >= 0.0:
                                flux_star = un * qm
                            else:
                                flux_star = un * qp
                        else:
                            flux_star = 0.5 * un * (qm + qp) - 0.5 * abs(un) * (qp - qm)

                        correction = flux_m - flux_star
                        acc += lift[f, i, j] * face_jacobian[k, f, j] * correction

                out[k, i] = acc / sqrt_g[k, i]

else:
    _surface_lift_correction_kernel = None


def surface_lift_correction(
    traces: FaceTraces,
    cache: SurfaceRHSCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    qM = np.asarray(traces.qM, dtype=float)
    qP = np.asarray(traces.qP, dtype=float)

    expected_face = (cache.n_elements, cache.n_faces, cache.n_face_points)
    expected_vol = (cache.n_elements, cache.n_points)

    if qM.shape != expected_face or qP.shape != expected_face:
        raise ValueError(f"qM and qP must have shape {expected_face}.")

    if out is None:
        surface = np.empty(expected_vol, dtype=float)
    else:
        surface = np.asarray(out, dtype=float)
        if surface.shape != expected_vol:
            raise ValueError("out has wrong shape.")

    if _should_use_numba(use_numba):
        _surface_lift_correction_kernel(
            qM,
            qP,
            cache.lift,
            cache.sqrt_g,
            cache.face_jacobian,
            cache.normal_velocity,
            cache.flux_id,
            surface,
        )
        return surface

    flux_m = cache.normal_velocity * qM
    flux_star = numerical_flux(
        qM=qM,
        qP=qP,
        normal_velocity=cache.normal_velocity,
        flux_id=cache.flux_id,
    )

    correction = cache.face_jacobian * (flux_m - flux_star)

    surface.fill(0.0)

    for f in range(cache.n_faces):
        surface += correction[:, f, :] @ cache.lift[f].T

    surface /= cache.sqrt_g

    return surface