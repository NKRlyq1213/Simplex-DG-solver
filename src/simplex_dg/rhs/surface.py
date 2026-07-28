from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplex_dg.geometry import GeometryCache
from simplex_dg.reference import ReferenceCache
from simplex_dg.rhs.volume import VolumeRHSCache, project_to_tangent
from simplex_dg.trace import FaceTraces, TraceCache, gather_neighbor_traces


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
    face_interp: np.ndarray
    sqrt_g: np.ndarray

    face_jacobian: np.ndarray
    face_velocity: np.ndarray
    normal_velocity: np.ndarray

    flux_type: str
    flux_id: int
    lf_alpha: float
def flux_id_from_name(flux_type: str) -> int:
    key = flux_type.lower().strip()

    if key not in _FLUX_TO_ID:
        raise ValueError("flux_type must be 'central', 'upwind', or 'lf'.")

    return _FLUX_TO_ID[key]


def build_lift_matrices(ref: ReferenceCache, trace: TraceCache) -> np.ndarray:
    """Copy the variant-selected face lift from the reference cache."""
    lift = np.zeros(
        (trace.n_faces, trace.n_points, trace.n_face_points),
        dtype=float,
    )

    for face_id in (1, 2, 3):
        f = face_id - 1
        lift[f] = np.asarray(ref.face_lift[face_id], dtype=float)

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
    lf_alpha: float = 1.0,
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
        face_interp=np.asarray(trace.face_interp, dtype=float),
        sqrt_g=np.asarray(geom.sqrt_g, dtype=float),
        face_jacobian=np.asarray(geom.face_jacobian, dtype=float),
        face_velocity=face_velocity,
        normal_velocity=normal_velocity,
        flux_type=flux_type.lower().strip(),
        flux_id=flux_id_from_name(flux_type),
        lf_alpha=float(lf_alpha),
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

    if cache.face_interp.shape != (n_faces, Nf, Np):
        raise ValueError("face_interp must have shape (3, Nf, Np).")

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

    #if tangent_error > tol:
    #    raise ValueError(f"face_velocity is not tangent: max error = {tangent_error}.")

    if cache.flux_id not in (0, 1, 2):
        raise ValueError("Invalid flux_id.")

    if cache.lf_alpha < 0.0:
        raise ValueError("lf_alpha must be non-negative.")


def numerical_flux(
    qM: np.ndarray,
    qP: np.ndarray,
    normal_velocity: np.ndarray,
    flux_id: int,
    lf_alpha: float = 1.0,
) -> np.ndarray:
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float)
    un = np.asarray(normal_velocity, dtype=float)

    if not (qM.shape == qP.shape == un.shape):
        raise ValueError("qM, qP, and normal_velocity must have the same shape.")

    if lf_alpha < 0.0:
        raise ValueError("lf_alpha must be non-negative.")

    if flux_id == 0:
        return 0.5 * un * (qM + qP)

    if flux_id == 1:
        return np.where(un >= 0.0, un * qM, un * qP)

    if flux_id == 2:
        return 0.5 * un * (qM + qP) - 0.5 * float(lf_alpha) * np.abs(un) * (qP - qM)

    raise ValueError("Invalid flux_id.")


# Reference-face derivatives with respect to the edge parameter t in [0, 1].
# These match reference.quadrature.reference_edge_nodes and geometry.sphere._face_direction_rs.
_FACE_DRDT = np.array([-2.0, 0.0, 2.0], dtype=float)
_FACE_DSDT = np.array([2.0, -2.0, 0.0], dtype=float)


def projected_interior_line_flux(
    q: np.ndarray,
    volume: VolumeRHSCache,
    cache: SurfaceRHSCache,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the projected interior physical line flux F_{n,P}^-.

    The volume derivative differentiates alpha*q and beta*q with SDG
    derivative matrices. Therefore the compatible boundary term is

        F_{n,P}^- = ds/dt * E P(alpha*q) - dr/dt * E P(beta*q),

    not

        face_jacobian * normal_velocity * E P(q).

    This function intentionally projects the physical volume fluxes
    alpha*q and beta*q before taking the boundary trace. It does not
    project the penalty residual p.
    """
    q = np.asarray(q, dtype=float)

    expected_vol = (cache.n_elements, cache.n_points)

    if q.shape != expected_vol:
        raise ValueError(f"q must have shape {expected_vol}.")

    if volume.alpha.shape != expected_vol or volume.beta.shape != expected_vol:
        raise ValueError("volume.alpha and volume.beta must match q shape.")

    expected_face = (cache.n_elements, cache.n_faces, cache.n_face_points)

    if out is None:
        line_flux = np.empty(expected_face, dtype=float)
    else:
        line_flux = np.asarray(out, dtype=float)
        if line_flux.shape != expected_face:
            raise ValueError("out has wrong shape.")

    alpha_q = volume.alpha * q
    beta_q = volume.beta * q

    for f in range(cache.n_faces):
        alpha_face = alpha_q @ cache.face_interp[f].T
        beta_face = beta_q @ cache.face_interp[f].T

        drdt = _FACE_DRDT[f]
        dsdt = _FACE_DSDT[f]

        line_flux[:, f, :] = dsdt * alpha_face - drdt * beta_face

    return line_flux


def projected_line_velocity(
    volume: VolumeRHSCache,
    cache: SurfaceRHSCache,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute projected line velocity a_{n,P}.

    a_{n,P} = ds/dt * E P(alpha) - dr/dt * E P(beta).

    The result is oriented with each element's own outward co-normal.
    """
    expected_vol = (cache.n_elements, cache.n_points)

    if volume.alpha.shape != expected_vol or volume.beta.shape != expected_vol:
        raise ValueError("volume.alpha and volume.beta must match cache shape.")

    expected_face = (cache.n_elements, cache.n_faces, cache.n_face_points)

    if out is None:
        line_velocity = np.empty(expected_face, dtype=float)
    else:
        line_velocity = np.asarray(out, dtype=float)
        if line_velocity.shape != expected_face:
            raise ValueError("out has wrong shape.")

    for f in range(cache.n_faces):
        alpha_face = volume.alpha @ cache.face_interp[f].T
        beta_face = volume.beta @ cache.face_interp[f].T

        drdt = _FACE_DRDT[f]
        dsdt = _FACE_DSDT[f]

        line_velocity[:, f, :] = dsdt * alpha_face - drdt * beta_face

    return line_velocity


def common_projected_line_velocity(
    volume: VolumeRHSCache,
    cache: SurfaceRHSCache,
    trace: TraceCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Compute interface-single-valued projected line velocity.

    First compute local outward speed aM. Then gather neighbor outward
    speed aP. Since the neighbor's outward normal is opposite to the
    current element's outward normal, the common minus-side speed is

        a_common = 0.5 * (aM - aP).

    Boundary faces keep the local speed.
    """
    expected_face = (cache.n_elements, cache.n_faces, cache.n_face_points)

    if out is None:
        common = np.empty(expected_face, dtype=float)
    else:
        common = np.asarray(out, dtype=float)
        if common.shape != expected_face:
            raise ValueError("out has wrong shape.")

    aM = projected_line_velocity(volume, cache)

    aP = gather_neighbor_traces(
        aM,
        trace,
        boundary_value=np.nan,
        use_numba=use_numba,
    )

    common[:, :, :] = 0.5 * (aM - aP)

    if np.any(trace.is_boundary):
        common[trace.is_boundary, :] = aM[trace.is_boundary, :]

    return common



def surface_lift_correction_projected_flux(
    q: np.ndarray,
    traces: FaceTraces,
    volume: VolumeRHSCache,
    cache: SurfaceRHSCache,
    trace: TraceCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Surface correction consistent with projected SBP.

    Uses

        p = F_{n,P}^- - F_n^*(q_P^-, q_P^+),

    where q_P plus/minus values are already supplied through FaceTraces and
    F_{n,P}^- is obtained by projecting alpha*q and beta*q to the face.

    The projector is not applied to p.
    """
    qM = np.asarray(traces.qM, dtype=float)
    qP = np.asarray(traces.qP, dtype=float)

    expected_face = (cache.n_elements, cache.n_faces, cache.n_face_points)
    expected_vol = (cache.n_elements, cache.n_points)

    if qM.shape != expected_face or qP.shape != expected_face:
        raise ValueError(f"qM and qP must have shape {expected_face}.")

    if trace.n_elements != cache.n_elements or trace.n_faces != cache.n_faces:
        raise ValueError("trace and surface cache dimensions are inconsistent.")

    if out is None:
        surface = np.empty(expected_vol, dtype=float)
    else:
        surface = np.asarray(out, dtype=float)
        if surface.shape != expected_vol:
            raise ValueError("out has wrong shape.")

    # F_{n,P}^- as a physical line flux, including the reference-face scaling.
    line_flux_m = projected_interior_line_flux(
        q=q,
        volume=volume,
        cache=cache,
    )

    # Numerical flux is represented as a physical line flux.
    # Use the common projected line velocity so both sides of an
    # interface see exactly opposite speeds.
    line_velocity = common_projected_line_velocity(
        volume=volume,
        cache=cache,
        trace=trace,
        use_numba=use_numba,
    )

    flux_star_line = numerical_flux(
        qM=qM,
        qP=qP,
        normal_velocity=line_velocity,
        flux_id=cache.flux_id,
        lf_alpha=cache.lf_alpha,
    )

    correction = line_flux_m - flux_star_line

    surface.fill(0.0)

    for f in range(cache.n_faces):
        surface += correction[:, f, :] @ cache.lift[f].T

    surface /= cache.sqrt_g

    return surface



def surface_lift_correction_split_projected_flux(
    q: np.ndarray,
    traces: FaceTraces,
    volume: VolumeRHSCache,
    cache: SurfaceRHSCache,
    trace: TraceCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Surface correction compatible with split-form volume derivative.

    Split volume uses

        0.5 D(alpha*q) + 0.5 alpha Dq + 0.5 q D(alpha)

    plus the beta-direction analogue.  The matching projected interior
    boundary line flux is

        F_split^-
        =
        0.5 F_cons^-
        + 0.5 a_{n,P}^- q_P^-,

    where

        F_cons^- = ds/dt EP(alpha*q) - dr/dt EP(beta*q),
        a_{n,P}^- = ds/dt EP(alpha) - dr/dt EP(beta),
        q_P^- = EPq.

    The numerical flux uses common projected line velocity.
    The penalty residual itself is not projected.
    """
    qM = np.asarray(traces.qM, dtype=float)
    qP = np.asarray(traces.qP, dtype=float)

    expected_face = (cache.n_elements, cache.n_faces, cache.n_face_points)
    expected_vol = (cache.n_elements, cache.n_points)

    if qM.shape != expected_face or qP.shape != expected_face:
        raise ValueError(f"qM and qP must have shape {expected_face}.")

    if trace.n_elements != cache.n_elements or trace.n_faces != cache.n_faces:
        raise ValueError("trace and surface cache dimensions are inconsistent.")

    if out is None:
        surface = np.empty(expected_vol, dtype=float)
    else:
        surface = np.asarray(out, dtype=float)
        if surface.shape != expected_vol:
            raise ValueError("out has wrong shape.")

    line_flux_cons = projected_interior_line_flux(
        q=q,
        volume=volume,
        cache=cache,
    )

    line_velocity_m = projected_line_velocity(
        volume=volume,
        cache=cache,
    )

    line_flux_m = 0.5 * line_flux_cons + 0.5 * line_velocity_m * qM

    line_velocity_common = common_projected_line_velocity(
        volume=volume,
        cache=cache,
        trace=trace,
        use_numba=use_numba,
    )

    flux_star_line = numerical_flux(
        qM=qM,
        qP=qP,
        normal_velocity=line_velocity_common,
        flux_id=cache.flux_id,
        lf_alpha=cache.lf_alpha,
    )

    correction = line_flux_m - flux_star_line

    surface.fill(0.0)

    for f in range(cache.n_faces):
        surface += correction[:, f, :] @ cache.lift[f].T

    surface /= cache.sqrt_g

    return surface
