from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplex_dg.geometry import GeometryCache
from simplex_dg.reference import ReferenceCache
from simplex_dg.rhs.surface import (
    SurfaceRHSCache,
    build_surface_rhs_cache,
    surface_lift_correction_projected_flux,
    surface_lift_correction_split_projected_flux,
)
from simplex_dg.rhs.volume import (
    VolumeRHSCache,
    build_volume_rhs_cache,
    volume_divergence_conservative,
    volume_divergence_split,
)
from simplex_dg.trace import TraceCache, pair_face_traces


@dataclass(frozen=True)
class FullRHSCache:
    volume: VolumeRHSCache
    surface: SurfaceRHSCache
    trace: TraceCache
    volume_form: str


def build_full_rhs_cache(
    ref: ReferenceCache,
    geom: GeometryCache,
    trace: TraceCache,
    velocity_volume: np.ndarray | None = None,
    velocity_face: np.ndarray | None = None,
    omega: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    flux_type: str = "upwind",
    lf_alpha: float = 1.0,
    project_velocity: bool = True,
    volume_form: str = "conservative",
    validate: bool = True,
) -> FullRHSCache:
    volume_form = volume_form.lower().strip()

    if volume_form in ("conservative", "cons", "divergence"):
        volume_form = "conservative"
    elif volume_form in ("split", "split_form", "skew"):
        volume_form = "split"
    else:
        raise ValueError("volume_form must be 'conservative' or 'split'.")

    volume = build_volume_rhs_cache(
        ref=ref,
        geom=geom,
        velocity=velocity_volume,
        omega=omega,
        project_velocity=project_velocity,
        validate=validate,
    )

    surface = build_surface_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        velocity_face=velocity_face,
        omega=omega,
        flux_type=flux_type,
        lf_alpha=lf_alpha,
        project_velocity=project_velocity,
        validate=validate,
    )

    return FullRHSCache(
        volume=volume,
        surface=surface,
        trace=trace,
        volume_form=volume_form,
    )


def full_rhs(
    q: np.ndarray,
    cache: FullRHSCache,
    out: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Evaluate the full semi-discrete RHS for the selected volume form."""
    q = np.asarray(q, dtype=float)

    expected = (cache.volume.n_elements, cache.volume.n_points)

    if q.shape != expected:
        raise ValueError(f"q must have shape {expected}.")

    if cache.volume_form == "split":
        div = volume_divergence_split(q, cache.volume, use_numba=use_numba)
    elif cache.volume_form == "conservative":
        div = volume_divergence_conservative(q, cache.volume, use_numba=use_numba)
    else:
        raise ValueError("cache.volume_form must be 'conservative' or 'split'.")

    traces = pair_face_traces(
        q,
        cache.trace,
        use_numba=use_numba,
    )

    if cache.volume_form == "split":
        surf = surface_lift_correction_split_projected_flux(
            q,
            traces,
            cache.volume,
            cache.surface,
            cache.trace,
            use_numba=use_numba,
        )
    elif cache.volume_form == "conservative":
        surf = surface_lift_correction_projected_flux(
            q,
            traces,
            cache.volume,
            cache.surface,
            cache.trace,
            use_numba=use_numba,
        )
    else:
        raise ValueError("cache.volume_form must be 'conservative' or 'split'.")

    if out is None:
        rhs = np.empty_like(q)
    else:
        rhs = np.asarray(out, dtype=float)
        if rhs.shape != expected:
            raise ValueError("out has wrong shape.")

    rhs[:, :] = -div + surf

    return rhs


full_rhs_split = full_rhs
