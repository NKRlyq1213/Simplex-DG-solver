from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.spatial import Delaunay

from simplex_dg.geometry import (
    GeometryCache,
    build_geometry_cache,
    map_reference_to_sphere_element,
)
from simplex_dg.mesh import ManifoldMesh, build_octa_sphere_mesh, normalize_vectors
from simplex_dg.problems import (
    exact_gaussian_solid_body,
    gaussian_center_solid_body,
    gaussian_on_sphere,
    normalize_vector,
    rodrigues_rotate,
)
from simplex_dg.reference import ReferenceCache, build_reference_cache
from simplex_dg.rhs.volume import solid_body_rotation_velocity


EARTH_RADIUS_METERS = 6.371e6
SECONDS_PER_DAY = 86400.0
DEFAULT_POSTER_AMPLITUDE = 1000.0
DEFAULT_ARROW_RADIUS_OFFSET = 0.055
SURFACE_ARROW_RADIUS_OFFSET = 0.0
DEFAULT_CAMERA_AZIMUTH = -48.30186567443501
DEFAULT_CAMERA_ELEVATION = 25.414597789659304
DEFAULT_CAMERA_DISTANCE = 4.077683165720456
DEFAULT_CAMERA_ROLL = 0.0
DEFAULT_CAMERA_POSITION = (
    (2.45, -2.75, 1.75),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)

GYROR_COLORS = [
    (0.00, "#00a65a"),
    (0.40, "#ffd24d"),
    (0.70, "#ff8c3a"),
    (1.00, "#d62728"),
]

_REFERENCE_VERTICES_RS = np.array(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PosterSphereFields:
    mesh: ManifoldMesh
    ref: ReferenceCache
    geom: GeometryCache
    radius: float
    sigma: float
    amplitude: float
    omega: np.ndarray
    omega_norm: float
    period: float
    t_half: float
    t_plot: float
    plot_days: float
    center0: np.ndarray
    center_half: np.ndarray
    center_plot: np.ndarray
    q0: np.ndarray
    q_half: np.ndarray
    q_plot: np.ndarray
    sanity: dict[str, float]


@dataclass(frozen=True)
class VelocityArrowSample:
    physical_points: np.ndarray
    plot_starts: np.ndarray
    directions: np.ndarray
    velocity: np.ndarray
    speed: np.ndarray
    speed_fraction: np.ndarray
    tangent_relative_error: float


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista is required for interactive poster rendering. "
            "Install it with `pip install pyvista` or use `--check-only` "
            "to run the geometry and field checks without rendering."
        ) from exc

    return pv


def resolve_sigma_physical(
    *,
    radius: float,
    sigma_angle: float,
    sigma_physical: float | None = None,
) -> float:
    radius = float(radius)
    sigma_angle = float(sigma_angle)

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if sigma_physical is None:
        sigma = radius * sigma_angle
    else:
        sigma = float(sigma_physical)

    if sigma <= 0.0:
        raise ValueError("sigma_physical must be positive.")

    return sigma


def build_gyror_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("gyror", GYROR_COLORS)


def resolve_colormap(colormap: str | LinearSegmentedColormap):
    if isinstance(colormap, str) and colormap.lower().strip() in ("gyror", "poster"):
        return build_gyror_colormap()

    return colormap


def scalar_color_from_colormap(
    value: float,
    *,
    amplitude: float = DEFAULT_POSTER_AMPLITUDE,
    colormap: str | LinearSegmentedColormap = "gyror",
) -> str:
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive.")

    cmap = resolve_colormap(colormap)

    if isinstance(cmap, str):
        cmap = colormaps[cmap]

    fraction = float(np.clip(float(value) / float(amplitude), 0.0, 1.0))
    rgba = cmap(fraction)
    rgb = tuple(int(round(255.0 * float(channel))) for channel in rgba[:3])

    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def format_scalar_values(values: Sequence[float]) -> str:
    out: list[str] = []

    for value in values:
        value_f = float(value)

        if abs(value_f - round(value_f)) < 1.0e-10 * max(1.0, abs(value_f)):
            out.append(f"{value_f:.0f}")
        else:
            out.append(f"{value_f:.3g}")

    return ", ".join(out)


def build_exact_fields(
    *,
    ndiv: int = 4,
    order: int = 4,
    table: str = "table1",
    radius: float = EARTH_RADIUS_METERS,
    sigma: float | None = None,
    amplitude: float = DEFAULT_POSTER_AMPLITUDE,
    omega: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    center0: np.ndarray | tuple[float, float, float] | None = None,
    t_plot: float | None = None,
    plot_days: float | None = None,
    validate: bool = True,
) -> PosterSphereFields:
    """Build physical exact fields on the existing SDG sphere geometry.

    ``radius`` and ``sigma`` are physical lengths. Plot normalization happens
    later and never changes the field evaluation coordinates.
    """
    if ndiv < 1:
        raise ValueError("ndiv must be >= 1.")

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive.")

    if sigma is None:
        sigma = radius * 0.35

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    omega_arr = np.asarray(omega, dtype=float).reshape(3)
    omega_norm = float(np.linalg.norm(omega_arr))

    if omega_norm <= 0.0:
        raise ValueError("omega must be nonzero to define the half period.")

    center0_arr = np.array([radius, 0.0, 0.0], dtype=float) if center0 is None else np.asarray(center0, dtype=float)
    center0_arr = normalize_vector(center0_arr, radius=radius)

    ref = build_reference_cache(order=order, table=table, validate=validate)
    mesh = build_octa_sphere_mesh(ndivs=ndiv, radius=radius)
    geom = build_geometry_cache(mesh, ref, validate=validate)

    period = 2.0 * np.pi / omega_norm
    t_half = np.pi / omega_norm
    t_plot_f = t_half if t_plot is None else float(t_plot)

    if t_plot_f < 0.0:
        raise ValueError("t_plot must be nonnegative.")

    if plot_days is None:
        plot_days_f = t_plot_f / SECONDS_PER_DAY
    else:
        plot_days_f = float(plot_days)

    if plot_days_f < 0.0:
        raise ValueError("plot_days must be nonnegative.")

    center_half = gaussian_center_solid_body(
        t=t_half,
        radius=radius,
        center0=center0_arr,
        omega=omega_arr,
    )
    center_plot = gaussian_center_solid_body(
        t=t_plot_f,
        radius=radius,
        center0=center0_arr,
        omega=omega_arr,
    )

    q0 = gaussian_on_sphere(
        X=geom.X,
        center=center0_arr,
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
    )
    q_half = exact_gaussian_solid_body(
        X=geom.X,
        t=t_half,
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
        center0=center0_arr,
        omega=omega_arr,
    )
    q_plot = exact_gaussian_solid_body(
        X=geom.X,
        t=t_plot_f,
        radius=radius,
        sigma=sigma,
        amplitude=amplitude,
        center0=center0_arr,
        omega=omega_arr,
    )

    sanity = _sanity_checks(
        mesh=mesh,
        geom=geom,
        q0=q0,
        amplitude=amplitude,
        omega=omega_arr,
        t_half=t_half,
        center0=center0_arr,
        center_half=center_half,
    )

    return PosterSphereFields(
        mesh=mesh,
        ref=ref,
        geom=geom,
        radius=float(radius),
        sigma=float(sigma),
        amplitude=float(amplitude),
        omega=omega_arr,
        omega_norm=omega_norm,
        period=float(period),
        t_half=float(t_half),
        t_plot=float(t_plot_f),
        plot_days=float(plot_days_f),
        center0=center0_arr,
        center_half=center_half,
        center_plot=center_plot,
        q0=q0,
        q_half=q_half,
        q_plot=q_plot,
        sanity=sanity,
    )


def _sanity_checks(
    *,
    mesh: ManifoldMesh,
    geom: GeometryCache,
    q0: np.ndarray,
    amplitude: float,
    omega: np.ndarray,
    t_half: float,
    center0: np.ndarray,
    center_half: np.ndarray,
) -> dict[str, float]:
    radius = float(mesh.radius)

    mesh_radius_error = float(np.max(np.abs(np.linalg.norm(mesh.vertices, axis=1) - radius)))
    node_radius_error = float(np.max(np.abs(np.linalg.norm(geom.X, axis=2) - radius)))
    element_vertex_radius_error = float(
        np.max(np.abs(np.linalg.norm(geom.element_vertices.reshape(-1, 3), axis=1) - radius))
    )

    velocity = solid_body_rotation_velocity(geom.X, omega=omega)
    x_dot_u = np.sum(geom.X * velocity, axis=2)
    tangent_scale = np.linalg.norm(geom.X, axis=2) * np.linalg.norm(velocity, axis=2)
    tangent_scale_max = max(float(np.max(tangent_scale)), np.finfo(float).tiny)

    center_half_expected = normalize_vector(
        rodrigues_rotate(center0, omega=omega, t=t_half),
        radius=radius,
    )

    return {
        "mesh_vertex_radius_error": mesh_radius_error,
        "geometry_node_radius_error": node_radius_error,
        "element_vertex_radius_error": element_vertex_radius_error,
        "q0_max": float(np.max(q0)),
        "q0_max_fraction_of_amplitude": float(np.max(q0) / amplitude),
        "half_period_angle_error": float(abs(np.linalg.norm(omega) * t_half - np.pi)),
        "half_center_rotation_error": float(np.linalg.norm(center_half - center_half_expected)),
        "half_center_radius_error": float(abs(np.linalg.norm(center_half) - radius)),
        "velocity_max_abs_x_dot_u": float(np.max(np.abs(x_dot_u))),
        "velocity_tangent_relative_error": float(np.max(np.abs(x_dot_u)) / tangent_scale_max),
    }


def _unique_rows_preserve_order(points: np.ndarray, decimals: int = 14) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    seen: set[tuple[float, ...]] = set()
    out: list[np.ndarray] = []

    for point in points:
        key = tuple(np.round(point, decimals=decimals))

        if key in seen:
            continue

        seen.add(key)
        out.append(point)

    return np.asarray(out, dtype=float)


def reference_plot_triangulation(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return reference plotting nodes and local triangles for each element."""
    plot_rs = _unique_rows_preserve_order(np.vstack([np.asarray(rs, dtype=float), _REFERENCE_VERTICES_RS]))
    simplices = np.asarray(Delaunay(plot_rs).simplices, dtype=int)

    a = plot_rs[simplices[:, 0]]
    b = plot_rs[simplices[:, 1]]
    c = plot_rs[simplices[:, 2]]
    signed_area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = signed_area2 < 0.0
    simplices[flip, 1], simplices[flip, 2] = simplices[flip, 2].copy(), simplices[flip, 1].copy()

    return plot_rs, simplices


def build_surface_polydata(fields: PosterSphereFields):
    """Build a PyVista surface colored by the selected exact plot-time scalar values."""
    pv = _require_pyvista()
    plot_rs, local_triangles = reference_plot_triangulation(fields.ref.rs)

    all_points: list[np.ndarray] = []
    all_q0: list[np.ndarray] = []
    all_q_half: list[np.ndarray] = []
    all_q_plot: list[np.ndarray] = []
    faces = np.empty((fields.mesh.elements.shape[0] * local_triangles.shape[0], 4), dtype=int)
    face_row = 0

    for k, vertices in enumerate(fields.geom.element_vertices):
        X_phys, _, _ = map_reference_to_sphere_element(
            rs=plot_rs,
            vertices=vertices,
            radius=fields.radius,
        )
        point_offset = k * plot_rs.shape[0]
        all_points.append(X_phys / fields.radius)
        all_q0.append(
            gaussian_on_sphere(
                X=X_phys.reshape(1, -1, 3),
                center=fields.center0,
                radius=fields.radius,
                sigma=fields.sigma,
                amplitude=fields.amplitude,
            ).reshape(-1)
        )
        all_q_half.append(
            exact_gaussian_solid_body(
                X=X_phys.reshape(1, -1, 3),
                t=fields.t_half,
                radius=fields.radius,
                sigma=fields.sigma,
                amplitude=fields.amplitude,
                center0=fields.center0,
                omega=fields.omega,
            ).reshape(-1)
        )
        all_q_plot.append(
            exact_gaussian_solid_body(
                X=X_phys.reshape(1, -1, 3),
                t=fields.t_plot,
                radius=fields.radius,
                sigma=fields.sigma,
                amplitude=fields.amplitude,
                center0=fields.center0,
                omega=fields.omega,
            ).reshape(-1)
        )

        n_local = local_triangles.shape[0]
        faces[face_row : face_row + n_local, 0] = 3
        faces[face_row : face_row + n_local, 1:] = local_triangles + point_offset
        face_row += n_local

    surface = pv.PolyData(np.vstack(all_points), faces.reshape(-1))
    surface["q0"] = np.concatenate(all_q0)
    surface["q_half"] = np.concatenate(all_q_half)
    surface["q_plot"] = np.concatenate(all_q_plot)

    return surface


def _unique_mesh_edges(elements: np.ndarray) -> np.ndarray:
    elements = np.asarray(elements, dtype=int)
    pairs = np.vstack(
        [
            elements[:, [0, 1]],
            elements[:, [1, 2]],
            elements[:, [2, 0]],
        ]
    )
    pairs.sort(axis=1)

    return np.unique(pairs, axis=0)


def build_mesh_edges(
    mesh: ManifoldMesh,
    *,
    edge_samples: int = 24,
    radius_offset: float = 0.004,
):
    pv = _require_pyvista()
    edge_samples = max(2, int(edge_samples))
    edge_pairs = _unique_mesh_edges(mesh.elements)

    points: list[np.ndarray] = []
    lines: list[int] = []
    t = np.linspace(0.0, 1.0, edge_samples)

    for va, vb in edge_pairs:
        p0 = mesh.vertices[int(va)]
        p1 = mesh.vertices[int(vb)]
        chord = (1.0 - t[:, None]) * p0[None, :] + t[:, None] * p1[None, :]
        X_edge = normalize_vectors(chord, radius=mesh.radius)
        X_plot = (1.0 + float(radius_offset)) * X_edge / mesh.radius
        start = len(points)
        points.extend(X_plot)
        lines.extend([edge_samples, *range(start, start + edge_samples)])

    return pv.PolyData(np.asarray(points, dtype=float), lines=np.asarray(lines, dtype=int))


def _iter_polydata_lines(polydata) -> list[np.ndarray]:
    lines = np.asarray(polydata.lines, dtype=int)
    out: list[np.ndarray] = []
    i = 0

    while i < lines.size:
        n = int(lines[i])
        ids = lines[i + 1 : i + 1 + n]

        if n >= 2:
            out.append(ids)

        i += n + 1

    return out


def _point_at_arc_length(points: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
    idx = min(max(idx, 0), points.shape[0] - 2)
    seg_len = cumulative[idx + 1] - cumulative[idx]

    if seg_len <= np.finfo(float).eps:
        return points[idx].copy()

    theta = (distance - cumulative[idx]) / seg_len
    return (1.0 - theta) * points[idx] + theta * points[idx + 1]


def _dedupe_consecutive(points: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    if points.shape[0] <= 1:
        return points

    keep = [0]

    for i in range(1, points.shape[0]):
        if np.linalg.norm(points[i] - points[keep[-1]]) > tol:
            keep.append(i)

    return points[np.asarray(keep, dtype=int)]


def _polyline_interval_points(
    points: np.ndarray,
    cumulative: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    segment_points = [_point_at_arc_length(points, cumulative, start)]
    interior = np.where((cumulative > start) & (cumulative < end))[0]

    for idx in interior:
        segment_points.append(points[int(idx)])

    segment_points.append(_point_at_arc_length(points, cumulative, end))

    return _dedupe_consecutive(np.asarray(segment_points, dtype=float))


def _offset_plot_points(points: np.ndarray, radius_offset: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    norm = np.linalg.norm(points, axis=1, keepdims=True)

    if np.any(norm <= 0.0):
        raise ValueError("Cannot offset points containing the origin.")

    return (1.0 + float(radius_offset)) * points / norm


def make_dashed_polyline(
    polyline,
    *,
    dash_length: float = 0.045,
    gap_length: float = 0.028,
):
    pv = _require_pyvista()

    if dash_length <= 0.0:
        return polyline

    period = float(dash_length) + max(float(gap_length), 0.0)

    if period <= 0.0 or polyline.n_points == 0 or polyline.lines.size == 0:
        return pv.PolyData()

    points = np.asarray(polyline.points, dtype=float)
    out_points: list[np.ndarray] = []
    out_lines: list[int] = []

    for ids in _iter_polydata_lines(polyline):
        line_points = points[ids]
        seg_lengths = np.linalg.norm(np.diff(line_points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = float(cumulative[-1])

        if total <= np.finfo(float).eps:
            continue

        dash_start = 0.0

        while dash_start < total:
            dash_end = min(dash_start + float(dash_length), total)

            if dash_end > dash_start:
                dash_points = _polyline_interval_points(line_points, cumulative, dash_start, dash_end)

                if dash_points.shape[0] >= 2:
                    start_id = len(out_points)
                    out_points.extend(dash_points)
                    out_lines.extend([dash_points.shape[0], *range(start_id, start_id + dash_points.shape[0])])

            dash_start += period

    if not out_points:
        return pv.PolyData()

    return pv.PolyData(
        np.asarray(out_points, dtype=float),
        lines=np.asarray(out_lines, dtype=int),
    )


def build_initial_contours(
    surface,
    *,
    levels: Sequence[float],
    radius_offset: float = 0.01,
    dash_length: float = 0.045,
    gap_length: float = 0.028,
):
    pv = _require_pyvista()
    levels_arr = np.asarray(list(levels), dtype=float)

    if levels_arr.size == 0:
        return pv.PolyData()

    contours = surface.contour(isosurfaces=levels_arr, scalars="q0")

    if contours.n_points == 0:
        return pv.PolyData()

    contours.points = _offset_plot_points(contours.points, radius_offset=radius_offset)

    return make_dashed_polyline(
        contours,
        dash_length=dash_length,
        gap_length=gap_length,
    )


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]

    if count >= n_points:
        return np.arange(n_points, dtype=int)

    selected = np.empty(count, dtype=int)
    selected[0] = int(np.argmax(points[:, 2] + 0.25 * points[:, 0]))
    min_dist2 = np.sum((points - points[selected[0]]) ** 2, axis=1)

    for i in range(1, count):
        selected[i] = int(np.argmax(min_dist2))
        dist2 = np.sum((points - points[selected[i]]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

    return selected


def sample_velocity_arrows(
    fields: PosterSphereFields,
    *,
    arrow_density: int = 80,
    radius_offset: float = DEFAULT_ARROW_RADIUS_OFFSET,
) -> VelocityArrowSample:
    target_count = max(0, int(arrow_density))

    if target_count == 0:
        empty = np.empty((0, 3), dtype=float)
        empty_scalar = np.empty((0,), dtype=float)
        return VelocityArrowSample(
            physical_points=empty,
            plot_starts=empty,
            directions=empty,
            velocity=empty,
            speed=empty_scalar,
            speed_fraction=empty_scalar,
            tangent_relative_error=0.0,
        )

    element_vertices = fields.mesh.vertices[fields.mesh.elements]
    centroids = normalize_vectors(np.mean(element_vertices, axis=1), radius=fields.radius)
    velocity = solid_body_rotation_velocity(centroids[:, None, :], omega=fields.omega)[:, 0, :]
    speed = np.linalg.norm(velocity, axis=1)
    speed_tol = np.finfo(float).eps * max(1.0, fields.omega_norm * fields.radius)
    valid = speed > speed_tol

    centroids = centroids[valid]
    velocity = velocity[valid]
    speed = speed[valid]

    if centroids.shape[0] == 0:
        empty = np.empty((0, 3), dtype=float)
        empty_scalar = np.empty((0,), dtype=float)
        return VelocityArrowSample(
            physical_points=empty,
            plot_starts=empty,
            directions=empty,
            velocity=empty,
            speed=empty_scalar,
            speed_fraction=empty_scalar,
            tangent_relative_error=0.0,
        )

    indices = _farthest_point_indices(centroids / fields.radius, min(target_count, centroids.shape[0]))
    sampled_points = centroids[indices]
    sampled_velocity = velocity[indices]
    sampled_speed = speed[indices]
    directions = sampled_velocity / sampled_speed[:, None]
    plot_starts = (1.0 + float(radius_offset)) * sampled_points / fields.radius
    max_sphere_speed = max(fields.omega_norm * fields.radius, float(np.max(sampled_speed)), np.finfo(float).tiny)
    speed_fraction = np.clip(sampled_speed / max_sphere_speed, 0.0, 1.0)

    x_dot_u = np.sum(sampled_points * sampled_velocity, axis=1)
    tangent_scale = np.linalg.norm(sampled_points, axis=1) * np.linalg.norm(sampled_velocity, axis=1)
    tangent_relative_error = float(np.max(np.abs(x_dot_u) / np.maximum(tangent_scale, np.finfo(float).tiny)))

    return VelocityArrowSample(
        physical_points=sampled_points,
        plot_starts=plot_starts,
        directions=directions,
        velocity=sampled_velocity,
        speed=sampled_speed,
        speed_fraction=speed_fraction,
        tangent_relative_error=tangent_relative_error,
    )


def build_velocity_arrows(
    fields: PosterSphereFields,
    *,
    arrow_density: int = 80,
    arrow_scale: float = 0.085,
    arrow_scale_mode: str = "magnitude",
    arrow_min_scale: float = 0.35,
    arrow_max_scale: float = 1.0,
    arrow_tip_length: float = 0.30,
    arrow_tip_radius: float = 0.045,
    arrow_shaft_radius: float = 0.014,
    radius_offset: float = DEFAULT_ARROW_RADIUS_OFFSET,
):
    pv = _require_pyvista()
    mode = str(arrow_scale_mode).lower().strip()

    if mode not in ("magnitude", "uniform"):
        raise ValueError("arrow_scale_mode must be 'magnitude' or 'uniform'.")

    if arrow_scale <= 0.0:
        raise ValueError("arrow_scale must be positive.")

    if arrow_min_scale < 0.0 or arrow_max_scale <= 0.0 or arrow_max_scale < arrow_min_scale:
        raise ValueError("arrow_min_scale must be >= 0 and arrow_max_scale must be positive and >= arrow_min_scale.")

    if arrow_tip_length <= 0.0 or arrow_tip_radius <= 0.0 or arrow_shaft_radius <= 0.0:
        raise ValueError("arrow_tip_length, arrow_tip_radius, and arrow_shaft_radius must be positive.")

    sample = sample_velocity_arrows(
        fields,
        arrow_density=arrow_density,
        radius_offset=radius_offset,
    )

    if sample.plot_starts.shape[0] == 0:
        return pv.PolyData()

    arrow_points = pv.PolyData(sample.plot_starts)
    arrow_points["direction"] = sample.directions
    arrow_points["speed"] = sample.speed
    arrow_points["speed_fraction"] = sample.speed_fraction
    arrow = pv.Arrow(
        tip_length=float(arrow_tip_length),
        tip_radius=float(arrow_tip_radius),
        shaft_radius=float(arrow_shaft_radius),
        shaft_resolution=12,
        tip_resolution=18,
    )

    if mode == "magnitude":
        arrow_points["glyph_scale"] = float(arrow_min_scale) + (
            float(arrow_max_scale) - float(arrow_min_scale)
        ) * sample.speed_fraction

        return arrow_points.glyph(
            orient="direction",
            scale="glyph_scale",
            factor=float(arrow_scale),
            geom=arrow,
        )

    return arrow_points.glyph(
        orient="direction",
        scale=False,
        factor=float(arrow_scale),
        geom=arrow,
    )


def configure_camera(
    plotter,
    *,
    camera_position: Sequence[Sequence[float]] | None = None,
    zoom: float = 1.05,
    parallel_projection: bool = True,
) -> None:
    if camera_position is None:
        camera_position = DEFAULT_CAMERA_POSITION

    plotter.camera_position = camera_position
    plotter.camera.parallel_projection = bool(parallel_projection)
    plotter.camera.zoom(float(zoom))
    plotter.reset_camera_clipping_range()


def camera_position_from_angles(
    *,
    azimuth: float = DEFAULT_CAMERA_AZIMUTH,
    elevation: float = DEFAULT_CAMERA_ELEVATION,
    distance: float = DEFAULT_CAMERA_DISTANCE,
    roll: float = DEFAULT_CAMERA_ROLL,
    focal_point: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    distance = float(distance)

    if distance <= 0.0:
        raise ValueError("distance must be positive.")

    azimuth_rad = np.deg2rad(float(azimuth))
    elevation_rad = np.deg2rad(float(elevation))
    focal = np.asarray(focal_point, dtype=float).reshape(3)

    direction = np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ],
        dtype=float,
    )
    position = focal + distance * direction

    view_direction = normalize_vector(focal - position, radius=1.0)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)

    if abs(float(np.dot(view_direction, world_up))) > 0.96:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)

    right = normalize_vector(np.cross(view_direction, world_up), radius=1.0)
    view_up = normalize_vector(np.cross(right, view_direction), radius=1.0)

    if float(roll) != 0.0:
        view_up = normalize_vector(
            rodrigues_rotate(view_up, omega=view_direction, t=np.deg2rad(float(roll))),
            radius=1.0,
        )

    return (
        tuple(float(value) for value in position),
        tuple(float(value) for value in focal),
        tuple(float(value) for value in view_up),
    )


def configure_camera_from_angles(
    plotter,
    *,
    azimuth: float = DEFAULT_CAMERA_AZIMUTH,
    elevation: float = DEFAULT_CAMERA_ELEVATION,
    distance: float = DEFAULT_CAMERA_DISTANCE,
    roll: float = DEFAULT_CAMERA_ROLL,
    zoom: float = 1.0,
    parallel_projection: bool = True,
) -> None:
    configure_camera(
        plotter,
        camera_position=camera_position_from_angles(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            roll=roll,
        ),
        zoom=zoom,
        parallel_projection=parallel_projection,
    )


def configure_lighting(plotter) -> None:
    pv = _require_pyvista()
    plotter.remove_all_lights()
    plotter.add_light(
        pv.Light(
            position=(3.0, -4.0, 4.0),
            focal_point=(0.0, 0.0, 0.0),
            intensity=0.72,
            light_type="scene light",
        )
    )
    plotter.add_light(
        pv.Light(
            position=(-2.5, 3.0, 2.0),
            focal_point=(0.0, 0.0, 0.0),
            intensity=0.18,
            light_type="scene light",
        )
    )

    try:
        plotter.enable_anti_aliasing("fxaa")
    except Exception:
        pass


def configure_axes(
    plotter,
    *,
    axes_mode: str = "none",
    axes_label_font_size: int = 12,
    axes_color: str = "#303030",
    axes_length: float = 1.28,
    axes_label_offset: float = 0.08,
    axes_tip_length: float = 0.16,
    axes_tip_radius: float = 0.025,
    axes_shaft_radius: float = 0.0075,
) -> None:
    pv = _require_pyvista()
    mode = str(axes_mode).lower().strip()

    if mode in ("none", "off", "false", "0"):
        return

    if mode not in ("corner", "origin", "scene", "both"):
        raise ValueError("axes_mode must be 'none', 'corner', 'origin', 'scene', or 'both'.")

    if mode in ("corner", "both"):
        plotter.add_axes(
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            color=axes_color,
            line_width=2,
            labels_off=False,
        )

    if mode in ("origin", "scene", "both"):
        axis_specs = (
            ("X", np.array([1.0, 0.0, 0.0], dtype=float)),
            ("Y", np.array([0.0, 1.0, 0.0], dtype=float)),
            ("Z", np.array([0.0, 0.0, 1.0], dtype=float)),
        )
        label_points = []
        labels = []

        for label, direction in axis_specs:
            axis_arrow = pv.Arrow(
                start=(0.0, 0.0, 0.0),
                direction=direction,
                scale=float(axes_length),
                tip_length=float(axes_tip_length),
                tip_radius=float(axes_tip_radius),
                shaft_radius=float(axes_shaft_radius),
                shaft_resolution=12,
                tip_resolution=18,
            )
            plotter.add_mesh(
                axis_arrow,
                color=axes_color,
                smooth_shading=True,
                lighting=False,
            )
            label_points.append((float(axes_length) + float(axes_label_offset)) * direction)
            labels.append(label)

        plotter.add_point_labels(
            np.asarray(label_points, dtype=float),
            labels,
            font_size=int(axes_label_font_size),
            text_color=axes_color,
            shape=None,
            show_points=False,
            always_visible=True,
        )


def resolve_rotation_axis_radii(
    *,
    axis_width: float = 4.0,
    tip_radius: float | None = None,
    shaft_radius: float | None = None,
) -> tuple[float, float]:
    if axis_width <= 0.0:
        raise ValueError("axis_width must be positive.")

    if tip_radius is not None and tip_radius <= 0.0:
        raise ValueError("tip_radius must be positive.")

    if shaft_radius is not None and shaft_radius <= 0.0:
        raise ValueError("shaft_radius must be positive.")

    shaft_radius_resolved = 0.0025 * float(axis_width) if shaft_radius is None else float(shaft_radius)
    tip_radius_resolved = 3.2 * shaft_radius_resolved if tip_radius is None else float(tip_radius)

    return shaft_radius_resolved, tip_radius_resolved


def build_rotation_axis(
    fields: PosterSphereFields,
    *,
    axis_length: float = 1.35,
    axis_width: float = 4.0,
    tip_length: float = 0.16,
    tip_radius: float | None = None,
    shaft_radius: float | None = None,
):
    pv = _require_pyvista()

    if axis_length <= 0.0:
        raise ValueError("axis_length must be positive.")

    if axis_width <= 0.0:
        raise ValueError("axis_width must be positive.")

    if tip_length <= 0.0:
        raise ValueError("tip_length must be positive.")

    direction = fields.omega / fields.omega_norm
    start = -float(axis_length) * direction
    shaft_radius_resolved, tip_radius_resolved = resolve_rotation_axis_radii(
        axis_width=axis_width,
        tip_radius=tip_radius,
        shaft_radius=shaft_radius,
    )

    return pv.Arrow(
        start=start,
        direction=direction,
        scale=2.0 * float(axis_length),
        tip_length=float(tip_length),
        tip_radius=tip_radius_resolved,
        shaft_radius=shaft_radius_resolved,
        shaft_resolution=18,
        tip_resolution=24,
    )


def build_rotation_direction_ring(
    fields: PosterSphereFields,
    *,
    arc_fraction: float = 0.75,
    ring_radius: float = 0.48,
    radius_offset: float = 0.032,
    samples: int = 160,
    cone_height: float = 0.18,
    cone_radius: float = 0.080,
    cone_offset: float = 0.0,
    cone_resolution: int = 32,
):
    pv = _require_pyvista()

    if not (0.0 < arc_fraction <= 1.0):
        raise ValueError("arc_fraction must be in (0, 1].")

    if not (0.0 < ring_radius <= 1.0):
        raise ValueError("ring_radius must be in (0, 1].")

    if radius_offset < 0.0:
        raise ValueError("radius_offset must be nonnegative.")

    if samples < 3:
        raise ValueError("samples must be >= 3.")

    if cone_height <= 0.0 or cone_radius <= 0.0:
        raise ValueError("cone_height and cone_radius must be positive.")

    if cone_offset < 0.0:
        raise ValueError("cone_offset must be nonnegative.")

    if cone_resolution < 3:
        raise ValueError("cone_resolution must be >= 3.")

    axis = fields.omega / fields.omega_norm
    base = normalize_vector(fields.center0, radius=1.0)
    base_parallel = float(np.dot(base, axis))
    base_perp = base - base_parallel * axis
    base_perp_norm = float(np.linalg.norm(base_perp))

    if base_perp_norm <= np.finfo(float).eps:
        candidate = np.array([1.0, 0.0, 0.0], dtype=float)

        if abs(float(np.dot(candidate, axis))) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0], dtype=float)

        e1 = candidate - float(np.dot(candidate, axis)) * axis
        e1 /= np.linalg.norm(e1)
    else:
        e1 = base_perp / base_perp_norm

    e2 = np.cross(axis, e1)
    e2 /= np.linalg.norm(e2)
    parallel_sign = -1.0 if base_parallel < 0.0 else 1.0
    parallel_length = parallel_sign * np.sqrt(max(0.0, 1.0 - float(ring_radius) ** 2))
    def point_at_angle(theta: float) -> np.ndarray:
        return normalize_vector(
            parallel_length * axis + float(ring_radius) * (np.cos(theta) * e1 + np.sin(theta) * e2),
            radius=1.0,
        )

    arc_angle = 2.0 * np.pi * float(arc_fraction)
    angles = np.linspace(0.0, arc_angle, int(samples))
    points_unit = np.asarray([point_at_angle(float(theta)) for theta in angles], dtype=float)
    points_plot = (1.0 + float(radius_offset)) * points_unit
    lines = np.asarray([points_plot.shape[0], *range(points_plot.shape[0])], dtype=int)
    ring = pv.PolyData(points_plot, lines=lines)

    arrow_start = points_plot[-1]
    arrow_direction = np.cross(axis, points_unit[-1])
    arrow_direction_norm = float(np.linalg.norm(arrow_direction))

    if arrow_direction_norm <= np.finfo(float).eps:
        return ring, pv.PolyData()

    arrow_direction /= arrow_direction_norm
    base_center = arrow_start + float(cone_offset) * arrow_direction
    tip = base_center + float(cone_height) * arrow_direction
    basis_candidate = points_unit[-1]
    normal1 = basis_candidate - float(np.dot(basis_candidate, arrow_direction)) * arrow_direction
    normal1_norm = float(np.linalg.norm(normal1))

    if normal1_norm <= np.finfo(float).eps:
        basis_candidate = axis
        normal1 = basis_candidate - float(np.dot(basis_candidate, arrow_direction)) * arrow_direction
        normal1_norm = float(np.linalg.norm(normal1))

    if normal1_norm <= np.finfo(float).eps:
        basis_candidate = np.array([1.0, 0.0, 0.0], dtype=float)

        if abs(float(np.dot(basis_candidate, arrow_direction))) > 0.9:
            basis_candidate = np.array([0.0, 1.0, 0.0], dtype=float)

        normal1 = basis_candidate - float(np.dot(basis_candidate, arrow_direction)) * arrow_direction
        normal1_norm = float(np.linalg.norm(normal1))

    normal1 /= normal1_norm
    normal2 = np.cross(arrow_direction, normal1)
    normal2 /= np.linalg.norm(normal2)
    phis = np.linspace(0.0, 2.0 * np.pi, int(cone_resolution), endpoint=False)
    base_points = np.asarray(
        [base_center + float(cone_radius) * (np.cos(phi) * normal1 + np.sin(phi) * normal2) for phi in phis],
        dtype=float,
    )
    cone_points = np.vstack([tip, base_points])
    faces: list[int] = []

    for i in range(int(cone_resolution)):
        j = (i + 1) % int(cone_resolution)
        faces.extend([3, 0, 1 + i, 1 + j])

    faces.extend([int(cone_resolution), *range(1, int(cone_resolution) + 1)])
    cone = pv.PolyData(cone_points, faces=np.asarray(faces, dtype=int))

    return ring, cone


def resolve_contour_label_position(
    position: str | tuple[float, float],
    *,
    colorbar_position_x: float,
    colorbar_position_y: float,
) -> tuple[str | tuple[float, float], bool]:
    if isinstance(position, str):
        key = position.lower().strip().replace("-", "_")

        if key in ("below_colorbar", "colorbar_below", "below_bar", "bar_below"):
            return (
                max(0.02, float(colorbar_position_x) - 0.055),
                max(0.02, float(colorbar_position_y) - 0.065),
            ), True

        return position, False

    return (float(position[0]), float(position[1])), True


def render_scene(
    fields: PosterSphereFields,
    *,
    contour_levels: Sequence[float] | None = None,
    arrow_density: int = 80,
    arrow_scale: float = 0.085,
    arrow_scale_mode: str = "magnitude",
    arrow_min_scale: float = 0.35,
    arrow_max_scale: float = 1.0,
    arrow_tip_length: float = 0.30,
    arrow_tip_radius: float = 0.045,
    arrow_shaft_radius: float = 0.014,
    arrow_radius_offset: float = DEFAULT_ARROW_RADIUS_OFFSET,
    edge_width: float = 1.0,
    edge_opacity: float = 0.36,
    edge_color: str = "#202020",
    contour_color: str = "#111111",
    contour_color_mode: str = "height",
    contour_width: float = 2.0,
    font_size: int = 12,
    show_contour_label: bool = True,
    contour_label_position: str | tuple[float, float] = "below_colorbar",
    contour_label_font_size: int | None = None,
    axes_mode: str = "none",
    axes_label_font_size: int | None = None,
    axes_color: str = "#303030",
    axes_length: float = 1.28,
    axes_label_offset: float = 0.08,
    axes_tip_length: float = 0.16,
    axes_tip_radius: float = 0.025,
    axes_shaft_radius: float = 0.0075,
    show_rotation_axis: bool = True,
    rotation_axis_length: float = 1.35,
    rotation_axis_color: str = "#d62728",
    rotation_axis_width: float = 4.0,
    rotation_axis_tip_length: float = 0.16,
    rotation_axis_tip_radius: float | None = None,
    rotation_axis_shaft_radius: float | None = None,
    show_rotation_ring: bool = True,
    rotation_ring_fraction: float = 0.75,
    rotation_ring_radius: float = 0.48,
    rotation_ring_radius_offset: float = 0.032,
    rotation_ring_width: float = 4.0,
    rotation_ring_color: str | None = None,
    rotation_ring_samples: int = 160,
    rotation_ring_cone_height: float = 0.18,
    rotation_ring_cone_radius: float = 0.080,
    rotation_ring_cone_offset: float = 0.0,
    rotation_ring_cone_resolution: int = 32,
    dash_length: float = 0.045,
    gap_length: float = 0.028,
    colormap: str = "gyror",
    show_colorbar: bool = True,
    colorbar_height: float = 0.38,
    colorbar_width: float = 0.07,
    colorbar_position_x: float = 0.88,
    colorbar_position_y: float = 0.28,
    colorbar_label_font_size: int | None = None,
    colorbar_title_font_size: int | None = None,
    colorbar_n_labels: int = 5,
    colorbar_format: str = "%.0f",
    enable_lighting: bool = False,
    background: str = "white",
    window_size: tuple[int, int] = (1600, 1200),
    off_screen: bool = False,
):
    pv = _require_pyvista()
    surface = build_surface_polydata(fields)
    edges = build_mesh_edges(fields.mesh)

    if contour_levels is None:
        levels = fields.amplitude * np.asarray([0.2, 0.5, 0.8], dtype=float)
    else:
        levels = np.asarray(contour_levels, dtype=float)

    contour_color_mode_key = str(contour_color_mode).lower().strip()

    if contour_color_mode_key not in ("height", "neutral"):
        raise ValueError("contour_color_mode must be 'height' or 'neutral'.")

    if contour_width <= 0.0:
        raise ValueError("contour_width must be positive.")

    if rotation_axis_width <= 0.0:
        raise ValueError("rotation_axis_width must be positive.")

    if rotation_axis_tip_length <= 0.0:
        raise ValueError("rotation_axis_tip_length must be positive.")

    if rotation_axis_tip_radius is not None and rotation_axis_tip_radius <= 0.0:
        raise ValueError("rotation_axis_tip_radius must be positive.")

    if rotation_axis_shaft_radius is not None and rotation_axis_shaft_radius <= 0.0:
        raise ValueError("rotation_axis_shaft_radius must be positive.")

    if rotation_ring_width <= 0.0:
        raise ValueError("rotation_ring_width must be positive.")

    if not (0.0 < rotation_ring_radius <= 1.0):
        raise ValueError("rotation_ring_radius must be in (0, 1].")

    if rotation_ring_cone_height <= 0.0 or rotation_ring_cone_radius <= 0.0:
        raise ValueError("rotation_ring_cone_height and rotation_ring_cone_radius must be positive.")

    if rotation_ring_cone_offset < 0.0:
        raise ValueError("rotation_ring_cone_offset must be nonnegative.")

    if rotation_ring_cone_resolution < 3:
        raise ValueError("rotation_ring_cone_resolution must be >= 3.")

    contour_meshes = []

    for level in levels:
        contour = build_initial_contours(
            surface,
            levels=[float(level)],
            dash_length=dash_length,
            gap_length=gap_length,
        )

        if contour.n_points == 0:
            continue

        if contour_color_mode_key == "height":
            line_color = scalar_color_from_colormap(
                float(level),
                amplitude=fields.amplitude,
                colormap=colormap,
            )
        else:
            line_color = contour_color

        contour_meshes.append((contour, line_color))

    arrows = build_velocity_arrows(
        fields,
        arrow_density=arrow_density,
        arrow_scale=arrow_scale,
        arrow_scale_mode=arrow_scale_mode,
        arrow_min_scale=arrow_min_scale,
        arrow_max_scale=arrow_max_scale,
        arrow_tip_length=arrow_tip_length,
        arrow_tip_radius=arrow_tip_radius,
        arrow_shaft_radius=arrow_shaft_radius,
        radius_offset=arrow_radius_offset,
    )
    rotation_axis = (
        build_rotation_axis(
            fields,
            axis_length=rotation_axis_length,
            axis_width=rotation_axis_width,
            tip_length=rotation_axis_tip_length,
            tip_radius=rotation_axis_tip_radius,
            shaft_radius=rotation_axis_shaft_radius,
        )
        if show_rotation_axis
        else None
    )
    rotation_ring = None

    if show_rotation_ring:
        rotation_ring = build_rotation_direction_ring(
            fields,
            arc_fraction=rotation_ring_fraction,
            ring_radius=rotation_ring_radius,
            radius_offset=rotation_ring_radius_offset,
            samples=rotation_ring_samples,
            cone_height=rotation_ring_cone_height,
            cone_radius=rotation_ring_cone_radius,
            cone_offset=rotation_ring_cone_offset,
            cone_resolution=rotation_ring_cone_resolution,
        )

    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
    plotter.set_background(background)

    contour_label_font_size_resolved = int(font_size if contour_label_font_size is None else contour_label_font_size)
    axes_label_font_size_resolved = int(font_size if axes_label_font_size is None else axes_label_font_size)
    colorbar_label_font_size_resolved = int(font_size if colorbar_label_font_size is None else colorbar_label_font_size)
    colorbar_title_font_size_resolved = int(
        max(font_size + 1, font_size) if colorbar_title_font_size is None else colorbar_title_font_size
    )

    scalar_bar_args = {
        "title": "scalar value",
        "vertical": True,
        "height": float(colorbar_height),
        "width": float(colorbar_width),
        "position_x": float(colorbar_position_x),
        "position_y": float(colorbar_position_y),
        "label_font_size": colorbar_label_font_size_resolved,
        "title_font_size": colorbar_title_font_size_resolved,
        "n_labels": int(colorbar_n_labels),
        "fmt": str(colorbar_format),
    }

    plotter.add_mesh(
        surface,
        scalars="q_plot",
        cmap=resolve_colormap(colormap),
        clim=(0.0, fields.amplitude),
        smooth_shading=bool(enable_lighting),
        lighting=bool(enable_lighting),
        ambient=0.44 if enable_lighting else 1.0,
        diffuse=0.58 if enable_lighting else 0.0,
        specular=0.04 if enable_lighting else 0.0,
        show_scalar_bar=show_colorbar,
        scalar_bar_args=scalar_bar_args if show_colorbar else None,
    )
    plotter.add_mesh(
        edges,
        color=edge_color,
        line_width=float(edge_width),
        opacity=float(edge_opacity),
        lighting=False,
    )

    if rotation_axis is not None and rotation_axis.n_points > 0:
        plotter.add_mesh(
            rotation_axis,
            color=rotation_axis_color,
            smooth_shading=True,
            lighting=False,
        )

    if rotation_ring is not None:
        ring, ring_cone = rotation_ring
        ring_color = rotation_axis_color if rotation_ring_color is None else rotation_ring_color

        if ring.n_points > 0:
            plotter.add_mesh(
                ring,
                color=ring_color,
                line_width=float(rotation_ring_width),
                lighting=False,
            )

        if ring_cone.n_points > 0:
            plotter.add_mesh(
                ring_cone,
                color=ring_color,
                smooth_shading=True,
                lighting=False,
            )

    if contour_meshes:
        for contour, line_color in contour_meshes:
            plotter.add_mesh(
                contour,
                color=line_color,
                line_width=float(contour_width),
                lighting=False,
            )

        if show_contour_label:
            label = f"Dashed initial contours q(t=0): {format_scalar_values(levels)}"
            label_position, label_uses_viewport = resolve_contour_label_position(
                contour_label_position,
                colorbar_position_x=colorbar_position_x,
                colorbar_position_y=colorbar_position_y,
            )
            try:
                plotter.add_text(
                    label,
                    position=label_position,
                    font_size=contour_label_font_size_resolved,
                    color=contour_color,
                    shadow=False,
                    viewport=label_uses_viewport,
                )
            except TypeError:
                plotter.add_text(
                    label,
                    position=label_position,
                    font_size=contour_label_font_size_resolved,
                    color=contour_color,
                    shadow=False,
                )

    if arrows.n_points > 0:
        plotter.add_mesh(
            arrows,
            color="#242424",
            smooth_shading=bool(enable_lighting),
            lighting=bool(enable_lighting),
            ambient=0.38 if enable_lighting else 1.0,
            diffuse=0.55 if enable_lighting else 0.0,
            specular=0.06 if enable_lighting else 0.0,
        )

    if enable_lighting:
        configure_lighting(plotter)

    configure_axes(
        plotter,
        axes_mode=axes_mode,
        axes_label_font_size=axes_label_font_size_resolved,
        axes_color=axes_color,
        axes_length=axes_length,
        axes_label_offset=axes_label_offset,
        axes_tip_length=axes_tip_length,
        axes_tip_radius=axes_tip_radius,
        axes_shaft_radius=axes_shaft_radius,
    )
    configure_camera(plotter)

    return plotter


def save_screenshot(
    plotter,
    output: str | Path,
    *,
    transparent_background: bool = False,
    window_size: tuple[int, int] | None = None,
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if window_size is not None:
        plotter.window_size = window_size

    plotter.screenshot(
        str(output_path),
        transparent_background=bool(transparent_background),
        return_img=False,
    )

    return output_path


def _build_surface_plot_arrays(fields: PosterSphereFields) -> dict[str, np.ndarray]:
    plot_rs, local_triangles = reference_plot_triangulation(fields.ref.rs)

    all_points: list[np.ndarray] = []
    all_q0: list[np.ndarray] = []
    all_q_plot: list[np.ndarray] = []
    triangles = np.empty((fields.mesh.elements.shape[0] * local_triangles.shape[0], 3), dtype=int)
    face_row = 0

    for k, vertices in enumerate(fields.geom.element_vertices):
        X_phys, _, _ = map_reference_to_sphere_element(
            rs=plot_rs,
            vertices=vertices,
            radius=fields.radius,
        )
        point_offset = k * plot_rs.shape[0]
        all_points.append(X_phys / fields.radius)
        all_q0.append(
            gaussian_on_sphere(
                X=X_phys.reshape(1, -1, 3),
                center=fields.center0,
                radius=fields.radius,
                sigma=fields.sigma,
                amplitude=fields.amplitude,
            ).reshape(-1)
        )
        all_q_plot.append(
            exact_gaussian_solid_body(
                X=X_phys.reshape(1, -1, 3),
                t=fields.t_plot,
                radius=fields.radius,
                sigma=fields.sigma,
                amplitude=fields.amplitude,
                center0=fields.center0,
                omega=fields.omega,
            ).reshape(-1)
        )

        n_local = local_triangles.shape[0]
        triangles[face_row : face_row + n_local] = local_triangles + point_offset
        face_row += n_local

    return {
        "points": np.vstack(all_points),
        "triangles": triangles,
        "q0": np.concatenate(all_q0),
        "q_plot": np.concatenate(all_q_plot),
    }


def _hex_to_rgb01(color: str) -> tuple[float, float, float]:
    text = str(color).strip()

    if not text.startswith("#") or len(text) != 7:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}.")

    return (
        int(text[1:3], 16) / 255.0,
        int(text[3:5], 16) / 255.0,
        int(text[5:7], 16) / 255.0,
    )


def _surface_vertex_colors(
    values: np.ndarray,
    *,
    amplitude: float,
    colormap: str,
) -> np.ndarray:
    colors = [
        _hex_to_rgb01(scalar_color_from_colormap(float(value), amplitude=amplitude, colormap=colormap))
        for value in np.asarray(values, dtype=float)
    ]

    return np.asarray(colors, dtype=float)


def _line_segment_points_from_polylines(polylines: Sequence[np.ndarray]) -> np.ndarray:
    segments: list[np.ndarray] = []

    for line in polylines:
        points = np.asarray(line, dtype=float)

        if points.shape[0] < 2:
            continue

        for i in range(points.shape[0] - 1):
            segments.append(points[[i, i + 1]])

    if not segments:
        return np.empty((0, 2, 3), dtype=float)

    return np.asarray(segments, dtype=float)


def _mesh_edge_segments(
    mesh: ManifoldMesh,
    *,
    edge_samples: int = 24,
    radius_offset: float = 0.004,
) -> np.ndarray:
    edge_samples = max(2, int(edge_samples))
    edge_pairs = _unique_mesh_edges(mesh.elements)
    t = np.linspace(0.0, 1.0, edge_samples)
    polylines: list[np.ndarray] = []

    for va, vb in edge_pairs:
        p0 = mesh.vertices[int(va)]
        p1 = mesh.vertices[int(vb)]
        chord = (1.0 - t[:, None]) * p0[None, :] + t[:, None] * p1[None, :]
        X_edge = normalize_vectors(chord, radius=mesh.radius)
        polylines.append((1.0 + float(radius_offset)) * X_edge / mesh.radius)

    return _line_segment_points_from_polylines(polylines)


def _level_segment_for_triangle(
    points: np.ndarray,
    values: np.ndarray,
    level: float,
) -> np.ndarray | None:
    crossings: list[np.ndarray] = []

    for ia, ib in ((0, 1), (1, 2), (2, 0)):
        va = float(values[ia])
        vb = float(values[ib])
        da = va - float(level)
        db = vb - float(level)

        if abs(da) <= 1.0e-12 and abs(db) <= 1.0e-12:
            continue

        if abs(da) <= 1.0e-12:
            crossings.append(points[ia])
        elif abs(db) <= 1.0e-12:
            crossings.append(points[ib])
        elif da * db < 0.0:
            theta = (float(level) - va) / (vb - va)
            crossings.append((1.0 - theta) * points[ia] + theta * points[ib])

    if len(crossings) < 2:
        return None

    unique = _unique_rows_preserve_order(np.asarray(crossings, dtype=float), decimals=12)

    if unique.shape[0] < 2:
        return None

    return unique[:2]


def _initial_contour_segments_by_level(
    surface_arrays: dict[str, np.ndarray],
    *,
    levels: Sequence[float],
    radius_offset: float = 0.01,
) -> list[dict[str, np.ndarray | float]]:
    points = np.asarray(surface_arrays["points"], dtype=float)
    triangles = np.asarray(surface_arrays["triangles"], dtype=int)
    q0 = np.asarray(surface_arrays["q0"], dtype=float)
    out: list[dict[str, np.ndarray | float]] = []

    for level in np.asarray(list(levels), dtype=float):
        segments: list[np.ndarray] = []

        for tri in triangles:
            segment = _level_segment_for_triangle(points[tri], q0[tri], float(level))

            if segment is None:
                continue

            segments.append(_offset_plot_points(segment, radius_offset=radius_offset))

        segment_array = np.asarray(segments, dtype=float) if segments else np.empty((0, 2, 3), dtype=float)
        out.append({"level": float(level), "segments": segment_array})

    return out


def _rotation_direction_ring_arrays(
    fields: PosterSphereFields,
    *,
    arc_fraction: float,
    ring_radius: float,
    radius_offset: float,
    samples: int,
    cone_height: float,
    cone_radius: float,
    cone_offset: float,
) -> dict[str, np.ndarray | float]:
    axis = fields.omega / fields.omega_norm
    base = normalize_vector(fields.center0, radius=1.0)
    base_parallel = float(np.dot(base, axis))
    base_perp = base - base_parallel * axis
    base_perp_norm = float(np.linalg.norm(base_perp))

    if base_perp_norm <= np.finfo(float).eps:
        candidate = np.array([1.0, 0.0, 0.0], dtype=float)

        if abs(float(np.dot(candidate, axis))) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0], dtype=float)

        e1 = candidate - float(np.dot(candidate, axis)) * axis
        e1 /= np.linalg.norm(e1)
    else:
        e1 = base_perp / base_perp_norm

    e2 = np.cross(axis, e1)
    e2 /= np.linalg.norm(e2)
    parallel_sign = -1.0 if base_parallel < 0.0 else 1.0
    parallel_length = parallel_sign * np.sqrt(max(0.0, 1.0 - float(ring_radius) ** 2))

    def point_at_angle(theta: float) -> np.ndarray:
        return normalize_vector(
            parallel_length * axis + float(ring_radius) * (np.cos(theta) * e1 + np.sin(theta) * e2),
            radius=1.0,
        )

    arc_angle = 2.0 * np.pi * float(arc_fraction)
    angles = np.linspace(0.0, arc_angle, int(samples))
    points_unit = np.asarray([point_at_angle(float(theta)) for theta in angles], dtype=float)
    points_plot = (1.0 + float(radius_offset)) * points_unit
    arrow_direction = np.cross(axis, points_unit[-1])
    arrow_direction_norm = float(np.linalg.norm(arrow_direction))

    if arrow_direction_norm <= np.finfo(float).eps:
        arrow_direction = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        arrow_direction /= arrow_direction_norm

    base_center = points_plot[-1] + float(cone_offset) * arrow_direction

    return {
        "points": points_plot,
        "cone_position": base_center + 0.5 * float(cone_height) * arrow_direction,
        "cone_direction": arrow_direction,
        "cone_height": float(cone_height),
        "cone_radius": float(cone_radius),
    }


def _camera_json(camera_state: dict[str, float] | None) -> dict[str, float]:
    if camera_state is None:
        return {
            "azimuth": DEFAULT_CAMERA_AZIMUTH,
            "elevation": DEFAULT_CAMERA_ELEVATION,
            "distance": DEFAULT_CAMERA_DISTANCE,
            "roll": DEFAULT_CAMERA_ROLL,
        }

    return {
        "azimuth": float(camera_state.get("azimuth", DEFAULT_CAMERA_AZIMUTH)),
        "elevation": float(camera_state.get("elevation", DEFAULT_CAMERA_ELEVATION)),
        "distance": float(camera_state.get("distance", DEFAULT_CAMERA_DISTANCE)),
        "roll": float(camera_state.get("roll", DEFAULT_CAMERA_ROLL)),
    }


def _flatten_points(points: np.ndarray, *, precision: int = 6) -> list[float]:
    return np.round(np.asarray(points, dtype=float).reshape(-1), precision).tolist()


def save_interactive_html(
    fields: PosterSphereFields,
    output: str | Path,
    *,
    contour_levels: Sequence[float],
    arrow_density: int = 80,
    arrow_scale: float = 0.085,
    arrow_scale_mode: str = "magnitude",
    arrow_min_scale: float = 0.35,
    arrow_max_scale: float = 1.0,
    arrow_radius_offset: float = DEFAULT_ARROW_RADIUS_OFFSET,
    edge_opacity: float = 0.36,
    contour_color_mode: str = "height",
    contour_width: float = 2.0,
    axes_mode: str = "none",
    axes_color: str = "#303030",
    axes_length: float = 1.28,
    show_rotation_axis: bool = True,
    rotation_axis_length: float = 1.35,
    rotation_axis_color: str = "#d62728",
    rotation_axis_width: float = 4.0,
    show_rotation_ring: bool = True,
    rotation_ring_fraction: float = 0.75,
    rotation_ring_radius: float = 0.48,
    rotation_ring_radius_offset: float = 0.032,
    rotation_ring_width: float = 4.0,
    rotation_ring_color: str | None = None,
    rotation_ring_samples: int = 160,
    rotation_ring_cone_height: float = 0.18,
    rotation_ring_cone_radius: float = 0.080,
    rotation_ring_cone_offset: float = 0.0,
    colormap: str = "gyror",
    show_colorbar: bool = True,
    colorbar_format: str = "%.0f",
    camera_state: dict[str, float] | None = None,
    title: str = "Poster Sphere",
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    surface_arrays = _build_surface_plot_arrays(fields)
    surface_colors = _surface_vertex_colors(
        surface_arrays["q_plot"],
        amplitude=fields.amplitude,
        colormap=colormap,
    )
    edge_segments = _mesh_edge_segments(fields.mesh)
    contour_segments = _initial_contour_segments_by_level(
        surface_arrays,
        levels=contour_levels,
    )
    arrows = sample_velocity_arrows(
        fields,
        arrow_density=arrow_density,
        radius_offset=arrow_radius_offset,
    )
    arrow_mode = str(arrow_scale_mode).lower().strip()

    if arrows.plot_starts.shape[0] > 0:
        if arrow_mode == "magnitude":
            arrow_factors = float(arrow_min_scale) + (
                float(arrow_max_scale) - float(arrow_min_scale)
            ) * arrows.speed_fraction
        else:
            arrow_factors = np.ones_like(arrows.speed_fraction)

        arrow_lengths = float(arrow_scale) * arrow_factors
    else:
        arrow_lengths = np.empty((0,), dtype=float)

    axis = fields.omega / fields.omega_norm
    ring_color = rotation_ring_color or rotation_axis_color
    rotation_ring = _rotation_direction_ring_arrays(
        fields,
        arc_fraction=rotation_ring_fraction,
        ring_radius=rotation_ring_radius,
        radius_offset=rotation_ring_radius_offset,
        samples=rotation_ring_samples,
        cone_height=rotation_ring_cone_height,
        cone_radius=rotation_ring_cone_radius,
        cone_offset=rotation_ring_cone_offset,
    )
    contour_payload = []

    for contour in contour_segments:
        level = float(contour["level"])
        color = "#111111" if contour_color_mode == "neutral" else scalar_color_from_colormap(
            level,
            amplitude=fields.amplitude,
            colormap=colormap,
        )
        contour_payload.append(
            {
                "level": level,
                "color": color,
                "segments": _flatten_points(np.asarray(contour["segments"], dtype=float)),
            }
        )

    scene_data = {
        "title": title,
        "surface": {
            "positions": _flatten_points(surface_arrays["points"]),
            "colors": _flatten_points(surface_colors, precision=5),
            "indices": np.asarray(surface_arrays["triangles"], dtype=int).reshape(-1).tolist(),
        },
        "edges": {
            "segments": _flatten_points(edge_segments),
            "color": "#202020",
            "opacity": float(edge_opacity),
        },
        "contours": {
            "items": contour_payload,
            "width": float(contour_width),
            "label": f"Dashed initial contours q(t=0): {format_scalar_values(contour_levels)}",
        },
        "arrows": {
            "starts": _flatten_points(arrows.plot_starts),
            "directions": _flatten_points(arrows.directions),
            "lengths": np.round(arrow_lengths, 6).tolist(),
            "color": "#242424",
        },
        "axes": {
            "mode": str(axes_mode),
            "color": axes_color,
            "length": float(axes_length),
        },
        "rotationAxis": {
            "show": bool(show_rotation_axis),
            "direction": _flatten_points(axis),
            "length": float(rotation_axis_length),
            "color": rotation_axis_color,
            "width": float(rotation_axis_width),
        },
        "rotationRing": {
            "show": bool(show_rotation_ring),
            "points": _flatten_points(np.asarray(rotation_ring["points"], dtype=float)),
            "color": ring_color,
            "width": float(rotation_ring_width),
            "conePosition": _flatten_points(np.asarray(rotation_ring["cone_position"], dtype=float)),
            "coneDirection": _flatten_points(np.asarray(rotation_ring["cone_direction"], dtype=float)),
            "coneHeight": float(rotation_ring["cone_height"]),
            "coneRadius": float(rotation_ring["cone_radius"]),
        },
        "colorbar": {
            "show": bool(show_colorbar),
            "amplitude": float(fields.amplitude),
            "format": colorbar_format,
            "colors": GYROR_COLORS,
        },
        "camera": _camera_json(camera_state),
    }
    html = _standalone_html_document(scene_data)
    output_path.write_text(html, encoding="utf-8")

    return output_path


def _standalone_html_document(scene_data: dict) -> str:
    data_json = json.dumps(scene_data, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{scene_data["title"]}</title>
<style>
html, body {{
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  font-family: Arial, sans-serif;
  background: #ffffff;
  color: #202020;
}}
#poster-viewer {{
  position: fixed;
  inset: 0;
}}
#camera-panel {{
  position: fixed;
  left: 16px;
  top: 16px;
  z-index: 10;
  display: grid;
  grid-template-columns: repeat(4, minmax(86px, 1fr)) auto;
  gap: 8px;
  align-items: end;
  max-width: calc(100vw - 32px);
  padding: 10px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 0, 0, 0.18);
}}
#camera-panel label {{
  display: grid;
  gap: 3px;
  font-size: 12px;
}}
#camera-panel input {{
  width: 100%;
  box-sizing: border-box;
  font: inherit;
  padding: 5px 6px;
  border: 1px solid #9a9a9a;
  background: #ffffff;
  color: #202020;
}}
#camera-panel button {{
  font: inherit;
  padding: 6px 10px;
  border: 1px solid #303030;
  background: #303030;
  color: #ffffff;
  cursor: pointer;
}}
#colorbar {{
  position: fixed;
  right: 22px;
  top: 26%;
  z-index: 8;
  display: grid;
  grid-template-columns: 28px auto;
  gap: 8px;
  align-items: stretch;
  font-size: 12px;
}}
#colorbar-gradient {{
  height: 220px;
  background: linear-gradient(to top, #00a65a 0%, #ffd24d 40%, #ff8c3a 70%, #d62728 100%);
  border: 1px solid rgba(0, 0, 0, 0.28);
}}
#colorbar-ticks {{
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}}
#contour-label {{
  position: fixed;
  right: 22px;
  top: calc(26% + 238px);
  z-index: 8;
  max-width: 280px;
  font-size: 12px;
  color: #202020;
  text-align: right;
}}
@media (max-width: 720px) {{
  #camera-panel {{
    grid-template-columns: repeat(2, minmax(110px, 1fr));
  }}
  #camera-panel button {{
    grid-column: 1 / -1;
  }}
}}
</style>
</head>
<body>
<div id="poster-viewer"></div>
<form id="camera-panel">
  <label>Azimuth<input id="camera-azimuth" type="number" step="1"></label>
  <label>Elevation<input id="camera-elevation" type="number" step="1"></label>
  <label>Distance<input id="camera-distance" type="number" step="0.1" min="0.1"></label>
  <label>Roll<input id="camera-roll" type="number" step="1"></label>
  <button type="submit">Apply</button>
</form>
<div id="colorbar" aria-label="Height colorbar">
  <div id="colorbar-gradient"></div>
  <div id="colorbar-ticks"></div>
</div>
<div id="contour-label"></div>
<script type="module">
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import {{ OrbitControls }} from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const DATA = {data_json};
const root = document.getElementById("poster-viewer");
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0xffffff, 1);
root.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;

function colorHex(hex) {{
  return new THREE.Color(hex);
}}

function setBufferAttribute(geometry, name, values, itemSize) {{
  geometry.setAttribute(name, new THREE.Float32BufferAttribute(values, itemSize));
}}

function addSurface() {{
  const geometry = new THREE.BufferGeometry();
  setBufferAttribute(geometry, "position", DATA.surface.positions, 3);
  setBufferAttribute(geometry, "color", DATA.surface.colors, 3);
  geometry.setIndex(DATA.surface.indices);
  geometry.computeVertexNormals();
  const material = new THREE.MeshBasicMaterial({{
    vertexColors: true,
    side: THREE.DoubleSide
  }});
  scene.add(new THREE.Mesh(geometry, material));
}}

function addLineSegments(values, color, opacity, dashed = false, dashSize = 0.045, gapSize = 0.028) {{
  if (!values.length) return;
  const geometry = new THREE.BufferGeometry();
  setBufferAttribute(geometry, "position", values, 3);
  const material = dashed
    ? new THREE.LineDashedMaterial({{ color: colorHex(color), dashSize, gapSize, linewidth: 1 }})
    : new THREE.LineBasicMaterial({{ color: colorHex(color), transparent: opacity < 1, opacity, linewidth: 1 }});
  const lines = new THREE.LineSegments(geometry, material);
  if (dashed) lines.computeLineDistances();
  scene.add(lines);
}}

function addPolyline(values, color) {{
  if (!values.length) return;
  const geometry = new THREE.BufferGeometry();
  setBufferAttribute(geometry, "position", values, 3);
  const material = new THREE.LineBasicMaterial({{ color: colorHex(color), linewidth: 1 }});
  scene.add(new THREE.Line(geometry, material));
}}

function addArrow(origin, direction, length, color, headLength, headWidth) {{
  const dir = new THREE.Vector3(direction[0], direction[1], direction[2]).normalize();
  const start = new THREE.Vector3(origin[0], origin[1], origin[2]);
  const arrow = new THREE.ArrowHelper(dir, start, length, colorHex(color), headLength, headWidth);
  scene.add(arrow);
}}

function addArrows() {{
  for (let i = 0; i < DATA.arrows.lengths.length; i += 1) {{
    const p = 3 * i;
    const origin = DATA.arrows.starts.slice(p, p + 3);
    const direction = DATA.arrows.directions.slice(p, p + 3);
    const length = DATA.arrows.lengths[i];
    addArrow(origin, direction, length, DATA.arrows.color, 0.32 * length, 0.08 * length);
  }}
}}

function addAxes() {{
  const mode = DATA.axes.mode;
  if (mode === "none" || mode === "corner") return;
  const L = DATA.axes.length;
  addArrow([0, 0, 0], [1, 0, 0], L, DATA.axes.color, 0.10 * L, 0.035 * L);
  addArrow([0, 0, 0], [0, 1, 0], L, DATA.axes.color, 0.10 * L, 0.035 * L);
  addArrow([0, 0, 0], [0, 0, 1], L, DATA.axes.color, 0.10 * L, 0.035 * L);
}}

function addRotationAxis() {{
  if (!DATA.rotationAxis.show) return;
  const d = DATA.rotationAxis.direction;
  const L = DATA.rotationAxis.length;
  addArrow([-L * d[0], -L * d[1], -L * d[2]], d, 2 * L, DATA.rotationAxis.color, 0.16 * 2 * L, 0.025 * DATA.rotationAxis.width);
}}

function addRotationRing() {{
  if (!DATA.rotationRing.show) return;
  addPolyline(DATA.rotationRing.points, DATA.rotationRing.color);
  const pos = DATA.rotationRing.conePosition;
  const dir = DATA.rotationRing.coneDirection;
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(DATA.rotationRing.coneRadius, DATA.rotationRing.coneHeight, 32),
    new THREE.MeshBasicMaterial({{ color: colorHex(DATA.rotationRing.color) }})
  );
  cone.position.set(pos[0], pos[1], pos[2]);
  const up = new THREE.Vector3(0, 1, 0);
  const target = new THREE.Vector3(dir[0], dir[1], dir[2]).normalize();
  cone.quaternion.setFromUnitVectors(up, target);
  scene.add(cone);
}}

function setCameraFromValues() {{
  const az = Number(document.getElementById("camera-azimuth").value) * Math.PI / 180;
  const el = Number(document.getElementById("camera-elevation").value) * Math.PI / 180;
  const dist = Math.max(0.1, Number(document.getElementById("camera-distance").value));
  const roll = Number(document.getElementById("camera-roll").value) * Math.PI / 180;
  camera.position.set(
    dist * Math.cos(el) * Math.cos(az),
    dist * Math.cos(el) * Math.sin(az),
    dist * Math.sin(el)
  );
  camera.up.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  camera.rotateZ(roll);
  controls.target.set(0, 0, 0);
  controls.update();
}}

function updateCameraInputsFromPosition() {{
  const p = camera.position;
  const dist = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
  if (dist <= 0) return;
  document.getElementById("camera-azimuth").value = (Math.atan2(p.y, p.x) * 180 / Math.PI).toFixed(2);
  document.getElementById("camera-elevation").value = (Math.asin(p.z / dist) * 180 / Math.PI).toFixed(2);
  document.getElementById("camera-distance").value = dist.toFixed(2);
}}

function initControls() {{
  document.getElementById("camera-azimuth").value = DATA.camera.azimuth;
  document.getElementById("camera-elevation").value = DATA.camera.elevation;
  document.getElementById("camera-distance").value = DATA.camera.distance;
  document.getElementById("camera-roll").value = DATA.camera.roll;
  document.getElementById("camera-panel").addEventListener("submit", (event) => {{
    event.preventDefault();
    setCameraFromValues();
  }});
  controls.addEventListener("end", updateCameraInputsFromPosition);
}}

function initOverlay() {{
  const colorbar = document.getElementById("colorbar");
  colorbar.style.display = DATA.colorbar.show ? "grid" : "none";
  const ticks = document.getElementById("colorbar-ticks");
  const amplitude = DATA.colorbar.amplitude;
  ticks.innerHTML = [amplitude, 0.75 * amplitude, 0.5 * amplitude, 0.25 * amplitude, 0]
    .map((value) => `<span>${{value.toFixed(0)}}</span>`)
    .join("");
  document.getElementById("contour-label").textContent = DATA.contours.label;
}}

function resize() {{
  const width = root.clientWidth;
  const height = root.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
}}

addSurface();
addLineSegments(DATA.edges.segments, DATA.edges.color, DATA.edges.opacity);
for (const contour of DATA.contours.items) {{
  addLineSegments(contour.segments, contour.color, 1, true);
}}
addArrows();
addAxes();
addRotationAxis();
addRotationRing();
initControls();
initOverlay();
resize();
setCameraFromValues();
window.addEventListener("resize", resize);

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


def format_sanity_checks(fields: PosterSphereFields) -> str:
    lines = [
        f"radius                 : {fields.radius:.16e}",
        f"sigma_physical         : {fields.sigma:.16e}",
        f"omega                  : [{fields.omega[0]:.16e}, {fields.omega[1]:.16e}, {fields.omega[2]:.16e}]",
        f"omega_norm             : {fields.omega_norm:.16e}",
        f"period                 : {fields.period:.16e}",
        f"t_half                 : {fields.t_half:.16e}",
        f"t_plot                 : {fields.t_plot:.16e}",
        f"plot_days              : {fields.plot_days:.16e}",
        f"center0                : [{fields.center0[0]:.16e}, {fields.center0[1]:.16e}, {fields.center0[2]:.16e}]",
        (
            "center_half            : "
            f"[{fields.center_half[0]:.16e}, {fields.center_half[1]:.16e}, {fields.center_half[2]:.16e}]"
        ),
        (
            "center_plot            : "
            f"[{fields.center_plot[0]:.16e}, {fields.center_plot[1]:.16e}, {fields.center_plot[2]:.16e}]"
        ),
        "sanity checks:",
    ]

    for key in sorted(fields.sanity):
        lines.append(f"  {key:<34s}: {fields.sanity[key]:.16e}")

    return "\n".join(lines)
