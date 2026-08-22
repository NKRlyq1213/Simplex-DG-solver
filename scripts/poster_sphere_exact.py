from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


try:
    from examples.check_metric_divergence import parse_float_expr, rotation_axis_from_alpha0
except ModuleNotFoundError:
    from check_metric_divergence import parse_float_expr, rotation_axis_from_alpha0

from simplex_dg.visualization.poster_sphere import (
    DEFAULT_ARROW_RADIUS_OFFSET,
    DEFAULT_CAMERA_AZIMUTH,
    DEFAULT_CAMERA_DISTANCE,
    DEFAULT_CAMERA_ELEVATION,
    DEFAULT_CAMERA_ROLL,
    DEFAULT_POSTER_AMPLITUDE,
    EARTH_RADIUS_METERS,
    SURFACE_ARROW_RADIUS_OFFSET,
    build_exact_fields,
    configure_camera_from_angles,
    format_sanity_checks,
    render_scene,
    resolve_rotation_axis_radii,
    resolve_sigma_physical,
    save_interactive_html,
    save_screenshot,
)


_EXPR_OPTIONS = {
    "--radius",
    "--R",
    "--alpha0",
    "--u0",
    "--sigma",
    "--sigma-physical",
    "--amplitude",
    "--height",
    "--days",
    "--period-days",
    "--arrow-scale",
    "--arrow-min-scale",
    "--arrow-max-scale",
    "--arrow-tip-length",
    "--arrow-tip-radius",
    "--arrow-shaft-radius",
    "--arrow-radius-offset",
    "--axes-length",
    "--axes-label-offset",
    "--axes-tip-length",
    "--axes-tip-radius",
    "--axes-shaft-radius",
    "--rotation-axis-length",
    "--rotation-axis-width",
    "--rotation-axis-tip-length",
    "--rotation-axis-tip-radius",
    "--rotation-axis-shaft-radius",
    "--rotation-ring-fraction",
    "--rotation-ring-radius",
    "--rotation-ring-radius-offset",
    "--rotation-ring-width",
    "--rotation-ring-cone-size",
    "--rotation-ring-cone-height",
    "--rotation-ring-cone-radius",
    "--rotation-ring-cone-offset",
    "--edge-width",
    "--edge-opacity",
    "--contour-width",
    "--colorbar-height",
    "--colorbar-width",
    "--colorbar-position-x",
    "--colorbar-position-y",
    "--dash-length",
    "--gap-length",
    "--camera-azimuth",
    "--camera-elevation",
    "--camera-distance",
    "--camera-roll",
}

DEFAULT_PERIOD_DAYS = 12.0


def resolve_rotation_ring_cone_dimensions(
    *,
    cone_size: float,
    cone_height: float,
    cone_radius: float,
) -> tuple[float, float]:
    cone_size_f = float(cone_size)
    cone_height_f = float(cone_height)
    cone_radius_f = float(cone_radius)

    if cone_size_f <= 0.0:
        raise ValueError("cone_size must be positive.")

    if cone_height_f <= 0.0 or cone_radius_f <= 0.0:
        raise ValueError("rotation ring cone dimensions must be positive.")

    return (
        cone_size_f * cone_height_f,
        cone_size_f * cone_radius_f,
    )


def resolve_plot_time_from_days(*, days: float, period: float, period_days: float = DEFAULT_PERIOD_DAYS) -> float:
    days_f = float(days)
    period_f = float(period)
    period_days_f = float(period_days)

    if days_f < 0.0:
        raise ValueError("days must be nonnegative.")

    if period_f <= 0.0:
        raise ValueError("period must be positive.")

    if period_days_f <= 0.0:
        raise ValueError("period_days must be positive.")

    return days_f / period_days_f * period_f


def _normalize_expr_option_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0

    while i < len(argv):
        token = argv[i]

        if token in _EXPR_OPTIONS and i + 1 < len(argv):
            normalized.append(f"{token}={argv[i + 1]}")
            i += 2
            continue

        normalized.append(token)
        i += 1

    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive exact Gaussian solid-body-rotation poster figure on the simplex sphere mesh."
    )

    geometry = parser.add_argument_group("mesh / reference geometry")
    geometry.add_argument("--ndiv", type=int, default=4, help="Octa-sphere subdivision count.")
    geometry.add_argument("--order", type=int, default=4, help="Reference rule polynomial order.")
    geometry.add_argument("--table", choices=["table1", "table2"], default="table1")

    physics = parser.add_argument_group("physical exact field")
    physics.add_argument("--radius", "--R", dest="radius", type=parse_float_expr, default=EARTH_RADIUS_METERS)
    physics.add_argument(
        "--alpha0",
        type=parse_float_expr,
        default=-np.pi / 4.0,
        help="Rotation-axis tilt. Uses the same convention as the existing simplex experiments.",
    )
    physics.add_argument(
        "--u0",
        type=parse_float_expr,
        default=1.0,
        help="Angular speed multiplier. omega = u0 * rotation_axis_from_alpha0(alpha0).",
    )
    physics.add_argument(
        "--sigma",
        type=parse_float_expr,
        default=0.35,
        help="Gaussian angular width in radians; physical width is radius*sigma unless overridden.",
    )
    physics.add_argument(
        "--sigma-physical",
        type=parse_float_expr,
        default=None,
        help="Optional Gaussian physical arc-length width in meters.",
    )
    physics.add_argument(
        "--amplitude",
        "--height",
        dest="amplitude",
        type=parse_float_expr,
        default=DEFAULT_POSTER_AMPLITUDE,
        help="Gaussian peak height. The poster default is 1000, so scalar values run 0..1000.",
    )
    physics.add_argument(
        "--days",
        type=parse_float_expr,
        default=6.0,
        help="Poster time in days for the plotted field. With --period-days 12, --days 6 is the half-period state.",
    )
    physics.add_argument(
        "--period-days",
        type=parse_float_expr,
        default=DEFAULT_PERIOD_DAYS,
        help="Poster-time period in days. Default is 12 days, so 6 days maps to the north-pole half-period state.",
    )

    visual = parser.add_argument_group("poster visual controls")
    visual.add_argument("--arrow-density", type=int, default=80, help="Target number of tangent arrows.")
    visual.add_argument("--arrow-scale", type=parse_float_expr, default=0.085, help="Arrow length in plot-radius units.")
    visual.add_argument(
        "--arrow-scale-mode",
        choices=["magnitude", "uniform"],
        default="magnitude",
        help="Use magnitude to make wind arrows longer where the wind speed is larger, or uniform for equal lengths.",
    )
    visual.add_argument(
        "--arrow-min-scale",
        type=parse_float_expr,
        default=0.35,
        help="Minimum glyph multiplier used by --arrow-scale-mode magnitude.",
    )
    visual.add_argument(
        "--arrow-max-scale",
        type=parse_float_expr,
        default=1.0,
        help="Maximum glyph multiplier used by --arrow-scale-mode magnitude.",
    )
    visual.add_argument("--arrow-tip-length", type=parse_float_expr, default=0.30)
    visual.add_argument("--arrow-tip-radius", type=parse_float_expr, default=0.045)
    visual.add_argument("--arrow-shaft-radius", type=parse_float_expr, default=0.014)
    visual.add_argument("--font-size", type=int, default=12, help="Default font size for poster annotations.")
    visual.add_argument(
        "--axes",
        choices=["none", "corner", "origin", "scene", "both"],
        default="none",
        help="Optional XYZ axes: none, corner orientation widget, origin axes from the sphere center, or both.",
    )
    visual.add_argument("--axes-label-font-size", "--axes-font-size", dest="axes_label_font_size", type=int, default=None)
    visual.add_argument("--axes-color", default="#303030")
    visual.add_argument("--axes-length", type=parse_float_expr, default=1.28)
    visual.add_argument("--axes-label-offset", type=parse_float_expr, default=0.08)
    visual.add_argument("--axes-tip-length", type=parse_float_expr, default=0.16)
    visual.add_argument("--axes-tip-radius", type=parse_float_expr, default=0.025)
    visual.add_argument("--axes-shaft-radius", type=parse_float_expr, default=0.0075)
    visual.add_argument(
        "--rotation-axis",
        dest="rotation_axis",
        action="store_true",
        default=True,
        help="Show the red rotation axis through the sphere center. Enabled by default.",
    )
    visual.add_argument("--no-rotation-axis", dest="rotation_axis", action="store_false")
    visual.add_argument(
        "--rotation-axis-length",
        type=parse_float_expr,
        default=1.35,
        help="Half-length of the red rotation axis from the sphere center in normalized plot-radius units.",
    )
    visual.add_argument(
        "--rotation-axis-width",
        type=parse_float_expr,
        default=4.0,
        help="Quick thickness multiplier for the red rotation-axis arrow when explicit shaft/tip radii are not set.",
    )
    visual.add_argument("--rotation-axis-color", default="#d62728")
    visual.add_argument("--rotation-axis-tip-length", type=parse_float_expr, default=0.16)
    visual.add_argument("--rotation-axis-tip-radius", type=parse_float_expr, default=None)
    visual.add_argument("--rotation-axis-shaft-radius", type=parse_float_expr, default=None)
    visual.add_argument(
        "--rotation-ring",
        dest="rotation_ring",
        action="store_true",
        default=True,
        help="Show a 3/4 circular arrow on the sphere to indicate the positive full-rotation direction.",
    )
    visual.add_argument("--no-rotation-ring", dest="rotation_ring", action="store_false")
    visual.add_argument("--rotation-ring-fraction", type=parse_float_expr, default=0.75)
    visual.add_argument(
        "--rotation-ring-radius",
        type=parse_float_expr,
        default=0.48,
        help="Normalized radius of the circular rotation-direction ring on the sphere. Smaller values draw a smaller ring.",
    )
    visual.add_argument("--rotation-ring-radius-offset", type=parse_float_expr, default=0.032)
    visual.add_argument("--rotation-ring-width", type=parse_float_expr, default=4.0)
    visual.add_argument("--rotation-ring-color", default=None)
    visual.add_argument("--rotation-ring-samples", type=int, default=160)
    visual.add_argument(
        "--rotation-ring-cone-size",
        type=parse_float_expr,
        default=1.0,
        help="Overall multiplier for the circular direction cone height and radius.",
    )
    visual.add_argument("--rotation-ring-cone-height", type=parse_float_expr, default=0.18)
    visual.add_argument("--rotation-ring-cone-radius", type=parse_float_expr, default=0.080)
    visual.add_argument("--rotation-ring-cone-offset", type=parse_float_expr, default=0.0)
    visual.add_argument("--rotation-ring-cone-resolution", type=int, default=32)
    visual.add_argument(
        "--arrow-placement",
        choices=["lifted", "surface"],
        default="lifted",
        help="Use lifted for the current raised arrows, or surface to place arrow bases on the sphere.",
    )
    visual.add_argument(
        "--arrow-radius-offset",
        type=parse_float_expr,
        default=None,
        help="Override arrow radial offset in normalized plot-radius units.",
    )
    visual.add_argument("--edge-width", type=parse_float_expr, default=1.0)
    visual.add_argument("--edge-opacity", type=parse_float_expr, default=0.36)
    visual.add_argument("--contour-width", type=parse_float_expr, default=2.0)
    visual.add_argument(
        "--contour-color-mode",
        choices=["height", "neutral"],
        default="height",
        help="Color dashed initial contours by initial-field height, or use one neutral contour color.",
    )
    visual.add_argument(
        "--contour-levels",
        nargs="+",
        type=parse_float_expr,
        default=[0.2, 0.5, 0.8],
        help="Initial-field contour levels as fractions of amplitude.",
    )
    visual.add_argument(
        "--contour-level-mode",
        choices=["fraction", "absolute"],
        default="fraction",
        help="Interpret --contour-levels as amplitude fractions or absolute scalar values.",
    )
    visual.add_argument("--no-contour-label", action="store_true")
    visual.add_argument("--contour-label-position", default="below_colorbar")
    visual.add_argument("--contour-label-font-size", type=int, default=None)
    visual.add_argument("--dash-length", type=parse_float_expr, default=0.045)
    visual.add_argument("--gap-length", type=parse_float_expr, default=0.028)
    visual.add_argument("--colormap", default="gyror", help="Use gyror for green-yellow-orange-red.")
    visual.add_argument("--lighting", action="store_true", help="Enable 3D scene lighting. Default is flat scalar colors.")
    visual.add_argument("--no-colorbar", action="store_true")
    visual.add_argument("--colorbar-height", type=parse_float_expr, default=0.38)
    visual.add_argument("--colorbar-width", type=parse_float_expr, default=0.07)
    visual.add_argument("--colorbar-position-x", type=parse_float_expr, default=0.88)
    visual.add_argument("--colorbar-position-y", type=parse_float_expr, default=0.28)
    visual.add_argument("--colorbar-label-font-size", type=int, default=None)
    visual.add_argument("--colorbar-title-font-size", type=int, default=None)
    visual.add_argument("--colorbar-n-labels", type=int, default=5)
    visual.add_argument("--colorbar-format", default="%.0f")

    output = parser.add_argument_group("preview / export")
    output.add_argument("--output", type=str, default=None, help="Optional PNG screenshot path.")
    output.add_argument(
        "--html-output",
        type=str,
        default=None,
        help="Optional standalone interactive HTML export path with camera angle inputs.",
    )
    output.add_argument("--window-size", nargs=2, type=int, default=[1600, 1200], metavar=("WIDTH", "HEIGHT"))
    output.add_argument("--transparent-background", action="store_true")
    output.add_argument("--show", action="store_true", help="Show the interactive window even when --output is set.")
    output.add_argument("--no-show", action="store_true", help="Do not open the interactive window.")
    output.add_argument("--off-screen", action="store_true", help="Force PyVista off-screen rendering.")
    output.add_argument(
        "--save-current-view",
        action="store_true",
        help=(
            "Open an interactive window and save the current camera view to --output "
            "and/or --html-output when --save-key is pressed."
        ),
    )
    output.add_argument(
        "--save-key",
        default="s",
        help="Keyboard key used by --save-current-view. Default: s.",
    )

    camera = parser.add_argument_group("camera angle controls")
    camera.add_argument("--camera-azimuth", type=parse_float_expr, default=None, help="Camera azimuth in degrees.")
    camera.add_argument("--camera-elevation", type=parse_float_expr, default=None, help="Camera elevation in degrees.")
    camera.add_argument(
        "--camera-distance",
        type=parse_float_expr,
        default=None,
        help="Camera distance from the sphere center in normalized plot-radius units.",
    )
    camera.add_argument("--camera-roll", type=parse_float_expr, default=None, help="Camera roll in degrees.")
    camera.add_argument(
        "--camera-input",
        action="store_true",
        help="In the interactive window, press --camera-input-key to type camera angles in the terminal.",
    )
    camera.add_argument(
        "--camera-input-key",
        default="c",
        help="Keyboard key used by --camera-input. Default: c.",
    )
    output.add_argument("--check-only", action="store_true", help="Build exact fields and print sanity checks only.")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(_normalize_expr_option_args(list(argv)))

    if args.ndiv < 1:
        parser.error("--ndiv must be >= 1.")

    if args.arrow_density < 0:
        parser.error("--arrow-density must be nonnegative.")

    if args.days < 0.0:
        parser.error("--days must be nonnegative.")

    if args.period_days <= 0.0:
        parser.error("--period-days must be positive.")

    if args.arrow_scale <= 0.0:
        parser.error("--arrow-scale must be positive.")

    if args.arrow_min_scale < 0.0 or args.arrow_max_scale <= 0.0 or args.arrow_max_scale < args.arrow_min_scale:
        parser.error("--arrow-min-scale must be >= 0 and --arrow-max-scale must be positive and >= --arrow-min-scale.")

    if args.arrow_tip_length <= 0.0 or args.arrow_tip_radius <= 0.0 or args.arrow_shaft_radius <= 0.0:
        parser.error("--arrow-tip-length, --arrow-tip-radius, and --arrow-shaft-radius must be positive.")

    if args.colorbar_height <= 0.0 or args.colorbar_width <= 0.0:
        parser.error("--colorbar-height and --colorbar-width must be positive.")

    if args.axes_length <= 0.0 or args.axes_label_offset < 0.0:
        parser.error("--axes-length must be positive and --axes-label-offset must be nonnegative.")

    if args.axes_tip_length <= 0.0 or args.axes_tip_radius <= 0.0 or args.axes_shaft_radius <= 0.0:
        parser.error("--axes-tip-length, --axes-tip-radius, and --axes-shaft-radius must be positive.")

    if args.rotation_axis_length <= 0.0 or args.rotation_axis_width <= 0.0:
        parser.error("--rotation-axis-length and --rotation-axis-width must be positive.")

    if args.rotation_axis_tip_length <= 0.0:
        parser.error("--rotation-axis-tip-length must be positive.")

    if args.rotation_axis_tip_radius is not None and args.rotation_axis_tip_radius <= 0.0:
        parser.error("--rotation-axis-tip-radius must be positive.")

    if args.rotation_axis_shaft_radius is not None and args.rotation_axis_shaft_radius <= 0.0:
        parser.error("--rotation-axis-shaft-radius must be positive.")

    if not (0.0 < args.rotation_ring_fraction <= 1.0):
        parser.error("--rotation-ring-fraction must be in (0, 1].")

    if not (0.0 < args.rotation_ring_radius <= 1.0):
        parser.error("--rotation-ring-radius must be in (0, 1].")

    if args.rotation_ring_radius_offset < 0.0:
        parser.error("--rotation-ring-radius-offset must be nonnegative.")

    if args.rotation_ring_width <= 0.0:
        parser.error("--rotation-ring-width must be positive.")

    if args.rotation_ring_samples < 3:
        parser.error("--rotation-ring-samples must be >= 3.")

    if args.rotation_ring_cone_size <= 0.0:
        parser.error("--rotation-ring-cone-size must be positive.")

    if args.rotation_ring_cone_height <= 0.0 or args.rotation_ring_cone_radius <= 0.0:
        parser.error("--rotation-ring-cone-height and --rotation-ring-cone-radius must be positive.")

    if args.rotation_ring_cone_offset < 0.0:
        parser.error("--rotation-ring-cone-offset must be nonnegative.")

    if args.rotation_ring_cone_resolution < 3:
        parser.error("--rotation-ring-cone-resolution must be >= 3.")

    if args.contour_width <= 0.0:
        parser.error("--contour-width must be positive.")

    font_size_values = [
        args.font_size,
        args.axes_label_font_size,
        args.contour_label_font_size,
        args.colorbar_label_font_size,
        args.colorbar_title_font_size,
    ]

    if any(value is not None and value <= 0 for value in font_size_values):
        parser.error("font size values must be positive.")

    if args.show and args.no_show:
        parser.error("--show and --no-show cannot both be set.")

    if args.save_current_view and args.output is None and args.html_output is None:
        parser.error("--save-current-view requires --output or --html-output.")

    if args.save_current_view and args.no_show:
        parser.error("--save-current-view cannot be used with --no-show.")

    if args.save_current_view and args.off_screen:
        parser.error("--save-current-view cannot be used with --off-screen.")

    if args.camera_input and args.no_show:
        parser.error("--camera-input cannot be used with --no-show.")

    if args.camera_input and args.off_screen:
        parser.error("--camera-input cannot be used with --off-screen.")

    if not args.save_key.strip():
        parser.error("--save-key must be nonempty.")

    if not args.camera_input_key.strip():
        parser.error("--camera-input-key must be nonempty.")

    if args.camera_distance is not None and args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive.")

    if args.camera_elevation is not None and not (-90.0 < args.camera_elevation < 90.0):
        parser.error("--camera-elevation must be in (-90, 90).")

    return args


def resolve_camera_angle_state(args: argparse.Namespace) -> dict[str, float]:
    return {
        "azimuth": DEFAULT_CAMERA_AZIMUTH if args.camera_azimuth is None else float(args.camera_azimuth),
        "elevation": DEFAULT_CAMERA_ELEVATION if args.camera_elevation is None else float(args.camera_elevation),
        "distance": DEFAULT_CAMERA_DISTANCE if args.camera_distance is None else float(args.camera_distance),
        "roll": DEFAULT_CAMERA_ROLL if args.camera_roll is None else float(args.camera_roll),
    }


def has_explicit_camera_angles(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.camera_azimuth,
            args.camera_elevation,
            args.camera_distance,
            args.camera_roll,
        )
    )


def apply_camera_angle_state(plotter, state: dict[str, float]) -> None:
    configure_camera_from_angles(
        plotter,
        azimuth=state["azimuth"],
        elevation=state["elevation"],
        distance=state["distance"],
        roll=state["roll"],
    )
    plotter.render()


def format_camera_angle_state(state: dict[str, float]) -> str:
    return (
        f"azimuth={state['azimuth']:.6g}, "
        f"elevation={state['elevation']:.6g}, "
        f"distance={state['distance']:.6g}, "
        f"roll={state['roll']:.6g}"
    )


def camera_angle_state_from_plotter(plotter, fallback: dict[str, float]) -> dict[str, float]:
    try:
        position = np.asarray(plotter.camera_position[0], dtype=float)
        focal_point = np.asarray(plotter.camera_position[1], dtype=float)
    except (AttributeError, IndexError, TypeError, ValueError):
        return dict(fallback)

    vector = position - focal_point
    distance = float(np.linalg.norm(vector))

    if distance <= 0.0:
        return dict(fallback)

    return {
        "azimuth": float(np.rad2deg(np.arctan2(vector[1], vector[0]))),
        "elevation": float(np.rad2deg(np.arcsin(np.clip(vector[2] / distance, -1.0, 1.0)))),
        "distance": distance,
        "roll": float(fallback.get("roll", DEFAULT_CAMERA_ROLL)),
    }


def read_camera_angle_value(name: str, current: float) -> float:
    raw = input(f"{name} [{current:.6g}]: ").strip()

    if not raw:
        return current

    return float(raw)


def add_camera_input_key(
    plotter,
    state: dict[str, float],
    *,
    key: str,
) -> None:
    def update_camera_from_terminal() -> None:
        print("")
        print("Camera angle input. Press Enter to keep the current value.")

        try:
            next_state = {
                "azimuth": read_camera_angle_value("azimuth degrees", state["azimuth"]),
                "elevation": read_camera_angle_value("elevation degrees (-90, 90)", state["elevation"]),
                "distance": read_camera_angle_value("distance plot-radius units", state["distance"]),
                "roll": read_camera_angle_value("roll degrees", state["roll"]),
            }
        except ValueError:
            print("Invalid camera value; camera unchanged.")
            return

        if next_state["distance"] <= 0.0:
            print("Invalid camera value; distance must be positive.")
            return

        if not (-90.0 < next_state["elevation"] < 90.0):
            print("Invalid camera value; elevation must be in (-90, 90).")
            return

        state.update(next_state)
        apply_camera_angle_state(plotter, state)
        print(f"Camera angles applied     : {format_camera_angle_state(state)}")

    plotter.add_key_event(key.strip(), update_camera_from_terminal)


def add_current_view_save_key(
    plotter,
    png_output: str | None,
    html_output: str | None,
    *,
    key: str,
    transparent_background: bool,
    window_size: tuple[int, int],
    camera_state: dict[str, float],
    html_save_callback,
) -> None:
    def save_current_view() -> None:
        if png_output is not None:
            output_path = save_screenshot(
                plotter,
                png_output,
                transparent_background=transparent_background,
                window_size=window_size,
            )
            print(f"Current camera PNG written to : {output_path}")

        if html_output is not None:
            html_output_path = html_save_callback(camera_angle_state_from_plotter(plotter, camera_state))
            print(f"Current camera HTML written to: {html_output_path}")

        print(f"Camera position             : {plotter.camera_position}")

    plotter.add_key_event(key.strip(), save_current_view)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    axis = np.asarray(rotation_axis_from_alpha0(args.alpha0), dtype=float)
    omega = float(args.u0) * axis
    sigma_physical = resolve_sigma_physical(
        radius=args.radius,
        sigma_angle=args.sigma,
        sigma_physical=args.sigma_physical,
    )
    omega_norm = float(np.linalg.norm(omega))
    model_period = 2.0 * np.pi / omega_norm
    t_plot = resolve_plot_time_from_days(
        days=args.days,
        period=model_period,
        period_days=args.period_days,
    )

    fields = build_exact_fields(
        ndiv=args.ndiv,
        order=args.order,
        table=args.table,
        radius=args.radius,
        sigma=sigma_physical,
        amplitude=args.amplitude,
        omega=omega,
        t_plot=t_plot,
        plot_days=args.days,
        validate=True,
    )

    contour_level_values = np.asarray(args.contour_levels, dtype=float)

    if args.contour_level_mode == "fraction":
        contour_levels = args.amplitude * contour_level_values
    else:
        contour_levels = contour_level_values

    if args.arrow_radius_offset is None:
        arrow_radius_offset = (
            SURFACE_ARROW_RADIUS_OFFSET
            if args.arrow_placement == "surface"
            else DEFAULT_ARROW_RADIUS_OFFSET
        )
    else:
        arrow_radius_offset = float(args.arrow_radius_offset)

    axes_label_font_size = args.axes_label_font_size or args.font_size
    contour_label_font_size = args.contour_label_font_size or args.font_size
    colorbar_label_font_size = args.colorbar_label_font_size or args.font_size
    colorbar_title_font_size = args.colorbar_title_font_size or max(args.font_size + 1, args.font_size)
    axis_shaft_radius_resolved, axis_tip_radius_resolved = resolve_rotation_axis_radii(
        axis_width=args.rotation_axis_width,
        tip_radius=args.rotation_axis_tip_radius,
        shaft_radius=args.rotation_axis_shaft_radius,
    )
    (
        rotation_ring_cone_height_resolved,
        rotation_ring_cone_radius_resolved,
    ) = resolve_rotation_ring_cone_dimensions(
        cone_size=args.rotation_ring_cone_size,
        cone_height=args.rotation_ring_cone_height,
        cone_radius=args.rotation_ring_cone_radius,
    )

    print("Poster sphere exact visualization")
    print("---------------------------------")
    print(f"ndiv                   : {args.ndiv}")
    print(f"order                  : {args.order}")
    print(f"table                  : {args.table}")
    print(f"alpha0                 : {args.alpha0:.16e}")
    print(f"u0                     : {args.u0:.16e}")
    print(f"days                   : {args.days:.16e}")
    print(f"period_days            : {args.period_days:.16e}")
    print(f"poster_time_fraction   : {args.days / args.period_days:.16e}")
    print(f"sigma_angle            : {args.sigma:.16e}")
    print(format_sanity_checks(fields))
    print(f"contour_levels         : {contour_levels.tolist()}")
    print(f"colormap               : {args.colormap}")
    print(f"contour_color_mode     : {args.contour_color_mode}")
    print(f"contour_width          : {args.contour_width:.16e}")
    print(f"arrow_placement        : {args.arrow_placement}")
    print(
        "arrow_scale_mode       : "
        f"{args.arrow_scale_mode}, min={args.arrow_min_scale}, max={args.arrow_max_scale}"
    )
    print(
        "arrow_shape            : "
        f"tip_length={args.arrow_tip_length}, tip_radius={args.arrow_tip_radius}, "
        f"shaft_radius={args.arrow_shaft_radius}"
    )
    print(f"arrow_radius_offset    : {arrow_radius_offset:.16e}")
    print(f"colorbar_size          : height={args.colorbar_height}, width={args.colorbar_width}")
    print(f"axes                   : {args.axes}")
    print(f"axes_length            : {args.axes_length:.16e}")
    print(f"axes_label_offset      : {args.axes_label_offset:.16e}")
    print(
        "axes_arrow_size        : "
        f"tip_length={args.axes_tip_length}, tip_radius={args.axes_tip_radius}, "
        f"shaft_radius={args.axes_shaft_radius}"
    )
    print(
        "rotation_axis          : "
        f"show={args.rotation_axis}, length={args.rotation_axis_length}, "
        f"width={args.rotation_axis_width}, color={args.rotation_axis_color}"
    )
    print(
        "rotation_axis_arrow    : "
        f"tip_length={args.rotation_axis_tip_length}, tip_radius={args.rotation_axis_tip_radius}, "
        f"shaft_radius={args.rotation_axis_shaft_radius}, "
        f"resolved_tip_radius={axis_tip_radius_resolved}, "
        f"resolved_shaft_radius={axis_shaft_radius_resolved}"
    )
    print(
        "rotation_ring          : "
        f"show={args.rotation_ring}, fraction={args.rotation_ring_fraction}, "
        f"radius={args.rotation_ring_radius}, radius_offset={args.rotation_ring_radius_offset}, "
        f"width={args.rotation_ring_width}, color={args.rotation_ring_color or args.rotation_axis_color}"
    )
    print(
        "rotation_ring_cone     : "
        f"size={args.rotation_ring_cone_size}, height={rotation_ring_cone_height_resolved}, "
        f"radius={rotation_ring_cone_radius_resolved}, offset={args.rotation_ring_cone_offset}, "
        f"resolution={args.rotation_ring_cone_resolution}"
    )
    print(f"lighting               : {args.lighting}")
    print(
        "font_sizes             : "
        f"base={args.font_size}, contour={contour_label_font_size}, "
        f"colorbar_label={colorbar_label_font_size}, colorbar_title={colorbar_title_font_size}, "
        f"axes={axes_label_font_size}"
    )
    camera_angle_state = resolve_camera_angle_state(args)
    print(f"camera_angles          : {format_camera_angle_state(camera_angle_state)}")

    if args.check_only:
        return 0

    has_export_output = args.output is not None or args.html_output is not None
    interactive = bool(
        args.save_current_view
        or args.camera_input
        or args.show
        or (not has_export_output and not args.no_show)
    )
    off_screen = bool(args.off_screen or not interactive)
    window_size = (int(args.window_size[0]), int(args.window_size[1]))

    def write_html_output(camera_state: dict[str, float]):
        return save_interactive_html(
            fields,
            args.html_output,
            contour_levels=contour_levels,
            arrow_density=args.arrow_density,
            arrow_scale=args.arrow_scale,
            arrow_scale_mode=args.arrow_scale_mode,
            arrow_min_scale=args.arrow_min_scale,
            arrow_max_scale=args.arrow_max_scale,
            arrow_radius_offset=arrow_radius_offset,
            edge_opacity=args.edge_opacity,
            contour_color_mode=args.contour_color_mode,
            contour_width=args.contour_width,
            axes_mode=args.axes,
            axes_color=args.axes_color,
            axes_length=args.axes_length,
            show_rotation_axis=args.rotation_axis,
            rotation_axis_length=args.rotation_axis_length,
            rotation_axis_color=args.rotation_axis_color,
            rotation_axis_width=args.rotation_axis_width,
            show_rotation_ring=args.rotation_ring,
            rotation_ring_fraction=args.rotation_ring_fraction,
            rotation_ring_radius=args.rotation_ring_radius,
            rotation_ring_radius_offset=args.rotation_ring_radius_offset,
            rotation_ring_width=args.rotation_ring_width,
            rotation_ring_color=args.rotation_ring_color,
            rotation_ring_samples=args.rotation_ring_samples,
            rotation_ring_cone_height=rotation_ring_cone_height_resolved,
            rotation_ring_cone_radius=rotation_ring_cone_radius_resolved,
            rotation_ring_cone_offset=args.rotation_ring_cone_offset,
            colormap=args.colormap,
            show_colorbar=not args.no_colorbar,
            colorbar_format=args.colorbar_format,
            camera_state=camera_state,
            title=f"Poster sphere day {args.days:g}",
        )

    if args.html_output is not None and args.output is None and not interactive:
        html_output_path = write_html_output(camera_angle_state)
        print(f"HTML written to         : {html_output_path}")
        return 0

    plotter = render_scene(
        fields,
        contour_levels=contour_levels,
        arrow_density=args.arrow_density,
        arrow_scale=args.arrow_scale,
        arrow_scale_mode=args.arrow_scale_mode,
        arrow_min_scale=args.arrow_min_scale,
        arrow_max_scale=args.arrow_max_scale,
        arrow_tip_length=args.arrow_tip_length,
        arrow_tip_radius=args.arrow_tip_radius,
        arrow_shaft_radius=args.arrow_shaft_radius,
        arrow_radius_offset=arrow_radius_offset,
        edge_width=args.edge_width,
        edge_opacity=args.edge_opacity,
        contour_width=args.contour_width,
        contour_color_mode=args.contour_color_mode,
        font_size=args.font_size,
        show_contour_label=not args.no_contour_label,
        contour_label_position=args.contour_label_position,
        contour_label_font_size=contour_label_font_size,
        axes_mode=args.axes,
        axes_label_font_size=axes_label_font_size,
        axes_color=args.axes_color,
        axes_length=args.axes_length,
        axes_label_offset=args.axes_label_offset,
        axes_tip_length=args.axes_tip_length,
        axes_tip_radius=args.axes_tip_radius,
        axes_shaft_radius=args.axes_shaft_radius,
        show_rotation_axis=args.rotation_axis,
        rotation_axis_length=args.rotation_axis_length,
        rotation_axis_color=args.rotation_axis_color,
        rotation_axis_width=args.rotation_axis_width,
        rotation_axis_tip_length=args.rotation_axis_tip_length,
        rotation_axis_tip_radius=args.rotation_axis_tip_radius,
        rotation_axis_shaft_radius=args.rotation_axis_shaft_radius,
        show_rotation_ring=args.rotation_ring,
        rotation_ring_fraction=args.rotation_ring_fraction,
        rotation_ring_radius=args.rotation_ring_radius,
        rotation_ring_radius_offset=args.rotation_ring_radius_offset,
        rotation_ring_width=args.rotation_ring_width,
        rotation_ring_color=args.rotation_ring_color,
        rotation_ring_samples=args.rotation_ring_samples,
        rotation_ring_cone_height=rotation_ring_cone_height_resolved,
        rotation_ring_cone_radius=rotation_ring_cone_radius_resolved,
        rotation_ring_cone_offset=args.rotation_ring_cone_offset,
        rotation_ring_cone_resolution=args.rotation_ring_cone_resolution,
        dash_length=args.dash_length,
        gap_length=args.gap_length,
        colormap=args.colormap,
        show_colorbar=not args.no_colorbar,
        colorbar_height=args.colorbar_height,
        colorbar_width=args.colorbar_width,
        colorbar_position_x=args.colorbar_position_x,
        colorbar_position_y=args.colorbar_position_y,
        colorbar_label_font_size=colorbar_label_font_size,
        colorbar_title_font_size=colorbar_title_font_size,
        colorbar_n_labels=args.colorbar_n_labels,
        colorbar_format=args.colorbar_format,
        enable_lighting=args.lighting,
        window_size=window_size,
        off_screen=off_screen,
    )

    if has_explicit_camera_angles(args) or args.camera_input:
        apply_camera_angle_state(plotter, camera_angle_state)

    if args.camera_input:
        add_camera_input_key(
            plotter,
            camera_angle_state,
            key=args.camera_input_key,
        )
        print(f"Interactive camera key  : {args.camera_input_key.strip()}")

    if args.save_current_view:
        add_current_view_save_key(
            plotter,
            args.output,
            args.html_output,
            key=args.save_key,
            transparent_background=args.transparent_background,
            window_size=window_size,
            camera_state=camera_angle_state,
            html_save_callback=write_html_output,
        )
        print(f"Interactive save key     : {args.save_key.strip()}")
        if args.output is not None:
            print(f"Current-view output PNG  : {args.output}")
        if args.html_output is not None:
            print(f"Current-view output HTML : {args.html_output}")
    else:
        if args.output is not None:
            output_path = save_screenshot(
                plotter,
                args.output,
                transparent_background=args.transparent_background,
                window_size=window_size,
            )
            print(f"PNG written to          : {output_path}")

        if args.html_output is not None:
            html_output_path = write_html_output(camera_angle_state_from_plotter(plotter, camera_angle_state))
            print(f"HTML written to         : {html_output_path}")

    if interactive:
        plotter.show()
    else:
        plotter.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
