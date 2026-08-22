from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from examples.check_metric_divergence import rotation_axis_from_alpha0
from scripts import poster_sphere_exact
from simplex_dg.visualization.poster_sphere import (
    DEFAULT_ARROW_RADIUS_OFFSET,
    DEFAULT_CAMERA_AZIMUTH,
    DEFAULT_CAMERA_DISTANCE,
    DEFAULT_CAMERA_ELEVATION,
    DEFAULT_CAMERA_ROLL,
    DEFAULT_POSTER_AMPLITUDE,
    EARTH_RADIUS_METERS,
    SURFACE_ARROW_RADIUS_OFFSET,
    _inject_pyvista_html_camera_controls,
    build_exact_fields,
    camera_position_from_angles,
    format_scalar_values,
    reference_plot_triangulation,
    resolve_contour_label_position,
    resolve_rotation_axis_radii,
    resolve_sigma_physical,
    sample_velocity_arrows,
    save_interactive_html,
    scalar_color_from_colormap,
)


def test_resolve_sigma_physical_matches_existing_gaussian_convention():
    assert resolve_sigma_physical(radius=2.0, sigma_angle=0.35) == pytest.approx(0.7)
    assert resolve_sigma_physical(radius=2.0, sigma_angle=0.35, sigma_physical=0.5) == pytest.approx(0.5)


def test_build_exact_fields_uses_physical_radius_and_automatic_half_period():
    radius = EARTH_RADIUS_METERS
    alpha0 = -np.pi / 4.0
    u0 = 2.0 * np.pi / 10.0
    omega = u0 * np.asarray(rotation_axis_from_alpha0(alpha0), dtype=float)

    fields = build_exact_fields(
        ndiv=1,
        order=2,
        table="table1",
        radius=radius,
        sigma=radius * 0.35,
        amplitude=1.0,
        omega=omega,
    )

    assert fields.t_half == pytest.approx(np.pi / np.linalg.norm(omega))
    assert fields.period == pytest.approx(2.0 * np.pi / np.linalg.norm(omega))
    assert fields.sanity["geometry_node_radius_error"] <= 1.0e-8
    assert fields.sanity["element_vertex_radius_error"] <= 1.0e-8
    assert fields.sanity["velocity_tangent_relative_error"] <= 1.0e-14
    assert fields.q0.shape == fields.q_half.shape
    assert fields.t_plot == pytest.approx(fields.t_half)
    assert fields.q0.shape == fields.q_plot.shape


def test_build_exact_fields_accepts_selected_plot_days():
    radius = 1.0
    omega = np.asarray((0.0, 0.0, 1.0e-5), dtype=float)
    period = 2.0 * np.pi / np.linalg.norm(omega)
    t_plot = poster_sphere_exact.resolve_plot_time_from_days(days=4.0, period=period, period_days=12.0)
    fields = build_exact_fields(
        ndiv=1,
        order=2,
        table="table1",
        radius=radius,
        sigma=0.35,
        amplitude=1.0,
        omega=omega,
        t_plot=t_plot,
        plot_days=4.0,
    )

    assert fields.t_plot == pytest.approx(period / 3.0)
    assert fields.plot_days == pytest.approx(4.0)
    assert fields.q_plot.shape == fields.q0.shape
    assert np.linalg.norm(fields.center_plot) == pytest.approx(radius)


def test_poster_days_uses_twelve_day_period_so_six_days_is_north_pole():
    radius = 1.0
    axis = np.asarray(rotation_axis_from_alpha0(-np.pi / 4.0), dtype=float)
    omega = axis
    period = 2.0 * np.pi / np.linalg.norm(omega)
    t_plot = poster_sphere_exact.resolve_plot_time_from_days(days=6.0, period=period, period_days=12.0)

    fields = build_exact_fields(
        ndiv=1,
        order=2,
        table="table1",
        radius=radius,
        sigma=0.35,
        amplitude=1.0,
        omega=omega,
        t_plot=t_plot,
        plot_days=6.0,
    )

    assert fields.t_plot == pytest.approx(fields.t_half)
    assert fields.plot_days == pytest.approx(6.0)
    np.testing.assert_allclose(fields.center_plot, np.array([0.0, 0.0, radius]), atol=1.0e-15)


def test_reference_plot_triangulation_includes_reference_vertices():
    rs = np.array(
        [
            [-0.5, -0.5],
            [0.0, -0.5],
            [-0.5, 0.0],
        ],
        dtype=float,
    )

    plot_rs, triangles = reference_plot_triangulation(rs)

    for vertex in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)):
        assert np.any(np.all(np.isclose(plot_rs, vertex), axis=1))

    assert triangles.shape[1] == 3


def test_velocity_arrow_sampling_is_subsampled_tangent_and_normalized():
    omega = np.asarray((0.3, -0.2, 0.7), dtype=float)
    fields = build_exact_fields(
        ndiv=3,
        order=2,
        table="table1",
        radius=1.0,
        sigma=0.35,
        amplitude=1.0,
        omega=omega,
    )

    sample = sample_velocity_arrows(fields, arrow_density=10)

    assert 0 < sample.directions.shape[0] <= 10
    np.testing.assert_allclose(np.linalg.norm(sample.directions, axis=1), 1.0, atol=2.0e-15, rtol=2.0e-15)
    assert sample.speed.shape == sample.speed_fraction.shape == (sample.directions.shape[0],)
    assert np.all(sample.speed > 0.0)
    assert np.all((sample.speed_fraction > 0.0) & (sample.speed_fraction <= 1.0))
    assert sample.tangent_relative_error <= 1.0e-14


def test_format_scalar_values_uses_poster_scale_labels():
    assert format_scalar_values([200.0, 500.0, 800.0]) == "200, 500, 800"
    assert format_scalar_values([0.125, 1.0]) == "0.125, 1"


def test_scalar_color_from_colormap_uses_poster_height_scale():
    assert scalar_color_from_colormap(0.0, amplitude=DEFAULT_POSTER_AMPLITUDE, colormap="gyror") == "#00a65a"
    assert scalar_color_from_colormap(1000.0, amplitude=DEFAULT_POSTER_AMPLITUDE, colormap="gyror") == "#d62728"


def test_rotation_axis_width_resolves_to_arrow_radii():
    shaft, tip = resolve_rotation_axis_radii(axis_width=4.0)

    assert shaft == pytest.approx(0.01)
    assert tip == pytest.approx(0.032)

    shaft_explicit, tip_explicit = resolve_rotation_axis_radii(
        axis_width=4.0,
        shaft_radius=0.02,
        tip_radius=0.08,
    )

    assert shaft_explicit == pytest.approx(0.02)
    assert tip_explicit == pytest.approx(0.08)


def test_rotation_ring_cone_size_scales_cone_dimensions():
    height, radius = poster_sphere_exact.resolve_rotation_ring_cone_dimensions(
        cone_size=0.5,
        cone_height=0.18,
        cone_radius=0.080,
    )

    assert height == pytest.approx(0.09)
    assert radius == pytest.approx(0.04)


def test_camera_position_from_angles_uses_degrees_and_plot_radius_units():
    position, focal_point, view_up = camera_position_from_angles(
        azimuth=0.0,
        elevation=0.0,
        distance=2.0,
        roll=0.0,
    )

    np.testing.assert_allclose(position, np.array([2.0, 0.0, 0.0]), atol=1.0e-15)
    np.testing.assert_allclose(focal_point, np.zeros(3), atol=1.0e-15)
    assert np.linalg.norm(view_up) == pytest.approx(1.0)
    assert np.dot(np.asarray(view_up), np.asarray(focal_point) - np.asarray(position)) == pytest.approx(0.0)


def test_default_camera_angle_constants_reproduce_default_camera_position():
    position, focal_point, _view_up = camera_position_from_angles(
        azimuth=DEFAULT_CAMERA_AZIMUTH,
        elevation=DEFAULT_CAMERA_ELEVATION,
        distance=DEFAULT_CAMERA_DISTANCE,
        roll=DEFAULT_CAMERA_ROLL,
    )

    np.testing.assert_allclose(position, np.array([2.45, -2.75, 1.75]), atol=1.0e-14)
    np.testing.assert_allclose(focal_point, np.zeros(3), atol=1.0e-15)


def test_resolve_contour_label_position_defaults_below_colorbar():
    position, viewport = resolve_contour_label_position(
        "below_colorbar",
        colorbar_position_x=0.88,
        colorbar_position_y=0.28,
    )

    assert viewport is True
    assert position == pytest.approx((0.825, 0.215))


def test_poster_script_parses_expression_arguments():
    args = poster_sphere_exact.parse_args(
        [
            "--ndiv",
            "2",
            "--radius",
            "6.371e6",
            "--alpha0",
            "-pi/4",
            "--u0",
            "2*pi/10",
            "--days",
            "5",
            "--period-days",
            "12",
            "--check-only",
        ]
    )

    assert args.ndiv == 2
    assert args.radius == pytest.approx(EARTH_RADIUS_METERS)
    assert args.alpha0 == pytest.approx(-np.pi / 4.0)
    assert args.u0 == pytest.approx(2.0 * np.pi / 10.0)
    assert args.days == pytest.approx(5.0)
    assert args.period_days == pytest.approx(12.0)
    assert args.amplitude == pytest.approx(DEFAULT_POSTER_AMPLITUDE)
    assert args.lighting is False
    assert args.rotation_axis is True
    assert args.rotation_ring is True
    assert args.rotation_ring_fraction == pytest.approx(0.75)
    assert args.rotation_ring_radius == pytest.approx(0.48)
    assert args.rotation_ring_cone_size == pytest.approx(1.0)
    assert args.rotation_ring_cone_height == pytest.approx(0.18)
    assert args.rotation_ring_cone_radius == pytest.approx(0.080)
    assert args.arrow_scale_mode == "magnitude"
    assert args.save_current_view is False
    assert args.save_key == "s"
    assert args.html_output is None
    assert args.camera_input is False
    assert args.camera_input_key == "c"
    assert args.camera_azimuth is None
    assert args.camera_elevation is None
    assert args.camera_distance is None
    assert args.camera_roll is None


def test_poster_script_parses_arrow_placement_and_colorbar_size():
    args = poster_sphere_exact.parse_args(
        [
            "--arrow-placement",
            "surface",
            "--arrow-scale-mode",
            "uniform",
            "--arrow-min-scale",
            "0.2",
            "--arrow-max-scale",
            "1.25",
            "--arrow-tip-length",
            "0.38",
            "--arrow-tip-radius",
            "0.052",
            "--arrow-shaft-radius",
            "0.018",
            "--axes",
            "origin",
            "--font-size",
            "16",
            "--axes-label-font-size",
            "14",
            "--axes-length",
            "1.6",
            "--axes-label-offset",
            "0.12",
            "--axes-tip-length",
            "0.22",
            "--axes-tip-radius",
            "0.04",
            "--axes-shaft-radius",
            "0.012",
            "--no-rotation-axis",
            "--rotation-axis-length",
            "1.7",
            "--rotation-axis-width",
            "6",
            "--rotation-axis-tip-length",
            "0.24",
            "--rotation-axis-tip-radius",
            "0.05",
            "--rotation-axis-shaft-radius",
            "0.014",
            "--no-rotation-ring",
            "--rotation-ring-fraction",
            "0.7",
            "--rotation-ring-radius",
            "0.52",
            "--rotation-ring-radius-offset",
            "0.04",
            "--rotation-ring-width",
            "5",
            "--rotation-ring-color",
            "#aa0000",
            "--rotation-ring-samples",
            "90",
            "--rotation-ring-cone-size",
            "0.5",
            "--rotation-ring-cone-height",
            "0.2",
            "--rotation-ring-cone-radius",
            "0.05",
            "--rotation-ring-cone-offset",
            "0.02",
            "--rotation-ring-cone-resolution",
            "24",
            "--contour-color-mode",
            "neutral",
            "--colorbar-height",
            "0.5",
            "--colorbar-width",
            "0.1",
        ]
    )

    assert args.arrow_placement == "surface"
    assert args.arrow_scale_mode == "uniform"
    assert args.arrow_min_scale == pytest.approx(0.2)
    assert args.arrow_max_scale == pytest.approx(1.25)
    assert args.arrow_tip_length == pytest.approx(0.38)
    assert args.arrow_tip_radius == pytest.approx(0.052)
    assert args.arrow_shaft_radius == pytest.approx(0.018)
    assert args.axes == "origin"
    assert args.font_size == 16
    assert args.axes_label_font_size == 14
    assert args.axes_length == pytest.approx(1.6)
    assert args.axes_label_offset == pytest.approx(0.12)
    assert args.axes_tip_length == pytest.approx(0.22)
    assert args.axes_tip_radius == pytest.approx(0.04)
    assert args.axes_shaft_radius == pytest.approx(0.012)
    assert args.rotation_axis is False
    assert args.rotation_axis_length == pytest.approx(1.7)
    assert args.rotation_axis_width == pytest.approx(6.0)
    assert args.rotation_axis_tip_length == pytest.approx(0.24)
    assert args.rotation_axis_tip_radius == pytest.approx(0.05)
    assert args.rotation_axis_shaft_radius == pytest.approx(0.014)
    assert args.rotation_ring is False
    assert args.rotation_ring_fraction == pytest.approx(0.7)
    assert args.rotation_ring_radius == pytest.approx(0.52)
    assert args.rotation_ring_radius_offset == pytest.approx(0.04)
    assert args.rotation_ring_width == pytest.approx(5.0)
    assert args.rotation_ring_color == "#aa0000"
    assert args.rotation_ring_samples == 90
    assert args.rotation_ring_cone_size == pytest.approx(0.5)
    assert args.rotation_ring_cone_height == pytest.approx(0.2)
    assert args.rotation_ring_cone_radius == pytest.approx(0.05)
    assert args.rotation_ring_cone_offset == pytest.approx(0.02)
    assert args.rotation_ring_cone_resolution == 24
    assert args.contour_color_mode == "neutral"
    assert args.colorbar_height == pytest.approx(0.5)
    assert args.colorbar_width == pytest.approx(0.1)
    assert DEFAULT_ARROW_RADIUS_OFFSET > SURFACE_ARROW_RADIUS_OFFSET


def test_poster_script_keeps_scene_axes_as_origin_alias():
    args = poster_sphere_exact.parse_args(["--axes", "scene"])

    assert args.axes == "scene"


def test_poster_script_accepts_axes_font_size_alias():
    args = poster_sphere_exact.parse_args(["--axes-font-size", "18"])

    assert args.axes_label_font_size == 18


def test_poster_script_lighting_is_explicit_opt_in():
    default_args = poster_sphere_exact.parse_args([])
    lighting_args = poster_sphere_exact.parse_args(["--lighting"])

    assert default_args.lighting is False
    assert lighting_args.lighting is True


def test_poster_script_parses_current_view_export_options():
    args = poster_sphere_exact.parse_args(
        [
            "--save-current-view",
            "--html-output",
            "outputs/poster/manual_view.html",
            "--output",
            "outputs/poster/manual_view.png",
            "--save-key",
            "p",
            "--camera-input",
            "--camera-input-key",
            "c",
            "--camera-azimuth",
            "-45",
            "--camera-elevation",
            "30",
            "--camera-distance",
            "3.5",
            "--camera-roll",
            "10",
        ]
    )

    assert args.save_current_view is True
    assert args.output == "outputs/poster/manual_view.png"
    assert args.html_output == "outputs/poster/manual_view.html"
    assert args.save_key == "p"
    assert args.camera_input is True
    assert args.camera_input_key == "c"
    assert args.camera_azimuth == pytest.approx(-45.0)
    assert args.camera_elevation == pytest.approx(30.0)
    assert args.camera_distance == pytest.approx(3.5)
    assert args.camera_roll == pytest.approx(10.0)


def test_camera_angle_state_uses_defaults_for_unset_values():
    args = poster_sphere_exact.parse_args(["--camera-azimuth", "15"])
    state = poster_sphere_exact.resolve_camera_angle_state(args)

    assert state["azimuth"] == pytest.approx(15.0)
    assert state["elevation"] == pytest.approx(DEFAULT_CAMERA_ELEVATION)
    assert state["distance"] == pytest.approx(DEFAULT_CAMERA_DISTANCE)
    assert state["roll"] == pytest.approx(DEFAULT_CAMERA_ROLL)


def test_current_view_export_can_write_html_without_png():
    args = poster_sphere_exact.parse_args(
        [
            "--save-current-view",
            "--html-output",
            "outputs/poster/manual_view.html",
        ]
    )

    assert args.save_current_view is True
    assert args.output is None
    assert args.html_output == "outputs/poster/manual_view.html"


def test_save_interactive_html_writes_camera_angle_controls():
    pytest.importorskip("pyvista")
    pytest.importorskip("trame_vtk")

    fields = build_exact_fields(
        ndiv=1,
        order=2,
        table="table1",
        radius=1.0,
        sigma=0.35,
        amplitude=1000.0,
        omega=np.asarray((0.0, 0.0, 1.0), dtype=float),
        plot_days=6.0,
    )
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        output = Path(tmp_dir) / "poster.html"
        output_path = save_interactive_html(
            fields,
            output,
            contour_levels=[200.0, 500.0, 800.0],
            arrow_density=4,
            camera_state={
                "azimuth": -45.0,
                "elevation": 30.0,
                "distance": 3.5,
                "roll": 0.0,
            },
        )
        html = output_path.read_text(encoding="utf-8")

        assert output_path == output
    assert 'id="camera-azimuth"' in html
    assert 'id="camera-elevation"' in html
    assert 'id="camera-distance"' in html
    assert 'id="poster-vtk-content" class="content"' in html
    assert 'id="poster-camera-controls-script"' in html
    assert 'id="poster-camera-status"' in html
    assert "OfflineLocalView.load" in html
    assert "global.renderWindow" in html
    assert "window.__posterRenderWindow" in html
    assert "Renderer not ready" in html
    assert "function cameraStateFromCamera" in html
    assert "function startCameraSync" in html
    assert "function rendererViewPropCount" in html
    assert "rendererViewPropCount(item) > 0" in html
    assert "onEndAnimation" in html
    assert "applyPosterCamera(POSTER_CAMERA" not in html
    assert "function drawSurface" not in html
    assert "cdn.jsdelivr.net" not in html
    assert "OrbitControls" not in html
    assert '"azimuth":-45.0' in html
    assert html.index('id="poster-vtk-content"') < html.index('id="poster-camera-panel"')
    assert html.index('id="poster-vtk-content"') < html.index("function loadDataSet")
    reinjected = _inject_pyvista_html_camera_controls(
        html,
        camera_state={
            "azimuth": -45.0,
            "elevation": 30.0,
            "distance": 3.5,
            "roll": 0.0,
        },
        title="Poster Sphere",
    )
    assert reinjected.count('id="poster-camera-panel"') == 1


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--font-size", "0"],
        ["--axes-label-font-size", "-1"],
        ["--contour-label-font-size", "0"],
        ["--colorbar-label-font-size", "0"],
        ["--colorbar-title-font-size", "-3"],
        ["--days", "-1"],
        ["--period-days", "0"],
        ["--arrow-scale", "0"],
        ["--arrow-min-scale", "-0.1"],
        ["--arrow-max-scale", "0"],
        ["--arrow-min-scale", "1.1", "--arrow-max-scale", "1.0"],
        ["--arrow-tip-length", "0"],
        ["--arrow-tip-radius", "-0.1"],
        ["--arrow-shaft-radius", "0"],
        ["--axes-length", "0"],
        ["--axes-label-offset", "-0.1"],
        ["--axes-tip-length", "0"],
        ["--axes-tip-radius", "-0.01"],
        ["--axes-shaft-radius", "0"],
        ["--rotation-axis-length", "0"],
        ["--rotation-axis-width", "-1"],
        ["--rotation-axis-tip-length", "0"],
        ["--rotation-axis-tip-radius", "-0.1"],
        ["--rotation-axis-shaft-radius", "0"],
        ["--rotation-ring-fraction", "0"],
        ["--rotation-ring-fraction", "1.1"],
        ["--rotation-ring-radius", "0"],
        ["--rotation-ring-radius", "1.1"],
        ["--rotation-ring-radius-offset", "-0.01"],
        ["--rotation-ring-width", "0"],
        ["--rotation-ring-samples", "2"],
        ["--rotation-ring-cone-size", "0"],
        ["--rotation-ring-cone-height", "0"],
        ["--rotation-ring-cone-radius", "-0.1"],
        ["--rotation-ring-cone-offset", "-0.01"],
        ["--rotation-ring-cone-resolution", "2"],
        ["--contour-width", "0"],
        ["--save-current-view"],
        ["--save-current-view", "--output", "outputs/poster/manual_view.png", "--no-show"],
        ["--save-current-view", "--output", "outputs/poster/manual_view.png", "--off-screen"],
        ["--save-key", " "],
        ["--camera-input", "--no-show"],
        ["--camera-input", "--off-screen"],
        ["--camera-input-key", " "],
        ["--camera-distance", "0"],
        ["--camera-elevation", "90"],
        ["--camera-elevation", "-90"],
    ],
)
def test_poster_script_rejects_invalid_positive_controls(bad_args: list[str]):
    with pytest.raises(SystemExit):
        poster_sphere_exact.parse_args(bad_args)
