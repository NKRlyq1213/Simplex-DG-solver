from __future__ import annotations

import csv
from pathlib import Path
import tempfile

import numpy as np

from examples import step10_projected_product_convergence as step10
from simplex_dg.diagnostics import (
    ProjectedProductConvergenceRow,
    attach_projected_product_rates,
    projected_product_residual_arrays,
    projected_product_residual_report,
)
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.problems import gaussian_on_sphere
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    build_full_rhs_cache,
    projected_interior_line_flux,
    projected_line_velocity,
)
from simplex_dg.trace import build_trace_cache, evaluate_face_traces


def _build_case(*, ndivs: int = 2, order: int = 4, table: str = "table1"):
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)
    omega = (
        -np.sin(-np.pi / 4.0),
        0.0,
        np.cos(-np.pi / 4.0),
    )
    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=omega,
        flux_type="central",
        lf_alpha=0.0,
        volume_form="conservative",
        project_velocity=True,
        validate=True,
    )
    return ref, mesh, geom, trace, full


def test_constant_state_exactness():
    ref, mesh, geom, trace, full = _build_case()

    q = np.full((mesh.elements.shape[0], ref.rs.shape[0]), 2.75)
    report = projected_product_residual_report(q, full.volume, full.surface, trace)

    assert report.absolute_linf < 1.0e-12
    assert report.absolute_weighted_l2 < 1.0e-12


def test_formula_consistency():
    ref, mesh, geom, trace, full = _build_case()

    q = gaussian_on_sphere(
        X=geom.X,
        center=(1.0, 0.0, 0.0),
        radius=1.0,
        sigma=0.35,
        amplitude=1.0,
    )

    arrays = projected_product_residual_arrays(q, full.volume, full.surface, trace)
    q_face = evaluate_face_traces(q, trace)
    expected = projected_interior_line_flux(q, full.volume, full.surface) - projected_line_velocity(full.volume, full.surface) * q_face

    np.testing.assert_allclose(arrays.q_face, q_face, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(arrays.residual, expected, atol=2e-13, rtol=2e-13)


def test_shapes_and_finite_values():
    ref, mesh, geom, trace, full = _build_case(ndivs=1, order=3)

    q = gaussian_on_sphere(
        X=geom.X,
        center=(1.0, 0.0, 0.0),
        radius=1.0,
        sigma=0.35,
        amplitude=1.0,
    )

    arrays = projected_product_residual_arrays(q, full.volume, full.surface, trace)
    report = projected_product_residual_report(q, full.volume, full.surface, trace)

    expected_face = (mesh.elements.shape[0], 3, ref.edge_rules[1].n_points)
    assert arrays.q_face.shape == expected_face
    assert arrays.projected_flux.shape == expected_face
    assert arrays.projected_velocity.shape == expected_face
    assert arrays.residual.shape == expected_face
    assert arrays.residual_r.shape == expected_face
    assert arrays.residual_s.shape == expected_face

    assert np.all(np.isfinite(arrays.q_face))
    assert np.all(np.isfinite(arrays.projected_flux))
    assert np.all(np.isfinite(arrays.projected_velocity))
    assert np.all(np.isfinite(arrays.residual))
    assert np.all(np.isfinite(arrays.residual_r))
    assert np.all(np.isfinite(arrays.residual_s))

    assert np.isfinite(report.absolute_weighted_l2)
    assert np.isfinite(report.relative_weighted_l2)
    assert np.isfinite(report.absolute_linf)
    assert np.isfinite(report.relative_linf)
    assert np.isfinite(report.r_absolute_weighted_l2)
    assert np.isfinite(report.s_absolute_weighted_l2)
    assert np.isfinite(report.reference_flux_weighted_l2)


def test_refinement_smoke_run_writes_csv():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        output_path = Path(tmpdir) / "projected_product.csv"

        rc = step10.main(
            [
                "--ndivs",
                "1",
                "2",
                "--order",
                "2",
                "--no-plot",
                "--output",
                str(output_path),
            ]
        )

        assert rc == 0
        assert output_path.exists()

        with output_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2

        required_columns = {
            "ndivs",
            "order",
            "n_elements",
            "total_dofs",
            "hmin",
            "absolute_weighted_l2",
            "relative_weighted_l2",
            "absolute_linf",
            "relative_linf",
            "r_absolute_weighted_l2",
            "s_absolute_weighted_l2",
            "reference_flux_weighted_l2",
            "relative_weighted_l2_rate",
            "relative_linf_rate",
        }
        assert required_columns.issubset(rows[0].keys())

        hmins = [float(row["hmin"]) for row in rows]
        assert all(np.isfinite(h) and h > 0.0 for h in hmins)
        assert hmins[1] < hmins[0]

        for row in rows:
            for field_name in required_columns - {"ndivs", "order", "n_elements", "total_dofs", "relative_weighted_l2_rate", "relative_linf_rate"}:
                value = float(row[field_name])
                assert np.isfinite(value), field_name
                assert value >= 0.0, field_name


def test_ndiv_rate_basis_uses_ndiv_ratio():
    rows = [
        ProjectedProductConvergenceRow(
            ndivs=4,
            order=4,
            n_elements=128,
            total_dofs=1000,
            hmin=0.8,
            absolute_weighted_l2=1.0,
            relative_weighted_l2=1.0 / 16.0,
            absolute_linf=1.0,
            relative_linf=1.0 / 8.0,
            r_absolute_weighted_l2=1.0,
            s_absolute_weighted_l2=1.0,
            reference_flux_weighted_l2=1.0,
        ),
        ProjectedProductConvergenceRow(
            ndivs=8,
            order=4,
            n_elements=512,
            total_dofs=4000,
            hmin=0.3,
            absolute_weighted_l2=1.0,
            relative_weighted_l2=1.0 / 256.0,
            absolute_linf=1.0,
            relative_linf=1.0 / 64.0,
            r_absolute_weighted_l2=1.0,
            s_absolute_weighted_l2=1.0,
            reference_flux_weighted_l2=1.0,
        ),
    ]

    rows_with_rates = attach_projected_product_rates(rows, rate_basis="ndiv")

    assert rows_with_rates[1].relative_weighted_l2_rate == np.log(16.0) / np.log(2.0)
    assert rows_with_rates[1].relative_linf_rate == np.log(8.0) / np.log(2.0)
