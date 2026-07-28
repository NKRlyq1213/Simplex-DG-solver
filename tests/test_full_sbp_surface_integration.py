from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    build_full_rhs_cache,
    build_surface_rhs_cache,
    full_rhs,
    projected_interior_line_flux,
    projected_line_velocity,
)
from simplex_dg.trace import build_trace_cache, pair_face_traces


OMEGA = np.array([0.25, -0.15, 0.45], dtype=float)


def _build_case(
    *,
    table: str = "table1",
    sbp_variant: str = "projected",
    order: int = 4,
    ndivs: int = 1,
    flux_type: str = "central",
    volume_form: str = "conservative",
):
    ref = build_reference_cache(order=order, table=table, validate=True, sbp_variant=sbp_variant)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    geom = build_geometry_cache(mesh, ref, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)
    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=OMEGA,
        flux_type=flux_type,
        lf_alpha=1.0,
        volume_form=volume_form,
        validate=True,
    )
    return ref, mesh, conn, geom, trace, full


@pytest.mark.parametrize(
    ("table", "sbp_variant"),
    [
        ("table1", "projected"),
        ("table2", "projected"),
        ("table1", "full-raw"),
        ("table1", "full-orth"),
    ],
)
def test_surface_cache_uses_reference_lift_and_trace_operator(table: str, sbp_variant: str):
    ref, mesh, conn, geom, trace, full = _build_case(table=table, sbp_variant=sbp_variant)

    np.testing.assert_allclose(full.surface.face_interp, trace.face_interp, atol=0.0, rtol=0.0)

    for face_id in (1, 2, 3):
        np.testing.assert_allclose(
            full.surface.lift[face_id - 1],
            ref.face_lift[face_id],
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize("sbp_variant", ["full-raw", "full-orth"])
def test_full_surface_product_identity_uses_direct_trace(sbp_variant: str):
    ref, mesh, conn, geom, trace, full = _build_case(table="table1", sbp_variant=sbp_variant)
    rng = np.random.default_rng(20260728)

    alpha = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    beta = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))
    q = rng.standard_normal((mesh.elements.shape[0], ref.rs.shape[0]))

    for f in range(trace.n_faces):
        E = full.surface.face_interp[f]
        np.testing.assert_allclose((alpha * q) @ E.T, (alpha @ E.T) * (q @ E.T), atol=0.0, rtol=0.0)
        np.testing.assert_allclose((beta * q) @ E.T, (beta @ E.T) * (q @ E.T), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("sbp_variant", ["full-raw", "full-orth"])
def test_full_direct_conservative_and_split_interior_boundary_terms_match(sbp_variant: str):
    ref, mesh, conn, geom, trace, full = _build_case(table="table1", sbp_variant=sbp_variant)
    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]
    traces = pair_face_traces(q, trace, use_numba=False)

    line_flux_cons = projected_interior_line_flux(q, full.volume, full.surface)
    line_velocity_m = projected_line_velocity(full.volume, full.surface)
    line_flux_split = 0.5 * line_flux_cons + 0.5 * line_velocity_m * traces.qM

    np.testing.assert_allclose(line_flux_split, line_flux_cons, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_single_rhs_full_raw_and_orthogonalized_agree(volume_form: str):
    ref_raw, mesh, conn, geom_raw, trace_raw, full_raw = _build_case(
        table="table1",
        sbp_variant="full-raw",
        volume_form=volume_form,
    )
    ref_orth, _, _, geom_orth, trace_orth, full_orth = _build_case(
        table="table1",
        sbp_variant="full-orth",
        volume_form=volume_form,
    )

    q = geom_raw.X[:, :, 0] - 0.4 * geom_raw.X[:, :, 1] + 0.2 * geom_raw.X[:, :, 2]

    rhs_raw = full_rhs(q, full_raw, use_numba=False)
    rhs_orth = full_rhs(q, full_orth, use_numba=False)

    np.testing.assert_allclose(rhs_orth, rhs_raw, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(geom_orth.X, geom_raw.X, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(trace_orth.face_interp, trace_raw.face_interp, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("volume_form", ["conservative", "split"])
def test_constant_state_single_rhs_is_finite_and_raw_orth_match(volume_form: str):
    ref_proj, mesh, conn, geom_proj, trace_proj, full_proj = _build_case(
        table="table1",
        sbp_variant="projected",
        volume_form=volume_form,
    )
    ref_raw, _, _, geom_raw, trace_raw, full_raw = _build_case(
        table="table1",
        sbp_variant="full-raw",
        volume_form=volume_form,
    )
    ref_orth, _, _, geom_orth, trace_orth, full_orth = _build_case(
        table="table1",
        sbp_variant="full-orth",
        volume_form=volume_form,
    )

    q = np.ones((mesh.elements.shape[0], ref_proj.rs.shape[0]), dtype=float)

    rhs_proj = full_rhs(q, full_proj, use_numba=False)
    rhs_raw = full_rhs(q, full_raw, use_numba=False)
    rhs_orth = full_rhs(q, full_orth, use_numba=False)

    assert np.all(np.isfinite(rhs_proj))
    assert np.all(np.isfinite(rhs_raw))
    assert np.all(np.isfinite(rhs_orth))

    np.testing.assert_allclose(rhs_orth, rhs_raw, atol=1e-11, rtol=1e-11)
