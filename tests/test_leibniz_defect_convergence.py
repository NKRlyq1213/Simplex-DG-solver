from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.diagnostics.leibniz import (
    LeibnizDefectRow,
    attach_rates,
    compute_leibniz_defect_component,
    compute_leibniz_defects,
    observed_rate,
    project_nodal_samples,
    projection_closure_linf,
)
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.problems import gaussian_on_sphere
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import apply_reference_operator, build_volume_rhs_cache


def _build_case(*, ndivs: int = 2, order: int = 4, table: str = "table1"):
    ref = build_reference_cache(order=order, table=table)
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    geom = build_geometry_cache(mesh, ref, validate=True)
    volume = build_volume_rhs_cache(
        ref=ref,
        geom=geom,
        omega=(-np.sin(-np.pi / 4.0), 0.0, np.cos(-np.pi / 4.0)),
        project_velocity=True,
        validate=True,
    )
    return ref, geom, volume


def test_constant_state_identity_is_zero_to_roundoff():
    ref, geom, volume = _build_case()

    q_h = 1.75 * np.ones((volume.n_elements, volume.n_points))
    defects = compute_leibniz_defects(q_h, volume)

    assert defects.tau_r.shape == (volume.n_elements, volume.n_points)
    assert defects.tau_s.shape == (volume.n_elements, volume.n_points)
    assert np.max(np.abs(defects.tau_r)) < 1.0e-12
    assert np.max(np.abs(defects.tau_s)) < 1.0e-12


def test_constant_coefficient_identity_is_zero_to_roundoff():
    ref, geom, volume = _build_case()

    q_raw = geom.X[:, :, 0] - 0.25 * geom.X[:, :, 1] + 0.1 * geom.X[:, :, 2]
    q_h = project_nodal_samples(q_raw, ref)

    alpha = 2.5 * np.ones_like(q_h)
    beta = -0.75 * np.ones_like(q_h)
    Dr_alpha = apply_reference_operator(volume.Dr, alpha)
    Ds_beta = apply_reference_operator(volume.Ds, beta)

    tau_r = compute_leibniz_defect_component(volume.Dr, alpha, Dr_alpha, q_h)
    tau_s = compute_leibniz_defect_component(volume.Ds, beta, Ds_beta, q_h)

    assert np.max(np.abs(Dr_alpha)) < 1.0e-12
    assert np.max(np.abs(Ds_beta)) < 1.0e-12
    assert np.max(np.abs(tau_r)) < 1.0e-12
    assert np.max(np.abs(tau_s)) < 1.0e-12


def test_polynomial_exactness_case_satisfies_product_rule():
    ref = build_reference_cache(order=3, table="table1")

    r = ref.rs[:, 0]
    s = ref.rs[:, 1]

    coeff = (1.0 + 0.2 * r - 0.1 * s)[None, :]
    q_h = (0.7 - 0.3 * r + 0.4 * s)[None, :]

    Dr_coeff = apply_reference_operator(ref.Dr, coeff)
    Ds_coeff = apply_reference_operator(ref.Ds, coeff)

    tau_r = compute_leibniz_defect_component(ref.Dr, coeff, Dr_coeff, q_h)
    tau_s = compute_leibniz_defect_component(ref.Ds, coeff, Ds_coeff, q_h)

    assert np.max(np.abs(tau_r)) < 1.0e-12
    assert np.max(np.abs(tau_s)) < 1.0e-12


def test_projected_gaussian_closure_is_near_machine_precision():
    ref, geom, _ = _build_case(ndivs=1, order=4)

    q_raw = gaussian_on_sphere(
        X=geom.X,
        center=(1.0, 0.0, 0.0),
        radius=1.0,
        sigma=0.35,
        amplitude=1.0,
    )
    q_h = project_nodal_samples(q_raw, ref)
    closure = projection_closure_linf(q_h, ref)

    assert q_h.shape == q_raw.shape
    assert closure < 1.0e-12


def test_observed_rate_uses_actual_h_ratio():
    rate = observed_rate(8.0, 2.0, 0.9, 0.3)

    assert rate == pytest.approx(np.log(4.0) / np.log(3.0))
    assert rate != pytest.approx(2.0)


def test_attach_rates_can_use_ndiv_basis():
    rows = [
        LeibnizDefectRow(
            order=4,
            table="table1",
            state="projected-gaussian",
            ndivs=4,
            n_elements=128,
            n_points_per_element=22,
            total_dofs=2816,
            hmin=0.8,
            q_projection_closure_linf=0.0,
            tau_r_linf=1.0,
            tau_r_l2_ref=1.0,
            tau_s_linf=1.0,
            tau_s_l2_ref=1.0,
            tau_sum_linf=1.0 / 16.0,
            tau_sum_l2_ref=1.0 / 16.0,
            physical_tau_r_linf=1.0,
            physical_tau_r_l2=1.0,
            physical_tau_s_linf=1.0,
            physical_tau_s_l2=1.0,
            physical_tau_sum_linf=1.0 / 8.0,
            physical_tau_sum_l2=1.0 / 16.0,
            energy=1.0,
            abs_energy_residual=1.0,
            relative_energy_residual=1.0 / 4.0,
        ),
        LeibnizDefectRow(
            order=4,
            table="table1",
            state="projected-gaussian",
            ndivs=8,
            n_elements=512,
            n_points_per_element=22,
            total_dofs=11264,
            hmin=0.3,
            q_projection_closure_linf=0.0,
            tau_r_linf=1.0,
            tau_r_l2_ref=1.0,
            tau_s_linf=1.0,
            tau_s_l2_ref=1.0,
            tau_sum_linf=1.0 / 256.0,
            tau_sum_l2_ref=1.0 / 256.0,
            physical_tau_r_linf=1.0,
            physical_tau_r_l2=1.0,
            physical_tau_s_linf=1.0,
            physical_tau_s_l2=1.0,
            physical_tau_sum_linf=1.0 / 64.0,
            physical_tau_sum_l2=1.0 / 256.0,
            energy=1.0,
            abs_energy_residual=1.0,
            relative_energy_residual=1.0 / 16.0,
        ),
    ]

    rows_with_rates = attach_rates(rows, rate_basis="ndiv")

    assert rows_with_rates[1].tau_sum_l2_ref_rate == pytest.approx(4.0)
    assert rows_with_rates[1].physical_tau_sum_linf_rate == pytest.approx(3.0)
    assert rows_with_rates[1].relative_energy_residual_rate == pytest.approx(2.0)
