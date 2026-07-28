from __future__ import annotations

import numpy as np
import pytest

from examples import step9_gaussian_convergence as step9
from simplex_dg.reference import build_reference_cache, vandermonde2d
from simplex_dg.trace import build_trace_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh


def test_reference_cache_projected_operators_and_face_interp_unchanged():
    for table in ("table1", "table2"):
        ref = build_reference_cache(order=4, table=table, validate=True)

        np.testing.assert_allclose(ref.Dr, ref.Vr @ ref.projection, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(ref.Ds, ref.Vs @ ref.projection, atol=2e-12, rtol=2e-12)

        for face_id in (1, 2, 3):
            edge = ref.edge_rules[face_id]
            v_face = vandermonde2d(ref.order, edge.rs[:, 0], edge.rs[:, 1])
            np.testing.assert_allclose(
                ref.face_interp[face_id],
                v_face @ ref.projection,
                atol=2e-12,
                rtol=2e-12,
            )


def test_trace_cache_copies_projected_face_interp_without_direct_extraction():
    ref = build_reference_cache(order=4, table="table1", validate=True)
    mesh = build_octa_sphere_mesh(ndivs=1, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    trace = build_trace_cache(ref, conn, validate=True)

    for face_id in (1, 2, 3):
        np.testing.assert_allclose(
            trace.face_interp[face_id - 1],
            ref.face_interp[face_id],
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    ("table", "volume_form", "expected"),
    [
        (
            "table1",
            "conservative",
            np.array(
                [
                    0.05,
                    0.013972659705298884,
                    0.022020371486561878,
                    0.05608277774095505,
                    0.0,
                    0.0002776097563683524,
                    -0.04554999726018237,
                    1.0441709042315452,
                    17.898931910756154,
                    3.7668263003976152,
                ],
                dtype=float,
            ),
        ),
        (
            "table1",
            "split",
            np.array(
                [
                    0.05,
                    0.011566147827302713,
                    0.018227801807061062,
                    0.04241268638583595,
                    0.0,
                    -2.2438052446337594e-05,
                    -0.03411529147771565,
                    1.0305008128764261,
                    17.896547728282314,
                    3.764787334990107,
                ],
                dtype=float,
            ),
        ),
        (
            "table2",
            "conservative",
            np.array(
                [
                    0.05,
                    0.013404201242605442,
                    0.021212539086812685,
                    0.021435485803986132,
                    1.5144980251905088e-16,
                    0.00011596919274839888,
                    -0.019553122867853382,
                    0.9789240284450123,
                    11.954147790218164,
                    2.883048769875402,
                ],
                dtype=float,
            ),
        ),
        (
            "table2",
            "split",
            np.array(
                [
                    0.05,
                    0.011996537252685114,
                    0.01898486980112837,
                    0.020580988976702796,
                    3.0289960503810176e-16,
                    -1.7492288370024655e-05,
                    -0.015494445138187046,
                    0.9788433867874665,
                    11.954126671286337,
                    2.882876536023401,
                ],
                dtype=float,
            ),
        ),
    ],
)
def test_step9_central_short_run_regression(table: str, volume_form: str, expected: np.ndarray):
    result = step9.run_one_ndiv(
        ndivs=1,
        order=4,
        table=table,
        cfl=0.5,
        tf=0.05,
        sigma=0.35,
        radius=1.0,
        amplitude=1.0,
        alpha0=-np.pi / 4.0,
        u0=2.0 * np.pi / 10.0,
        lf_alpha=1.0,
        flux_type="central",
        volume_form=volume_form,
        use_numba=False,
        history_every=1,
    )

    actual = np.array(
        [
            result.row.dt,
            result.row.l2_error,
            result.row.relative_l2_error,
            result.row.linf_error,
            result.row.relative_mass_drift,
            result.row.relative_energy_drift,
            result.row.q_min,
            result.row.q_max,
            float(result.q_final.sum()),
            float(np.linalg.norm(result.q_final)),
        ],
        dtype=float,
    )

    np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=5e-12)
