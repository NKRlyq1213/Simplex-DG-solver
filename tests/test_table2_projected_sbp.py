from __future__ import annotations

import numpy as np
import pytest

from examples.check_projected_sbp import (
    projected_sbp_boundary_matrices,
    projected_sbp_lhs_matrices,
)
from simplex_dg.reference import build_reference_cache


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_table2_projected_sbp_identity(order: int):
    ref = build_reference_cache(
        order=order,
        table="table2",
        n_face=order + 1,
        validate=True,
    )

    lhs_r, lhs_s = projected_sbp_lhs_matrices(ref)
    boundary_r, boundary_s = projected_sbp_boundary_matrices(ref)

    residual_r = lhs_r - boundary_r
    residual_s = lhs_s - boundary_s

    residual_r_inf = np.linalg.norm(residual_r, ord=np.inf)
    residual_s_inf = np.linalg.norm(residual_s, ord=np.inf)

    assert np.allclose(lhs_r, boundary_r, atol=1e-10, rtol=1e-10), (
        f"order={order}, ||L_r - B_r||_inf={residual_r_inf:.6e}"
    )
    assert np.allclose(lhs_s, boundary_s, atol=1e-10, rtol=1e-10), (
        f"order={order}, ||L_s - B_s||_inf={residual_s_inf:.6e}"
    )
