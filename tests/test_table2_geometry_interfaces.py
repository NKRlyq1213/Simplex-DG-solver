from __future__ import annotations

import numpy as np
import pytest

from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache


@pytest.fixture(scope="module")
def table2_geometry_interface_case():
    ref = build_reference_cache(
        order=4,
        table="table2",
        n_face=5,
        validate=True,
    )
    mesh = build_octa_sphere_mesh(
        ndivs=4,
        radius=1.0,
    )
    geom = build_geometry_cache(
        mesh=mesh,
        ref=ref,
        validate=True,
    )
    conn = build_connectivity_cache_from_mesh(mesh, validate=True)
    return mesh, ref, geom, conn


def test_table2_shared_face_geometry_matches(table2_geometry_interface_case):
    mesh, ref, geom, conn = table2_geometry_interface_case

    max_coord = 0.0
    max_jac = 0.0
    max_normal = 0.0
    max_tangent_sum = 0.0
    max_conormal_sum = 0.0

    for k, f, nbr, nbr_f in conn.interior_faces:
        x_minus = geom.X_face[k, f]
        x_plus = geom.X_face[nbr, nbr_f]
        j_plus = geom.face_jacobian[nbr, nbr_f]
        n_plus = geom.face_normal[nbr, nbr_f]
        t_plus = geom.face_tangent[nbr, nbr_f]
        c_plus = geom.face_conormal[nbr, nbr_f]

        if conn.face_flip[k, f]:
            x_plus = x_plus[::-1]
            j_plus = j_plus[::-1]
            n_plus = n_plus[::-1]
            t_plus = t_plus[::-1]
            c_plus = c_plus[::-1]

        np.testing.assert_allclose(x_minus, x_plus, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(geom.face_jacobian[k, f], j_plus, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(geom.face_normal[k, f], n_plus, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(geom.face_tangent[k, f], -t_plus, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(geom.face_conormal[k, f], -c_plus, atol=2e-13, rtol=2e-13)

        max_coord = max(max_coord, float(np.max(np.linalg.norm(x_minus - x_plus, axis=1))))
        max_jac = max(max_jac, float(np.max(np.abs(geom.face_jacobian[k, f] - j_plus))))
        max_normal = max(max_normal, float(np.max(np.abs(geom.face_normal[k, f] - n_plus))))
        max_tangent_sum = max(max_tangent_sum, float(np.max(np.abs(geom.face_tangent[k, f] + t_plus))))
        max_conormal_sum = max(max_conormal_sum, float(np.max(np.abs(geom.face_conormal[k, f] + c_plus))))

    assert max_coord < 2e-13
    assert max_jac < 2e-13
    assert max_normal < 2e-13
    assert max_tangent_sum < 2e-13
    assert max_conormal_sum < 2e-13
