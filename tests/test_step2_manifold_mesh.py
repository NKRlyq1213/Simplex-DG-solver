import numpy as np

from simplex_dg.mesh import (
    build_connectivity_cache_from_mesh,
    build_octa_sphere_mesh,
    triangle_outward_signed_area_proxy,
)


def test_ndivs1_octa_sphere_counts():
    mesh = build_octa_sphere_mesh(ndivs=1, radius=1.0)

    assert mesh.vertices.shape == (6, 3)
    assert mesh.elements.shape == (8, 3)
    assert mesh.element_patch_ids.shape == (8,)

    r = np.linalg.norm(mesh.vertices, axis=1)
    assert np.allclose(r, 1.0)


def test_refined_octa_sphere_triangle_count():
    for ndivs in (2, 4, 8):
        mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)

        expected_k = 8 * (ndivs**2)
        assert mesh.elements.shape[0] == expected_k

        r = np.linalg.norm(mesh.vertices, axis=1)
        assert np.allclose(r, 1.0)


def test_non_power_of_two_ndivs_triangle_count():
    for ndivs in (3, 6):
        mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)

        assert mesh.ndivs == ndivs
        assert mesh.elements.shape[0] == 8 * (ndivs**2)


def test_all_triangles_outward_oriented():
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    signed = triangle_outward_signed_area_proxy(mesh.vertices, mesh.elements)

    assert np.all(signed > 0.0)


def test_closed_sphere_has_no_boundary_faces():
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)

    assert conn.boundary_faces.shape == (0, 2)
    assert np.count_nonzero(conn.is_boundary) == 0


def test_connectivity_face_count_identity_closed_sphere():
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)

    k = mesh.elements.shape[0]
    n_unique_interior_faces = conn.interior_faces.shape[0]

    assert 3 * k == 2 * n_unique_interior_faces


def test_connectivity_symmetry():
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)

    for k in range(mesh.elements.shape[0]):
        for f in range(3):
            nbr = conn.EToE[k, f]
            nbr_f = conn.EToF[k, f]

            assert nbr >= 0
            assert nbr_f >= 0

            assert conn.EToE[nbr, nbr_f] == k
            assert conn.EToF[nbr, nbr_f] == f
