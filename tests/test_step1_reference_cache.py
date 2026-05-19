import numpy as np

from simplex_dg.reference import build_reference_cache, mode_indices_2d, num_modes_2d


def test_num_modes_2d():
    assert num_modes_2d(0) == 1
    assert num_modes_2d(1) == 3
    assert num_modes_2d(2) == 6
    assert num_modes_2d(4) == 15


def test_mode_indices_2d():
    assert mode_indices_2d(2) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (2, 0),
    ]


def test_reference_cache_table1_order4_shapes():
    cache = build_reference_cache(order=4, table="table1")

    np_points = cache.rs.shape[0]
    nmodes = num_modes_2d(4)

    assert cache.V.shape == (np_points, nmodes)
    assert cache.Vr.shape == (np_points, nmodes)
    assert cache.Vs.shape == (np_points, nmodes)

    assert cache.M.shape == (nmodes, nmodes)
    assert cache.Dr.shape == (np_points, np_points)
    assert cache.Ds.shape == (np_points, np_points)

    for face_id in (1, 2, 3):
        assert cache.face_interp[face_id].shape == (5, np_points)


def test_reference_cache_table2_order4_shapes():
    cache = build_reference_cache(order=4, table="table2")

    np_points = cache.rs.shape[0]
    nmodes = num_modes_2d(4)

    assert cache.V.shape == (np_points, nmodes)
    assert cache.M.shape == (nmodes, nmodes)

    for face_id in (1, 2, 3):
        assert cache.face_interp[face_id].shape == (5, np_points)


def test_mass_matrix_symmetric():
    cache = build_reference_cache(order=3, table="table1")

    assert np.allclose(cache.M, cache.M.T, atol=1e-12, rtol=1e-12)


def test_derivative_constant_zero():
    cache = build_reference_cache(order=3, table="table1")

    ones = np.ones(cache.rs.shape[0])

    assert np.allclose(cache.Dr @ ones, 0.0, atol=1e-8)
    assert np.allclose(cache.Ds @ ones, 0.0, atol=1e-8)