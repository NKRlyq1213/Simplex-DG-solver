from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from simplex_dg.mesh.manifold import ManifoldMesh


_LOCAL_FACE_VERTICES = np.array(
    [
        [1, 2],  # face 0 corresponds to reference face 1: v2 -> v3
        [2, 0],  # face 1 corresponds to reference face 2: v3 -> v1
        [0, 1],  # face 2 corresponds to reference face 3: v1 -> v2
    ],
    dtype=int,
)


@dataclass(frozen=True)
class ConnectivityCache:
    EToE: np.ndarray
    EToF: np.ndarray
    is_boundary: np.ndarray
    face_flip: np.ndarray
    face_vertex_ids: np.ndarray
    boundary_faces: np.ndarray
    interior_faces: np.ndarray


def local_face_vertex_ids(elem_vids: np.ndarray) -> np.ndarray:
    elem_vids = np.asarray(elem_vids, dtype=int).reshape(-1)

    if elem_vids.shape != (3,):
        raise ValueError("elem_vids must have shape (3,).")

    return elem_vids[_LOCAL_FACE_VERTICES]


def all_face_vertex_ids(elements: np.ndarray) -> np.ndarray:
    elements = np.asarray(elements, dtype=int)

    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("elements must have shape (K, 3).")

    out = np.zeros((elements.shape[0], 3, 2), dtype=int)

    for k in range(elements.shape[0]):
        out[k] = local_face_vertex_ids(elements[k])

    return out


def build_connectivity_cache(elements: np.ndarray, validate: bool = True) -> ConnectivityCache:
    elements = np.asarray(elements, dtype=int)

    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("elements must have shape (K, 3).")

    K = elements.shape[0]

    face_vids = all_face_vertex_ids(elements)

    EToE = -np.ones((K, 3), dtype=int)
    EToF = -np.ones((K, 3), dtype=int)
    is_boundary = np.ones((K, 3), dtype=bool)
    face_flip = np.zeros((K, 3), dtype=bool)

    edge_map: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)

    for k in range(K):
        for f in range(3):
            va, vb = face_vids[k, f]
            key = (min(int(va), int(vb)), max(int(va), int(vb)))
            edge_map[key].append((k, f, int(va), int(vb)))

    boundary_faces: list[tuple[int, int]] = []
    interior_faces: list[tuple[int, int, int, int]] = []

    for key, entries in edge_map.items():
        if len(entries) == 1:
            k, f, _, _ = entries[0]
            boundary_faces.append((k, f))
            continue

        if len(entries) != 2:
            raise ValueError(f"Invalid non-manifold edge {key}: shared by {len(entries)} elements.")

        (k1, f1, va1, vb1), (k2, f2, va2, vb2) = entries

        EToE[k1, f1] = k2
        EToF[k1, f1] = f2
        is_boundary[k1, f1] = False

        EToE[k2, f2] = k1
        EToF[k2, f2] = f1
        is_boundary[k2, f2] = False

        if va1 == vb2 and vb1 == va2:
            flip = True
        elif va1 == va2 and vb1 == vb2:
            flip = False
        else:
            raise ValueError("Canonical edge matched but oriented vertices are inconsistent.")

        face_flip[k1, f1] = flip
        face_flip[k2, f2] = flip

        interior_faces.append((k1, f1, k2, f2))

    boundary_arr = np.asarray(boundary_faces, dtype=int)
    if boundary_arr.size == 0:
        boundary_arr = boundary_arr.reshape(0, 2)

    interior_arr = np.asarray(interior_faces, dtype=int)
    if interior_arr.size == 0:
        interior_arr = interior_arr.reshape(0, 4)

    cache = ConnectivityCache(
        EToE=EToE,
        EToF=EToF,
        is_boundary=is_boundary,
        face_flip=face_flip,
        face_vertex_ids=face_vids,
        boundary_faces=boundary_arr,
        interior_faces=interior_arr,
    )

    if validate:
        validate_connectivity_cache(elements, cache)

    return cache


def build_connectivity_cache_from_mesh(mesh: ManifoldMesh, validate: bool = True) -> ConnectivityCache:
    return build_connectivity_cache(mesh.elements, validate=validate)


def validate_connectivity_cache(elements: np.ndarray, cache: ConnectivityCache) -> None:
    elements = np.asarray(elements, dtype=int)
    K = elements.shape[0]

    EToE = np.asarray(cache.EToE, dtype=int)
    EToF = np.asarray(cache.EToF, dtype=int)
    is_boundary = np.asarray(cache.is_boundary, dtype=bool)
    face_flip = np.asarray(cache.face_flip, dtype=bool)
    face_vids = np.asarray(cache.face_vertex_ids, dtype=int)

    if EToE.shape != (K, 3):
        raise ValueError("EToE must have shape (K, 3).")

    if EToF.shape != (K, 3):
        raise ValueError("EToF must have shape (K, 3).")

    if is_boundary.shape != (K, 3):
        raise ValueError("is_boundary must have shape (K, 3).")

    if face_flip.shape != (K, 3):
        raise ValueError("face_flip must have shape (K, 3).")

    if face_vids.shape != (K, 3, 2):
        raise ValueError("face_vertex_ids must have shape (K, 3, 2).")

    seen_pairs = set()

    for k in range(K):
        for f in range(3):
            nbr = EToE[k, f]
            nbr_f = EToF[k, f]

            if is_boundary[k, f]:
                if nbr != -1 or nbr_f != -1:
                    raise ValueError(f"Boundary face ({k}, {f}) should have EToE=EToF=-1.")
                continue

            if not (0 <= nbr < K):
                raise ValueError(f"Interior face ({k}, {f}) has invalid neighbor {nbr}.")

            if nbr_f not in (0, 1, 2):
                raise ValueError(f"Interior face ({k}, {f}) has invalid neighbor face {nbr_f}.")

            if EToE[nbr, nbr_f] != k:
                raise ValueError(f"EToE symmetry failure at ({k}, {f}).")

            if EToF[nbr, nbr_f] != f:
                raise ValueError(f"EToF symmetry failure at ({k}, {f}).")

            va, vb = face_vids[k, f]
            vc, vd = face_vids[nbr, nbr_f]

            key1 = (min(int(va), int(vb)), max(int(va), int(vb)))
            key2 = (min(int(vc), int(vd)), max(int(vc), int(vd)))

            if key1 != key2:
                raise ValueError(f"Paired faces ({k}, {f}) and ({nbr}, {nbr_f}) do not share the same edge.")

            should_flip = bool(va == vd and vb == vc)

            if bool(face_flip[k, f]) != should_flip:
                raise ValueError(f"face_flip mismatch at ({k}, {f}).")

            if bool(face_flip[nbr, nbr_f]) != should_flip:
                raise ValueError(f"face_flip symmetry mismatch at ({nbr}, {nbr_f}).")

            pair_key = tuple(sorted(((int(k), int(f)), (int(nbr), int(nbr_f)))))
            seen_pairs.add(pair_key)

    n_total_local_faces = 3 * K
    n_boundary_faces = int(np.sum(is_boundary))
    n_unique_interior_faces = len(seen_pairs)

    if n_total_local_faces != 2 * n_unique_interior_faces + n_boundary_faces:
        raise ValueError("Face counting identity failed.")