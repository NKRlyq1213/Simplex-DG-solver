from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ManifoldMesh:
    vertices: np.ndarray
    elements: np.ndarray
    radius: float
    ndivs: int
    element_patch_ids: np.ndarray


def normalize_vectors(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    nrm = np.linalg.norm(x, axis=-1, keepdims=True)

    if np.any(nrm <= 0.0):
        raise ValueError("Cannot normalize zero vector.")

    return radius * x / nrm


def _orient_elements_outward(vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    elements = np.asarray(elements, dtype=int).copy()

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]

    normals = np.cross(p1 - p0, p2 - p0)
    centroids = (p0 + p1 + p2) / 3.0

    signed = np.einsum("ij,ij->i", normals, centroids)
    bad = signed < 0.0

    if np.any(bad):
        tmp = elements[bad, 1].copy()
        elements[bad, 1] = elements[bad, 2]
        elements[bad, 2] = tmp

    return elements


def _base_octahedron(radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )

    vertices = radius * vertices

    elements = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ],
        dtype=int,
    )

    elements = _orient_elements_outward(vertices, elements)

    return vertices, elements


def build_octa_sphere_mesh(ndivs: int, radius: float = 1.0, round_decimals: int = 14) -> ManifoldMesh:
    if ndivs < 1:
        raise ValueError("ndivs must be >= 1.")

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    base_vertices, base_elements = _base_octahedron(radius=radius)

    if ndivs == 1:
        element_patch_ids = np.arange(base_elements.shape[0], dtype=int)

        mesh = ManifoldMesh(
            vertices=base_vertices,
            elements=base_elements,
            radius=float(radius),
            ndivs=int(ndivs),
            element_patch_ids=element_patch_ids,
        )

        validate_manifold_mesh(mesh)

        return mesh

    n = int(ndivs)

    vertices: list[np.ndarray] = []
    vertex_map: dict[tuple[float, float, float], int] = {}

    def add_vertex(p: np.ndarray) -> int:
        p = normalize_vectors(np.asarray(p, dtype=float), radius=radius).reshape(3)
        key = tuple(np.round(p, round_decimals))

        if key in vertex_map:
            return vertex_map[key]

        idx = len(vertices)
        vertex_map[key] = idx
        vertices.append(p)

        return idx

    elements: list[list[int]] = []
    patch_ids: list[int] = []

    for patch_id, tri in enumerate(base_elements):
        a = base_vertices[tri[0]]
        b = base_vertices[tri[1]]
        c = base_vertices[tri[2]]

        local: dict[tuple[int, int], int] = {}

        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                p = (i * a + j * b + k * c) / n
                local[(i, j)] = add_vertex(p)

        for i in range(n):
            for j in range(n - i):
                v0 = local[(i, j)]
                v1 = local[(i + 1, j)]
                v2 = local[(i, j + 1)]

                elements.append([v0, v1, v2])
                patch_ids.append(patch_id)

                if i + j < n - 1:
                    v3 = local[(i + 1, j + 1)]
                    elements.append([v1, v3, v2])
                    patch_ids.append(patch_id)

    vertices_arr = np.asarray(vertices, dtype=float)
    elements_arr = np.asarray(elements, dtype=int)
    elements_arr = _orient_elements_outward(vertices_arr, elements_arr)

    mesh = ManifoldMesh(
        vertices=vertices_arr,
        elements=elements_arr,
        radius=float(radius),
        ndivs=int(ndivs),
        element_patch_ids=np.asarray(patch_ids, dtype=int),
    )

    validate_manifold_mesh(mesh)

    return mesh


def triangle_outward_signed_area_proxy(vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    elements = np.asarray(elements, dtype=int)

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]

    normals = np.cross(p1 - p0, p2 - p0)
    centroids = (p0 + p1 + p2) / 3.0

    return np.einsum("ij,ij->i", normals, centroids)


def validate_manifold_mesh(mesh: ManifoldMesh, tol: float = 1e-12) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    elements = np.asarray(mesh.elements, dtype=int)
    patch_ids = np.asarray(mesh.element_patch_ids, dtype=int)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh.vertices must have shape (Nv, 3).")

    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("mesh.elements must have shape (K, 3).")

    if patch_ids.shape != (elements.shape[0],):
        raise ValueError("mesh.element_patch_ids must have shape (K,).")

    if np.any(elements < 0) or np.any(elements >= vertices.shape[0]):
        raise ValueError("mesh.elements contains invalid vertex indices.")

    r = np.linalg.norm(vertices, axis=1)

    if not np.allclose(r, mesh.radius, atol=tol, rtol=tol):
        raise ValueError("Not all vertices lie on the requested sphere radius.")

    signed = triangle_outward_signed_area_proxy(vertices, elements)

    if np.any(signed <= tol):
        bad = np.where(signed <= tol)[0]
        raise ValueError(f"Found non-outward or degenerate elements: {bad.tolist()}")
