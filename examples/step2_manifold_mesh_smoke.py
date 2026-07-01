from __future__ import annotations

import numpy as np

from simplex_dg.backends import backend_status
from simplex_dg.mesh import (
    build_connectivity_cache_from_mesh,
    build_octa_sphere_mesh,
    triangle_outward_signed_area_proxy,
)


def main() -> None:
    status = backend_status()

    print("Backend status")
    print("--------------")
    print(f"Numba available: {status.numba_available}")
    print(f"JAX available  : {status.jax_available}")
    print(f"JAX devices    : {status.jax_devices}")
    print()

    for ndivs in (1, 2, 4, 8):
        mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
        conn = build_connectivity_cache_from_mesh(mesh)

        signed = triangle_outward_signed_area_proxy(mesh.vertices, mesh.elements)
        radius_error = np.max(np.abs(np.linalg.norm(mesh.vertices, axis=1) - mesh.radius))

        print(f"ndivs = {ndivs}")
        print(f"  Nv                    : {mesh.vertices.shape[0]}")
        print(f"  K                     : {mesh.elements.shape[0]}")
        print(f"  expected K            : {8 * (ndivs ** 2)}")
        print(f"  boundary faces        : {conn.boundary_faces.shape[0]}")
        print(f"  unique interior faces : {conn.interior_faces.shape[0]}")
        print(f"  max radius error      : {radius_error:.6e}")
        print(f"  min outward proxy     : {signed.min():.6e}")
        print()

    if status.numba_available:
        import numba

        @numba.njit(cache=True)
        def count_elements(elements):
            return elements.shape[0]

        mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
        print("Numba smoke")
        print("-----------")
        print(f"count_elements(ndivs=4): {count_elements(mesh.elements)}")
        print()

    if status.jax_available:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def radius_norm(vertices):
            return jnp.max(jnp.abs(jnp.linalg.norm(vertices, axis=1) - 1.0))

        mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
        err = radius_norm(jnp.asarray(mesh.vertices))

        print("JAX smoke")
        print("---------")
        print(f"radius norm error(ndivs=4): {float(err):.6e}")


if __name__ == "__main__":
    main()
