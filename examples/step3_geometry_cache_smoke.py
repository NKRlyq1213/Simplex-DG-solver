from __future__ import annotations

import numpy as np

from simplex_dg.backends import backend_status
from simplex_dg.geometry import build_geometry_cache, dual_basis_residuals
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache


def main() -> None:
    status = backend_status()

    print("Backend status")
    print("--------------")
    print(f"Numba available: {status.numba_available}")
    print(f"JAX available  : {status.jax_available}")
    print(f"JAX devices    : {status.jax_devices}")
    print()

    ref = build_reference_cache(order=4, table="table1")
    mesh = build_octa_sphere_mesh(level=2, radius=1.0)
    geom = build_geometry_cache(mesh, ref)

    print("Geometry cache")
    print("--------------")
    print(f"K                    : {mesh.elements.shape[0]}")
    print(f"Np                   : {ref.rs.shape[0]}")
    print(f"Nf                   : {ref.edge_rules[1].n_points}")
    print(f"X shape              : {geom.X.shape}")
    print(f"X_face shape         : {geom.X_face.shape}")
    print(f"sqrt_g min/max       : {geom.sqrt_g.min():.6e}, {geom.sqrt_g.max():.6e}")
    print(f"face jac min/max     : {geom.face_jacobian.min():.6e}, {geom.face_jacobian.max():.6e}")

    radius_error = np.max(np.abs(np.linalg.norm(geom.X, axis=2) - mesh.radius))
    normal_error = np.max(np.abs(np.linalg.norm(geom.normal, axis=2) - 1.0))

    print(f"max radius error     : {radius_error:.6e}")
    print(f"max normal error     : {normal_error:.6e}")

    print()
    print("Dual basis residuals")
    print("--------------------")
    for k, v in dual_basis_residuals(geom).items():
        print(f"{k}: {v:.6e}")

    if status.numba_available:
        import numba

        @numba.njit(cache=True)
        def min_value(x):
            return np.min(x)

        print()
        print("Numba smoke")
        print("-----------")
        print(f"min sqrt_g: {min_value(geom.sqrt_g):.6e}")

    if status.jax_available:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def max_radius_error_jax(X):
            r = jnp.linalg.norm(X, axis=2)
            return jnp.max(jnp.abs(r - 1.0))

        err = max_radius_error_jax(jnp.asarray(geom.X))

        print()
        print("JAX smoke")
        print("---------")
        print(f"max radius error: {float(err):.6e}")


if __name__ == "__main__":
    main()