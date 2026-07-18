from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplex_dg.backends import backend_status
from simplex_dg.geometry import build_geometry_cache, dual_basis_residuals
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry cache smoke test.")
    parser.add_argument("--table", type=str, default="table1", choices=["table1", "table2"])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--ndivs", type=int, default=4)
    parser.add_argument("--radius", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = backend_status()

    print("Backend status")
    print("--------------")
    print(f"Numba available: {status.numba_available}")
    print(f"JAX available  : {status.jax_available}")
    print(f"JAX devices    : {status.jax_devices}")
    print()

    ref = build_reference_cache(order=args.order, table=args.table)
    mesh = build_octa_sphere_mesh(ndivs=args.ndivs, radius=args.radius)
    geom = build_geometry_cache(mesh, ref)

    print("Geometry cache")
    print("--------------")
    print(f"table                : {ref.table}")
    print(f"order                : {ref.order}")
    print(f"ndivs                : {mesh.ndivs}")
    print(f"radius               : {mesh.radius}")
    print(f"K                    : {mesh.elements.shape[0]}")
    print(f"Np                   : {ref.rs.shape[0]}")
    print(f"Nf                   : {ref.edge_rules[1].n_points}")
    print(f"X shape              : {geom.X.shape}")
    print(f"X_face shape         : {geom.X_face.shape}")
    print(f"sqrt_g min/max       : {geom.sqrt_g.min():.6e}, {geom.sqrt_g.max():.6e}")
    print(f"face jac min/max     : {geom.face_jacobian.min():.6e}, {geom.face_jacobian.max():.6e}")

    volume_radius_error = np.max(np.abs(np.linalg.norm(geom.X, axis=2) - mesh.radius))
    face_radius_error = np.max(np.abs(np.linalg.norm(geom.X_face, axis=3) - mesh.radius))
    sqrt_g_cross = np.linalg.norm(np.cross(geom.Xr, geom.Xs), axis=2)
    surface_jacobian_error = np.max(np.abs(geom.sqrt_g - sqrt_g_cross))
    face_jacobian_from_tangent = np.linalg.norm(geom.face_tangent, axis=3)
    face_jacobian_error = np.max(np.abs(geom.face_jacobian - face_jacobian_from_tangent))

    print(f"max volume radius error: {volume_radius_error:.6e}")
    print(f"max face radius error  : {face_radius_error:.6e}")
    print(f"max surface jac error  : {surface_jacobian_error:.6e}")
    print(f"max face jac error     : {face_jacobian_error:.6e}")

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
