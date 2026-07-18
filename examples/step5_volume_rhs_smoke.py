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
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    build_volume_rhs_cache,
    volume_divergence_conservative,
    volume_divergence_split,
    volume_rhs_conservative,
    volume_rhs_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the step5 volume RHS operators on the sphere."
    )
    parser.add_argument(
        "--table",
        type=str,
        default="table1",
        choices=["table1", "table2"],
        help="Triangle quadrature table. Default: table1.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=4,
        help="Polynomial order. Default: 4.",
    )
    parser.add_argument(
        "--ndivs",
        type=int,
        default=4,
        help="Octahedron subdivision count. Default: 4.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Sphere radius. Default: 1.0.",
    )
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
    rhs_cache = build_volume_rhs_cache(ref, geom, omega=(0.0, 0.0, 1.0))

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    div_split = volume_divergence_split(q, rhs_cache)
    div_cons = volume_divergence_conservative(q, rhs_cache)
    rhs_split = volume_rhs_split(q, rhs_cache)
    rhs_cons = volume_rhs_conservative(q, rhs_cache)

    tangent_error = np.max(np.abs(np.sum(rhs_cache.velocity * geom.normal, axis=2)))
    ur = rhs_cache.alpha / rhs_cache.sqrt_g
    us = rhs_cache.beta / rhs_cache.sqrt_g
    velocity_reconstructed = ur[..., None] * geom.Xr + us[..., None] * geom.Xs
    recon_error = np.max(np.abs(velocity_reconstructed - rhs_cache.velocity))

    print("Volume RHS cache")
    print("----------------")
    print(f"table                   : {args.table}")
    print(f"order                   : {args.order}")
    print(f"ndivs                   : {args.ndivs}")
    print(f"radius                  : {args.radius:.6e}")
    print(f"K                       : {rhs_cache.n_elements}")
    print(f"Np                      : {rhs_cache.n_points}")
    print(f"max speed               : {rhs_cache.max_speed:.6e}")
    print(f"tangent error           : {tangent_error:.6e}")
    print(f"contravariant recon err : {recon_error:.6e}")
    print(f"alpha min/max           : {rhs_cache.alpha.min():+.6e}, {rhs_cache.alpha.max():+.6e}")
    print(f"beta min/max            : {rhs_cache.beta.min():+.6e}, {rhs_cache.beta.max():+.6e}")
    print(f"metric-div min/max      : {rhs_cache.div_velocity.min():+.6e}, {rhs_cache.div_velocity.max():+.6e}")
    print()

    print("Operator output")
    print("---------------")
    print(f"q min/max               : {q.min():+.6e}, {q.max():+.6e}")
    print(f"split div min/max       : {div_split.min():+.6e}, {div_split.max():+.6e}")
    print(f"conservative div min/max: {div_cons.min():+.6e}, {div_cons.max():+.6e}")
    print(f"split rhs min/max       : {rhs_split.min():+.6e}, {rhs_split.max():+.6e}")
    print(f"cons rhs min/max        : {rhs_cons.min():+.6e}, {rhs_cons.max():+.6e}")
    print(f"max |rhs_split + div|   : {np.max(np.abs(rhs_split + div_split)):.6e}")
    print(f"max |rhs_cons + div|    : {np.max(np.abs(rhs_cons + div_cons)):.6e}")

    if status.numba_available:
        cons_nb = volume_divergence_conservative(q, rhs_cache, use_numba=True)
        div_nb = volume_divergence_split(q, rhs_cache, use_numba=True)
        split_diff = np.max(np.abs(div_nb - div_split))
        cons_diff = np.max(np.abs(cons_nb - div_cons))

        print()
        print("Numba smoke")
        print("-----------")
        print(f"max numpy/numba split diff: {split_diff:.6e}")
        print(f"max numpy/numba cons diff : {cons_diff:.6e}")

    if status.jax_available:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def reference_matvec(D, x):
            return D @ x

        y = reference_matvec(jnp.asarray(rhs_cache.Dr), jnp.asarray(q[0]))

        print()
        print("JAX smoke")
        print("---------")
        print(f"Dr@q[0] norm: {float(jnp.linalg.norm(y)):.6e}")


if __name__ == "__main__":
    main()
