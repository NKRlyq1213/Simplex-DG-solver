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
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.trace import (
    build_trace_cache,
    check_constant_trace_consistency,
    max_interior_trace_mismatch,
    pair_face_traces,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace exchange smoke test.")
    parser.add_argument("--table", type=str, default="table1", choices=["table1", "table2"])
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

    ref = build_reference_cache(order=4, table=args.table)
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    print("Trace cache")
    print("-----------")
    print(f"K                  : {trace.n_elements}")
    print(f"Np                 : {trace.n_points}")
    print(f"Nf                 : {trace.n_face_points}")
    print(f"table              : {ref.table}")
    print(f"face_interp shape  : {trace.face_interp.shape}")
    print(f"boundary faces     : {np.count_nonzero(trace.is_boundary)}")
    print(f"face flips         : {np.count_nonzero(trace.face_flip)}")
    print()

    const_mismatch = check_constant_trace_consistency(trace, value=1.0)

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]
    traces = pair_face_traces(q, trace)

    smooth_mismatch = max_interior_trace_mismatch(traces, trace)

    print("Trace consistency")
    print("-----------------")
    print(f"constant mismatch : {const_mismatch:.6e}")
    print(f"smooth mismatch   : {smooth_mismatch:.6e}")
    print(f"qM min/max        : {traces.qM.min():+.6e}, {traces.qM.max():+.6e}")
    print(f"qP min/max        : {traces.qP.min():+.6e}, {traces.qP.max():+.6e}")
    print()

    if status.numba_available:
        traces_nb = pair_face_traces(q, trace, use_numba=True)
        diff = max(
            np.max(np.abs(traces_nb.qM - traces.qM)),
            np.max(np.abs(traces_nb.qP - traces.qP)),
        )

        print("Numba smoke")
        print("-----------")
        print(f"max numpy/numba diff: {diff:.6e}")
        print()

    if status.jax_available:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def eval_face0(q_elem, E0):
            return E0 @ q_elem

        y = eval_face0(jnp.asarray(q[0]), jnp.asarray(trace.face_interp[0]))

        print("JAX smoke")
        print("---------")
        print(f"face0 trace norm: {float(jnp.linalg.norm(y)):.6e}")


if __name__ == "__main__":
    main()
