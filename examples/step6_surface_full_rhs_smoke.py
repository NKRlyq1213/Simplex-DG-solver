from __future__ import annotations

import numpy as np

from simplex_dg.backends import backend_status
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import (
    build_full_rhs_cache,
    full_rhs_split,
    surface_lift_correction,
    volume_divergence_split,
)
from simplex_dg.trace import build_trace_cache, pair_face_traces


def main() -> None:
    status = backend_status()

    print("Backend status")
    print("--------------")
    print(f"Numba available: {status.numba_available}")
    print(f"JAX available  : {status.jax_available}")
    print(f"JAX devices    : {status.jax_devices}")
    print()

    ref = build_reference_cache(order=4, table="table1")
    mesh = build_octa_sphere_mesh(ndivs=4, radius=1.0)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=(0.0, 0.0, 1.0),
        flux_type="upwind",
    )

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]

    traces = pair_face_traces(q, trace)
    div = volume_divergence_split(q, full.volume)
    surf = surface_lift_correction(traces, full.surface)
    rhs = full_rhs_split(q, full)

    print("Full RHS cache")
    print("--------------")
    print(f"K                        : {mesh.elements.shape[0]}")
    print(f"Np                       : {ref.rs.shape[0]}")
    print(f"Nf                       : {ref.edge_rules[1].n_points}")
    print(f"flux type                : {full.surface.flux_type}")
    print(f"max speed volume         : {full.volume.max_speed:.6e}")
    print(f"normal velocity min/max  : {full.surface.normal_velocity.min():+.6e}, {full.surface.normal_velocity.max():+.6e}")
    print(f"lift shape               : {full.surface.lift.shape}")
    print()

    print("Operator output")
    print("---------------")
    print(f"q min/max                : {q.min():+.6e}, {q.max():+.6e}")
    print(f"volume div min/max       : {div.min():+.6e}, {div.max():+.6e}")
    print(f"surface corr min/max     : {surf.min():+.6e}, {surf.max():+.6e}")
    print(f"full rhs min/max         : {rhs.min():+.6e}, {rhs.max():+.6e}")
    print(f"rhs - (-div+surf) max abs: {np.max(np.abs(rhs - (-div + surf))):.6e}")

    if status.numba_available:
        rhs_nb = full_rhs_split(q, full, use_numba=True)
        diff = np.max(np.abs(rhs_nb - rhs))

        print()
        print("Numba smoke")
        print("-----------")
        print(f"max numpy/numba full rhs diff: {diff:.6e}")

    if status.jax_available:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def face_flux_norm(un, qM):
            return jnp.linalg.norm(un * qM)

        val = face_flux_norm(
            jnp.asarray(full.surface.normal_velocity),
            jnp.asarray(traces.qM),
        )

        print()
        print("JAX smoke")
        print("---------")
        print(f"face flux norm: {float(val):.6e}")


if __name__ == "__main__":
    main()
