from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplex_dg.backends import backend_status
from simplex_dg.diagnostics import error_report
from simplex_dg.geometry import build_geometry_cache
from simplex_dg.mesh import build_connectivity_cache_from_mesh, build_octa_sphere_mesh
from simplex_dg.problems import (
    exact_gaussian_solid_body,
    gaussian_center_solid_body,
    gaussian_on_sphere,
)
from simplex_dg.reference import build_reference_cache
from simplex_dg.rhs import build_full_rhs_cache, full_rhs
from simplex_dg.time import (
    cfl_dt_from_geometry,
    integrate_lsrk54,
    manifold_integral,
    manifold_l2_norm,
)
from simplex_dg.trace import build_trace_cache


def main() -> None:
    status = backend_status()

    print("Backend status")
    print("--------------")
    print(f"Numba available: {status.numba_available}")
    print(f"JAX available  : {status.jax_available}")
    print(f"JAX devices    : {status.jax_devices}")
    print()

    radius = 1.0
    sigma = 0.35
    omega = (0.0, 0.0, 1.0)

    ref = build_reference_cache(order=3, table="table1")
    mesh = build_octa_sphere_mesh(ndivs=2, radius=radius)
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=omega,
        flux_type="upwind",
    )

    center0 = (radius, 0.0, 0.0)

    q0 = gaussian_on_sphere(
        X=geom.X,
        center=center0,
        radius=radius,
        sigma=sigma,
        amplitude=1.0,
    )

    dt = cfl_dt_from_geometry(
        ref=ref,
        geom=geom,
        max_speed=full.volume.max_speed,
        cfl=1.0,
    )

    tf = 5.0

    def rhs(t, q):
        return full_rhs(q, full, use_numba=True)

    result = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=0.0,
        tf=tf,
        dt=dt,
        monitor=None,
    )

    q_exact = exact_gaussian_solid_body(
        X=geom.X,
        t=tf,
        radius=radius,
        sigma=sigma,
        amplitude=1.0,
        center0=center0,
        omega=omega,
    )

    rep = error_report(result.q, q_exact, ref, geom)

    center_t = gaussian_center_solid_body(
        t=tf,
        radius=radius,
        center0=center0,
        omega=omega,
    )

    mass0 = manifold_integral(q0, ref, geom)
    massf = manifold_integral(result.q, ref, geom)

    l20 = manifold_l2_norm(q0, ref, geom)
    l2f = manifold_l2_norm(result.q, ref, geom)

    print("Gaussian solid-body advection")
    print("-----------------------------")
    print(f"K                    : {mesh.elements.shape[0]}")
    print(f"Np                   : {ref.rs.shape[0]}")
    print(f"initial center       : ({center0[0]:+.6e}, {center0[1]:+.6e}, {center0[2]:+.6e})")
    print(f"exact center(tf)     : ({center_t[0]:+.6e}, {center_t[1]:+.6e}, {center_t[2]:+.6e})")
    print(f"sigma                : {sigma:.6e}")
    print(f"volume form          : {full.volume_form}")
    print(f"dt                   : {dt:.6e}")
    print(f"tf                   : {tf:.6e}")
    print(f"nsteps               : {result.nsteps}")
    print()

    print("Conservation / norm")
    print("-------------------")
    print(f"mass0                : {mass0:+.12e}")
    print(f"massf                : {massf:+.12e}")
    print(f"mass drift           : {massf - mass0:+.12e}")
    print(f"L2 norm initial      : {l20:.12e}")
    print(f"L2 norm final        : {l2f:.12e}")
    print(f"L2 norm drift        : {l2f - l20:+.12e}")
    print()

    print("Exact-solution error")
    print("--------------------")
    print(f"L2 error             : {rep.l2_error:.12e}")
    print(f"relative L2 error    : {rep.relative_l2_error:.12e}")
    print(f"Linf error           : {rep.linf_error:.12e}")
    print(f"mass error vs exact  : {rep.mass_error:+.12e}")
    print(f"L2 exact norm        : {rep.l2_norm_exact:.12e}")
    print(f"L2 numerical norm    : {rep.l2_norm_numerical:.12e}")

    if status.jax_available:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def state_max(q):
            return jnp.max(q)

        val = state_max(jnp.asarray(result.q))

        print()
        print("JAX smoke")
        print("---------")
        print(f"max(q_final): {float(val):.6e}")


if __name__ == "__main__":
    main()
