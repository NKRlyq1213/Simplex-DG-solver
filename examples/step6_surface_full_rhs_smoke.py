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
from simplex_dg.rhs import (
    build_lift_matrices,
    build_full_rhs_cache,
    common_projected_line_velocity,
    full_rhs,
    numerical_flux,
    projected_interior_line_flux,
    projected_line_velocity,
    surface_lift_correction_projected_flux,
    surface_lift_correction_split_projected_flux,
    volume_divergence_conservative,
    volume_divergence_split,
)
from simplex_dg.time import manifold_integral
from simplex_dg.trace import build_trace_cache, pair_face_traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the step6 projected-SBP surface RHS operators on the sphere."
    )
    parser.add_argument("--table", type=str, default="table1", choices=["table1", "table2"])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--ndivs", type=int, default=4)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--flux-type", type=str, default="upwind", choices=["central", "upwind", "lf"])
    parser.add_argument("--lf-alpha", type=float, default=1.0)
    parser.add_argument("--volume-form", type=str, default="conservative", choices=["conservative", "split"])
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
    conn = build_connectivity_cache_from_mesh(mesh)
    geom = build_geometry_cache(mesh, ref)
    trace = build_trace_cache(ref, conn)

    full = build_full_rhs_cache(
        ref=ref,
        geom=geom,
        trace=trace,
        omega=(0.0, 0.0, 1.0),
        flux_type=args.flux_type,
        lf_alpha=args.lf_alpha,
        volume_form=args.volume_form,
    )

    q = geom.X[:, :, 0] + 0.25 * geom.X[:, :, 1] - 0.5 * geom.X[:, :, 2]
    ones = np.ones_like(q)
    q_random = np.random.default_rng(12345).standard_normal(q.shape)

    traces = pair_face_traces(q, trace)
    if full.volume_form == "conservative":
        div = volume_divergence_conservative(q, full.volume)
        surf = surface_lift_correction_projected_flux(q, traces, full.volume, full.surface, full.trace)
    else:
        div = volume_divergence_split(q, full.volume)
        surf = surface_lift_correction_split_projected_flux(q, traces, full.volume, full.surface, full.trace)
    rhs = full_rhs(q, full)

    line_velocity = projected_line_velocity(full.volume, full.surface)
    common_velocity = common_projected_line_velocity(full.volume, full.surface, full.trace, use_numba=False)
    line_flux = projected_interior_line_flux(q, full.volume, full.surface)
    flux_star = numerical_flux(
        traces.qM,
        traces.qP,
        common_velocity,
        full.surface.flux_id,
        lf_alpha=full.surface.lf_alpha,
    )

    lift_direct_error = 0.0
    lift_adjoint_error = 0.0
    for face_id in (1, 2, 3):
        f = face_id - 1
        edge = ref.edge_rules[face_id]
        expected_lift = build_lift_matrices(ref, trace)[f]
        lift_direct_error = max(lift_direct_error, float(np.max(np.abs(full.surface.lift[f] - expected_lift))))

        q_volume = np.linspace(-0.5, 0.5, ref.rs.shape[0])
        p_face = np.linspace(0.25, 1.25, edge.n_points)
        lhs = ref.area * np.dot(ref.weights * q_volume, full.surface.lift[f] @ p_face)
        rhs_adj = np.dot(edge.weights * (full.surface.face_interp[f] @ q_volume), p_face)
        lift_adjoint_error = max(lift_adjoint_error, abs(float(lhs - rhs_adj)))

    face_velocity_exact = np.cross(np.broadcast_to(np.array([0.0, 0.0, 1.0]), geom.X_face.shape), geom.X_face)
    face_velocity_tangency = float(np.max(np.abs(np.sum(full.surface.face_velocity * geom.face_normal, axis=3))))
    normal_velocity_error = float(
        np.max(np.abs(full.surface.normal_velocity - np.sum(full.surface.face_velocity * geom.face_conormal, axis=3)))
    )
    projected_product_wrong = np.empty_like(line_flux)
    for f in range(full.surface.n_faces):
        alpha_face = full.volume.alpha @ full.surface.face_interp[f].T
        beta_face = full.volume.beta @ full.surface.face_interp[f].T
        q_face = q @ full.surface.face_interp[f].T
        if f == 0:
            drdt, dsdt = -2.0, 2.0
        elif f == 1:
            drdt, dsdt = 0.0, -2.0
        else:
            drdt, dsdt = 2.0, 0.0
        projected_product_wrong[:, f, :] = dsdt * alpha_face * q_face - drdt * beta_face * q_face
    ordering_gap = float(np.max(np.abs(line_flux - projected_product_wrong)))

    common_interface_sum = 0.0
    flux_interface_sum = 0.0
    for k, f, nbr, nbr_f in conn.interior_faces:
        common_plus = common_velocity[nbr, nbr_f]
        flux_plus = flux_star[nbr, nbr_f]
        if trace.face_flip[k, f]:
            common_plus = common_plus[::-1]
            flux_plus = flux_plus[::-1]
        common_interface_sum = max(common_interface_sum, float(np.max(np.abs(common_velocity[k, f] + common_plus))))
        flux_interface_sum = max(flux_interface_sum, float(np.max(np.abs(flux_star[k, f] + flux_plus))))

    mass_residual = float(abs(np.sum(ref.area * ref.weights[None, :] * geom.sqrt_g * rhs)))
    rhs_const = full_rhs(ones, full, use_numba=False)
    const_linf = float(np.max(np.abs(rhs_const)))
    const_l2 = float(np.sqrt(np.sum(ref.area * ref.weights[None, :] * geom.sqrt_g * rhs_const * rhs_const)))
    const_global = float(abs(manifold_integral(rhs_const, ref, geom)))
    rhs_random = full_rhs(q_random, full, use_numba=False)
    random_mass_residual = float(abs(manifold_integral(rhs_random, ref, geom)))
    random_mass_scale = max(float(manifold_integral(np.abs(q_random), ref, geom)), 1.0)
    direct_composition_error = float(np.max(np.abs(rhs - (-div + surf))))

    print("Full RHS cache")
    print("--------------")
    print(f"table                    : {args.table}")
    print(f"order                    : {args.order}")
    print(f"ndivs                    : {args.ndivs}")
    print(f"radius                   : {args.radius:.6e}")
    print(f"K                        : {mesh.elements.shape[0]}")
    print(f"Np                       : {ref.rs.shape[0]}")
    print(f"Nf                       : {ref.edge_rules[1].n_points}")
    print(f"flux type                : {full.surface.flux_type}")
    print(f"lf alpha                 : {full.surface.lf_alpha:.6e}")
    print(f"volume form              : {full.volume_form}")
    print(f"max speed volume         : {full.volume.max_speed:.6e}")
    print(f"normal velocity min/max  : {full.surface.normal_velocity.min():+.6e}, {full.surface.normal_velocity.max():+.6e}")
    print(f"lift shape               : {full.surface.lift.shape}")
    print()

    print("Operator output")
    print("---------------")
    print(f"q min/max                : {q.min():+.6e}, {q.max():+.6e}")
    print(f"volume div min/max       : {div.min():+.6e}, {div.max():+.6e}")
    print(f"projected line vel min/max: {line_velocity.min():+.6e}, {line_velocity.max():+.6e}")
    print(f"surface corr min/max     : {surf.min():+.6e}, {surf.max():+.6e}")
    print(f"full rhs min/max         : {rhs.min():+.6e}, {rhs.max():+.6e}")
    print(f"rhs - (-div+surf) max abs: {direct_composition_error:.6e}")
    print(f"lift direct error        : {lift_direct_error:.6e}")
    print(f"lift adjoint error       : {lift_adjoint_error:.6e}")
    print(f"face velocity def error  : {np.max(np.abs(full.surface.face_velocity - face_velocity_exact)):.6e}")
    print(f"face velocity tangent err: {face_velocity_tangency:.6e}")
    print(f"normal velocity def error: {normal_velocity_error:.6e}")
    print(f"common line vel intf sum : {common_interface_sum:.6e}")
    print(f"projected ordering gap   : {ordering_gap:.6e}")
    print(f"numerical flux intf sum  : {flux_interface_sum:.6e}")
    print(f"global mass residual     : {mass_residual:.6e}")
    print(f"constant-state Linf      : {const_linf:.6e}")
    print(f"constant-state physical L2: {const_l2:.6e}")
    print(f"constant-state global int: {const_global:.6e}")
    print(f"random-state scaled mass : {(random_mass_residual / random_mass_scale):.6e}")

    if status.numba_available:
        rhs_nb = full_rhs(q, full, use_numba=True)
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
