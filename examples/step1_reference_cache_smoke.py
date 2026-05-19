from __future__ import annotations

import numpy as np

from simplex_dg.backends import backend_status
from simplex_dg.reference import build_reference_cache


def main() -> None:
    status = backend_status()

    print("Backend status")
    print("--------------")
    print(f"Numba available: {status.numba_available}")
    print(f"JAX available  : {status.jax_available}")
    print(f"JAX devices    : {status.jax_devices}")
    print()

    cache = build_reference_cache(order=4, table="table1")

    print("Reference cache")
    print("---------------")
    print(f"order       : {cache.order}")
    print(f"table       : {cache.table}")
    print(f"Np          : {cache.rs.shape[0]}")
    print(f"Nmodes      : {cache.V.shape[1]}")
    print(f"V shape     : {cache.V.shape}")
    print(f"Dr shape    : {cache.Dr.shape}")
    print(f"Ds shape    : {cache.Ds.shape}")
    print(f"cond(M)     : {np.linalg.cond(cache.M):.6e}")

    q = np.sin(np.pi * cache.rs[:, 0]) * np.cos(np.pi * cache.rs[:, 1])
    qr = cache.Dr @ q
    qs = cache.Ds @ q

    print()
    print("Operator smoke")
    print("--------------")
    print(f"q min/max    : {q.min():+.6e}, {q.max():+.6e}")
    print(f"qr min/max   : {qr.min():+.6e}, {qr.max():+.6e}")
    print(f"qs min/max   : {qs.min():+.6e}, {qs.max():+.6e}")

    if status.numba_available:
        import numba

        @numba.njit(cache=True)
        def matvec_numba(A, x):
            return A @ x

        y = matvec_numba(cache.Dr, q)

        print()
        print("Numba smoke")
        print("-----------")
        print(f"numba Dr@q norm: {np.linalg.norm(y):.6e}")

    if status.jax_available:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def matvec_jax(A, x):
            return A @ x

        y = matvec_jax(jnp.asarray(cache.Dr), jnp.asarray(q))

        print()
        print("JAX smoke")
        print("---------")
        print(f"jax Dr@q norm  : {float(jnp.linalg.norm(y)):.6e}")


if __name__ == "__main__":
    main()