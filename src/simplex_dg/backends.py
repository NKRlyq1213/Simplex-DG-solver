from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendStatus:
    numba_available: bool
    jax_available: bool
    jax_devices: tuple[str, ...]


def backend_status() -> BackendStatus:
    try:
        import numba  # noqa: F401

        numba_available = True
    except Exception:
        numba_available = False

    try:
        import jax

        jax_available = True
        devices = tuple(str(d) for d in jax.devices())
    except Exception:
        jax_available = False
        devices = tuple()

    return BackendStatus(
        numba_available=numba_available,
        jax_available=jax_available,
        jax_devices=devices,
    )


def require_numba():
    try:
        import numba

        return numba
    except Exception as exc:
        raise RuntimeError("Numba is not available.") from exc


def require_jax():
    try:
        import jax
        import jax.numpy as jnp

        return jax, jnp
    except Exception as exc:
        raise RuntimeError("JAX is not available.") from exc