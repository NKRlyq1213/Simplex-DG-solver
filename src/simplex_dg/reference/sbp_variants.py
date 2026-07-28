from __future__ import annotations

from typing import Literal


SBPVariant = Literal["projected", "full-raw", "full-orth"]


_ALLOWED_SBP_VARIANTS = ("projected", "full-raw", "full-orth")
_FULL_SBP_CONSTRUCTIONS = {
    "full-raw": "raw",
    "full-orth": "orthogonalized",
}


def normalize_sbp_variant(sbp_variant: str) -> SBPVariant:
    if not isinstance(sbp_variant, str):
        raise ValueError("sbp_variant must be 'projected', 'full-raw', or 'full-orth'.")

    normalized = sbp_variant.lower().strip()

    if normalized not in _ALLOWED_SBP_VARIANTS:
        raise ValueError("sbp_variant must be 'projected', 'full-raw', or 'full-orth'.")

    return normalized  # type: ignore[return-value]


def is_full_sbp_variant(sbp_variant: SBPVariant) -> bool:
    return sbp_variant in _FULL_SBP_CONSTRUCTIONS


def boundary_representation_for_variant(sbp_variant: SBPVariant) -> Literal["projected", "direct"]:
    if sbp_variant == "projected":
        return "projected"

    return "direct"


def full_sbp_construction_for_variant(
    sbp_variant: SBPVariant,
) -> Literal["raw", "orthogonalized"] | None:
    return _FULL_SBP_CONSTRUCTIONS.get(sbp_variant)
