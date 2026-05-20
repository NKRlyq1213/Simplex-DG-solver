from __future__ import annotations


def num_modes_2d(order: int) -> int:
    if order < 0:
        raise ValueError("order must be >= 0")
    return (order + 1) * (order + 2) // 2


def mode_indices_2d(order: int) -> list[tuple[int, int]]:
    if order < 0:
        raise ValueError("order must be >= 0")

    out: list[tuple[int, int]] = []
    for total_degree in range(order + 1):
        for i in range(total_degree + 1):
            j = total_degree - i
            out.append((i, j))

    return out