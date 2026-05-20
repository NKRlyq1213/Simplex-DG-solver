from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import roots_legendre


REFERENCE_AREA = 2.0


@dataclass(frozen=True)
class TriangleRule:
    table: str
    order: int
    bary_raw: np.ndarray
    rs: np.ndarray
    weights: np.ndarray
    edge_weights: np.ndarray | None


@dataclass(frozen=True)
class EdgeRule:
    edge_id: int
    n_points: int
    t01: np.ndarray
    weights: np.ndarray
    rs: np.ndarray
    length: float


TABLE1_RAW: dict[int, list[dict[str, float | str | None]]] = {
    1: [
        {"sym": "S6", "b1": 0.2113248654051871, "b2": 0.0, "ws": 0.16666666666666667, "we": 0.5000000000000000},
    ],
    2: [
        {"sym": "S6", "b1": 0.1127016653792583, "b2": 0.0, "ws": 0.04166666666666666, "we": 0.2777777777777777},
        {"sym": "S3", "b1": 0.5000000000000000, "b2": 0.0, "ws": 0.09999999999999999, "we": 0.4444444444444444},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.45000000000000000, "we": None},
    ],
    3: [
        {"sym": "S6", "b1": 0.06943184420297367, "b2": 0.0, "ws": 0.01509901487256561, "we": 0.1739274225687269},
        {"sym": "S6", "b1": 0.3300094782075718, "b2": 0.0, "ws": 0.04045654068298990, "we": 0.3260725774312731},
        {"sym": "S6", "b1": 0.5841571139756568, "b2": 0.1870738791912763, "ws": 0.11111111111111111, "we": None},
    ],
    4: [
        {"sym": "S6", "b1": 0.04691007703066797, "b2": 0.0, "ws": 0.006601315081001592, "we": 0.1184634425280944},
        {"sym": "S6", "b1": 0.2307653449471584, "b2": 0.0, "ws": 0.02053045968042892, "we": 0.2393143352496833},
        {"sym": "S3", "b1": 0.5000000000000000, "b2": 0.0, "ws": 0.01853708483394990, "we": 0.2844444444444446},
        {"sym": "S3", "b1": 0.1394337314154536, "b2": 0.1394337314154536, "ws": 0.10542932962084440, "we": None},
        {"sym": "S3", "b1": 0.4384239524408185, "b2": 0.4384239524408185, "ws": 0.12473673228977350, "we": None},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.09109991119771331, "we": None},
    ],
}

TABLE2_RAW: dict[int, list[dict[str, float | str]]] = {
    1: [
        {"sym": "S3", "b1": 0.1666666666666666, "b2": 0.1666666666666666, "ws": 0.3333333333333333},
    ],
    2: [
        {"sym": "S3", "b1": 0.09157621350977067, "b2": 0.09157621350977067, "ws": 0.1099517436553218},
        {"sym": "S3", "b1": 0.4459484909159648, "b2": 0.4459484909159648, "ws": 0.2233815896780115},
    ],
    3: [
        {"sym": "S3", "b1": 0.2194299825497830, "b2": 0.2194299825497830, "ws": 0.1713331241529809},
        {"sym": "S3", "b1": 0.4801379641122150, "b2": 0.4801379641122150, "ws": 0.08073108959303095},
        {"sym": "S6", "b1": 0.1416190159239682, "b2": 0.0193717243612408, "ws": 0.04063455979366068},
    ],
    4: [
        {"sym": "S6", "b1": 0.7284923929554044, "b2": 0.2631128296346379, "ws": 0.02723031417443505},
        {"sym": "S3", "b1": 0.4592925882927232, "b2": 0.4592925882927232, "ws": 0.09509163426728455},
        {"sym": "S3", "b1": 0.1705693077517602, "b2": 0.1705693077517602, "ws": 0.1032173705347182},
        {"sym": "S3", "b1": 0.05054722831703096, "b2": 0.05054722831703096, "ws": 0.03245849762319804},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.1443156076777874},
    ],
}


_RAW_TO_REF_PERM = np.array([1, 2, 0], dtype=int)

_REF_VERTICES_RS = np.array(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
    ],
    dtype=float,
)


def _expected_count(sym: str) -> int:
    lookup = {"S1": 1, "S3": 3, "S6": 6}
    if sym not in lookup:
        raise ValueError(f"Unknown symmetry label: {sym}")
    return lookup[sym]


def _unique_permutations(values: tuple[float, float, float], ndigits: int = 15) -> np.ndarray:
    perms = set()

    for p in itertools.permutations(values):
        perms.add(tuple(round(v, ndigits) for v in p))

    return np.array(sorted(perms), dtype=float)


def _expand_row(sym: str, b1: float, b2: float) -> np.ndarray:
    b3 = 1.0 - b1 - b2
    bary = _unique_permutations((b1, b2, b3))
    expected = _expected_count(sym)

    if bary.shape[0] != expected:
        raise ValueError(
            f"Symmetry expansion mismatch: sym={sym}, got={bary.shape[0]}, expected={expected}"
        )

    return bary


def raw_barycentric_to_reference_rs(bary_raw: np.ndarray) -> np.ndarray:
    bary_raw = np.asarray(bary_raw, dtype=float)

    if bary_raw.shape[-1] != 3:
        raise ValueError("bary_raw must have last dimension 3.")

    bary_ref = bary_raw[..., _RAW_TO_REF_PERM]

    return bary_ref @ _REF_VERTICES_RS


def load_triangle_rule(table: str, order: int) -> TriangleRule:
    table_norm = table.lower().strip()

    if table_norm == "table1":
        raw = TABLE1_RAW
        has_edge_weights = True
    elif table_norm == "table2":
        raw = TABLE2_RAW
        has_edge_weights = False
    else:
        raise ValueError("table must be either 'table1' or 'table2'.")

    if order not in raw:
        raise KeyError(f"{table_norm} order {order} is not available.")

    bary_all = []
    ws_all = []
    we_all = []

    for row in raw[order]:
        sym = str(row["sym"])
        b1 = float(row["b1"])
        b2 = float(row["b2"])
        ws = float(row["ws"])

        bary_row = _expand_row(sym, b1, b2)

        bary_all.append(bary_row)
        ws_all.append(np.full(bary_row.shape[0], ws, dtype=float))

        if has_edge_weights:
            we = np.nan if row["we"] is None else float(row["we"])
            we_all.append(np.full(bary_row.shape[0], we, dtype=float))

    bary = np.vstack(bary_all)
    weights = np.concatenate(ws_all)
    edge_weights = np.concatenate(we_all) if has_edge_weights else None
    rs = raw_barycentric_to_reference_rs(bary)

    return TriangleRule(
        table=table_norm,
        order=order,
        bary_raw=bary,
        rs=rs,
        weights=weights,
        edge_weights=edge_weights,
    )


def reference_edge_nodes(edge_id: int, t01: np.ndarray) -> np.ndarray:
    t01 = np.asarray(t01, dtype=float).reshape(-1)

    if edge_id == 1:
        r = 1.0 - 2.0 * t01
        s = -1.0 + 2.0 * t01
    elif edge_id == 2:
        r = -np.ones_like(t01)
        s = 1.0 - 2.0 * t01
    elif edge_id == 3:
        r = -1.0 + 2.0 * t01
        s = -np.ones_like(t01)
    else:
        raise ValueError("edge_id must be 1, 2, or 3.")

    return np.column_stack([r, s])


def edge_length_reference(edge_id: int) -> float:
    if edge_id == 1:
        return float(2.0 * np.sqrt(2.0))
    if edge_id in (2, 3):
        return 2.0

    raise ValueError("edge_id must be 1, 2, or 3.")


def edge_gl_rule(edge_id: int, n_points: int) -> EdgeRule:
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    x, w = roots_legendre(n_points)

    t01 = 0.5 * (x + 1.0)
    weights = 0.5 * w
    rs = reference_edge_nodes(edge_id, t01)

    return EdgeRule(
        edge_id=edge_id,
        n_points=n_points,
        t01=np.asarray(t01, dtype=float),
        weights=np.asarray(weights, dtype=float),
        rs=rs,
        length=edge_length_reference(edge_id),
    )